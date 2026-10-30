# -*- coding: utf-8 -*-
"""
agents/task_agents/base_task_agent.py
=====================================
任务级专家智能体公共底座（阶段三 JSRL 中期检查版）。

职责：
- 构建扁平 BVR 任务环境（ActionV1 动作空间）+ 固定尺度观测包装；
- SB3 PPO 基线训练（小规模验证，不执行完整 1e7 步训练）；
- 统一 `get_action` → PolicyOutput(ActionV1)，预留 JSRL guide_action/guide_steps；
- 保存/加载（SB3 zip + ActionSpec JSON），推理一致性由测试保证。

四个任务级智能体仅声明 task_id/mission 与环境/训练配置差异，不重复实现协议。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO

from agents.action_adapter import coerce_action_v1, to_flat_action
from agents.policy_interface import (ACTION_V1_VERSION, SAFE_ACTION_V1,
                                     PolicyOutput, action_spec, apply_action_mask,
                                     clip_action_v1, compute_action_mask)
from agents.task_agents.mission_env import (MISSION_NAMES_ZH,
                                            FlatTaskMissionEnv, ScaledObservation,
                                            TaskMissionEnv, normalize_obs,
                                            validate_mission, wrap_observation)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

DEFAULT_TRAIN_CONFIG = {
    "algo": "PPO",
    "policy": "MlpPolicy",
    "net_arch": [128, 128],
    "activation": "tanh",
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "total_timesteps_full": 10_000_000,
    "validation_timesteps": 512,
}


class TaskDRLExpertAgent:
    """任务级专家智能体基类（PolicyProtocol 实现）。"""

    task_id: str | None = None
    mission: str | None = None
    algo_cls = PPO

    def __init__(self, env_config: dict | None = None, train_config: dict | None = None,
                 seed: int = 42, model_path: str | Path | None = None,
                 auto_build: bool = False):
        if self.task_id is None:
            raise ValueError("子类必须指定 task_id")
        self.mission = validate_mission(self.mission or self.task_id)
        self.task_id = self.mission
        self.env_config = dict(env_config or {})
        self.control_mode = str(self.env_config.get("control_mode", "assisted"))
        self.train_config = dict(DEFAULT_TRAIN_CONFIG)
        if train_config:
            self.train_config.update(train_config)
        self.seed = int(seed)
        self.model_path = Path(model_path) if model_path else (
            MODEL_DIR / f"task_{self.task_id}_ppo")
        self.env = None
        self.train_env = None
        self.model = None
        self.last_train_metrics: dict[str, Any] = {}
        self._last_obs = None
        if auto_build:
            self.build_model()

    # ------------------------------------------------------------------
    # 环境
    # ------------------------------------------------------------------
    def build_env(self) -> FlatTaskMissionEnv:
        """训练/推理用扁平环境（ActionV1 动作空间，原始 28 维观测）。"""
        if self.env is None:
            self.env = FlatTaskMissionEnv(mission=self.mission, config=self.env_config)
        return self.env

    def build_deploy_env(self) -> TaskMissionEnv:
        """行为树部署环境（Dict 动作，配合 to_env_action）。"""
        return TaskMissionEnv(mission=self.mission, config=self.env_config)

    def reset(self) -> None:
        self._last_obs = None

    def close(self):
        for env in (self.env, self.train_env):
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        self.env = None
        self.train_env = None

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    def build_model(self, env=None):
        """按统一配置构建 SB3 PPO；env 为扁平原始环境（内部自动加观测尺度层）。"""
        raw_env = env if env is not None else self.build_env()
        self.env = raw_env
        if isinstance(raw_env, ScaledObservation):
            wrapped = raw_env
        else:
            wrapped = wrap_observation(raw_env)
        self.train_env = wrapped

        act_name = self.train_config.get("activation", "tanh")
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[act_name]
        self.model = self.algo_cls(
            self.train_config.get("policy", "MlpPolicy"),
            wrapped,
            policy_kwargs={"net_arch": list(self.train_config["net_arch"]),
                           "activation_fn": act_fn},
            learning_rate=float(self.train_config["learning_rate"]),
            n_steps=int(self.train_config["n_steps"]),
            batch_size=int(self.train_config["batch_size"]),
            n_epochs=int(self.train_config["n_epochs"]),
            gamma=float(self.train_config["gamma"]),
            gae_lambda=float(self.train_config["gae_lambda"]),
            clip_range=float(self.train_config["clip_range"]),
            ent_coef=float(self.train_config["ent_coef"]),
            max_grad_norm=float(self.train_config["max_grad_norm"]),
            seed=self.seed,
            verbose=0,
        )
        return self.model

    def _ensure_model(self):
        if self.model is None:
            self.build_model()
        return self.model

    # ------------------------------------------------------------------
    # 统一推理协议
    # ------------------------------------------------------------------
    def action_spec(self) -> dict:
        spec = action_spec(self.task_id, self.control_mode)
        spec.update({
            "mission": self.mission,
            "mission_name_zh": MISSION_NAMES_ZH[self.mission],
            "observation": {"version": "bvr_obs_v1", "dim": 28,
                            "policy_input": "fixed_scale[-1,1]"},
            "algorithm": self.train_config.get("algo", "PPO"),
            "model_path": str(self.model_path),
        })
        return spec

    def get_action(self, obs, *, info: dict | None = None, deterministic: bool = True,
                   mask=None, guide_action=None, guide_steps: int = 0) -> PolicyOutput:
        """统一推理入口。

        - guide_steps > 0 且 guide_action 非空：返回 source="guide"（本轮仅透传占位）；
        - 否则由 PPO 策略输出 source="drl"；
        - 输出始终裁剪为合法 ActionV1，并按 ActionMaskV1 清零非法武器位。
        """
        mask_arr = (np.asarray(mask, dtype=bool).reshape(7) if mask is not None
                    else compute_action_mask(obs, info, self.env_config))
        source = "drl"
        meta: dict[str, Any] = {"deterministic": bool(deterministic)}

        if guide_steps and int(guide_steps) > 0 and guide_action is not None:
            action = coerce_action_v1(guide_action)
            source = "guide"
            meta["guide_steps"] = int(guide_steps)
        else:
            try:
                model = self._ensure_model()
                policy_obs = normalize_obs(obs)
                raw_action, _ = model.predict(policy_obs, deterministic=deterministic)
                action = clip_action_v1(raw_action)
            except Exception as exc:  # 行为树安全兜底：不中断整棵树
                action = SAFE_ACTION_V1.copy()
                source = "fallback"
                meta["error"] = f"{type(exc).__name__}: {exc}"

        action = apply_action_mask(action, mask_arr)
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return PolicyOutput(action=action, source=source, masked=mask_arr,
                            action_version=ACTION_V1_VERSION, task_id=self.task_id,
                            info=meta)

    def predict(self, obs, deterministic: bool = True):
        """SB3 风格轻量入口：只返回 ActionV1（兼容旧调用）。"""
        return self.get_action(obs, deterministic=deterministic).action

    # ------------------------------------------------------------------
    # 训练/评估
    # ------------------------------------------------------------------
    def train(self, timesteps: int | None = None, progress_bar: bool = False) -> dict:
        """小规模/完整训练共用入口；本轮验收仅跑 512~2048 步。"""
        self._ensure_model()
        ts = int(timesteps or self.train_config.get("validation_timesteps", 512))
        t0 = time.time()
        self.model.learn(total_timesteps=ts, progress_bar=progress_bar)
        wall = time.time() - t0
        metrics = self._collect_metrics()
        self.last_train_metrics = {
            "task_id": self.task_id, "timesteps": ts, "wall_s": round(wall, 3),
            "metrics": metrics,
            "all_finite": bool(all(np.isfinite(v) for v in metrics.values())),
        }
        return self.last_train_metrics

    def _collect_metrics(self) -> dict:
        values = {}
        logger = getattr(self.model, "logger", None)
        if logger is not None:
            for key, value in logger.name_to_value.items():
                try:
                    arr = np.asarray(value, dtype=float).reshape(-1)
                    if arr.size == 1:
                        values[str(key)] = float(arr[0])
                except (TypeError, ValueError):
                    continue
        return values

    def rollout(self, n_steps: int = 64, deterministic: bool = True,
                seed: int | None = None) -> dict:
        """在扁平训练环境跑一小段，记录 ActionV1/奖励/信息，供测试与报告。"""
        env = self.build_env()
        obs, info = env.reset(seed=self.seed if seed is None else int(seed))
        self.reset()
        records = []
        for step in range(int(n_steps)):
            out = self.get_action(obs, info=info, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(to_flat_action(out.action))
            records.append({
                "step": step,
                "action": out.action.tolist(),
                "source": out.source,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "radar_state": info.get("radar_state"),
                "dist_m": info.get("dist_m"),
                "in_zone": bool(info.get("in_zone", False)),
            })
            if terminated or truncated:
                break
        return {
            "task_id": self.task_id,
            "steps": len(records),
            "records": records,
            "total_reward": float(sum(r["reward"] for r in records)),
            "terminated_reason": info.get("terminated_reason"),
        }

    def evaluate(self, n_steps: int = 64) -> dict:
        """推理统计：动作范围/均值/发射率/奖励有限性。"""
        traj = self.rollout(n_steps=n_steps, deterministic=True)
        actions = np.asarray([r["action"] for r in traj["records"]], dtype=np.float32)
        rewards = np.asarray([r["reward"] for r in traj["records"]], dtype=np.float32)
        stats = {
            "task_id": self.task_id,
            "steps": traj["steps"],
            "reward_total": traj["total_reward"],
            "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "reward_finite": bool(np.all(np.isfinite(rewards))),
            "action_shape": list(actions.shape),
            "action_dtype": str(actions.dtype),
            "action_min": actions.min(axis=0).tolist() if actions.size else [],
            "action_max": actions.max(axis=0).tolist() if actions.size else [],
            "action_mean": actions.mean(axis=0).tolist() if actions.size else [],
            "fire_rate": float(actions[:, 4].mean()) if actions.size else 0.0,
            "terminated_reason": traj["terminated_reason"],
        }
        return stats

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        self._ensure_model()
        path = Path(path) if path is not None else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        spec_path = path.with_suffix(path.suffix + ".spec.json")
        spec_path.write_text(json.dumps(self.action_spec(), ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return path

    def load(self, path: str | Path | None = None, build_env: bool = True):
        path = Path(path) if path is not None else self.model_path
        if not path.exists() and path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(f"模型不存在: {path}")
        env = wrap_observation(self.build_env()) if build_env else None
        self.model = self.algo_cls.load(str(path), env=env)
        self.train_env = env
        return self.model


__all__ = ["TaskDRLExpertAgent", "DEFAULT_TRAIN_CONFIG", "MODEL_DIR"]
