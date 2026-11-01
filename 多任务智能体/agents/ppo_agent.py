# -*- coding: utf-8 -*-
"""
agents/ppo_agent.py
===================
阶段三 P1 训练入口：SB3 PPO + 混合动作策略（opt-in）。

    FlatTaskMissionEnv（Box(7)）
        └─ ScaledObservation（28 维 [-1,1]）
             └─ MaskedObsWrapper（28 + 7 掩码 = 35）
                  └─ HybridActorCriticPolicy（连续 4 + 武器 3 Bernoulli）

用法（在 多任务智能体/ 下）：
    python -X utf8 -m agents.ppo_agent --task strike --policy hybrid \
        --timesteps 2048 --n-envs 2 --seed 0 --out-dir models/hybrid_smoke

    # 兼容 MLP 基线（仍为 ActionV1 Box(7)，不含概率化武器头）
    python -X utf8 -m agents.ppo_agent --task strike --policy mlp \
        --timesteps 2048 --n-envs 2 --seed 0 --out-dir models/mlp_smoke

说明：默认 `agents.task_agents` 的 MlpPolicy 训练路径不变；本入口是 P1 新通道。
不跑长训练；P1 的 >60% 命中率验收留到服务器长训练。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from agents.hybrid_policy import (HybridActorCriticPolicy, MaskedObsWrapper,
                                  masked_obs)
from agents.policy_interface import (ACTION_V1_DIM, ACTION_V1_VERSION,
                                     PolicyOutput, action_spec,
                                     apply_action_mask, clip_action_v1,
                                     compute_action_mask)
from agents.task_agents.mission_env import (MISSION_NAMES_ZH,
                                            FlatTaskMissionEnv, normalize_obs,
                                            validate_mission, wrap_observation)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

DEFAULT_TRAIN_CONFIG = {
    "policy": "hybrid",
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
}


def make_masked_env(task: str = "strike", config: dict | None = None):
    """构建训练环境：FlatTaskMissionEnv → 尺度化 → 拼 7 维掩码（35 维）。"""
    raw = FlatTaskMissionEnv(mission=task, config=config)
    return MaskedObsWrapper(wrap_observation(raw))


class _RewardStatsCallback(BaseCallback):
    """训练期奖励统计（SB3 默认 logger 未记录 ep_rew_mean 时的补充）。"""

    def __init__(self):
        super().__init__()
        self.reward_sum = 0.0
        self.reward_n = 0
        self.episode_returns = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.reward_sum += float(np.sum(rewards))
            self.reward_n += int(np.size(rewards))
        for info in self.locals.get("infos", []) or []:
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode and "r" in episode:
                self.episode_returns.append(float(episode["r"]))
        return True

    def summary(self) -> dict:
        out = {}
        if self.reward_n:
            out["train/mean_step_reward"] = self.reward_sum / self.reward_n
        if self.episode_returns:
            out["rollout/ep_rew_mean"] = float(np.mean(self.episode_returns))
        return out


class HybridPPOAgent:
    """P1 混合动作 PPO 智能体（PolicyProtocol 兼容）。"""

    def __init__(self, task: str = "strike", seed: int = 0,
                 env_config: dict | None = None, train_config: dict | None = None,
                 model_path: str | Path | None = None):
        self.task_id = validate_mission(str(task))
        self.seed = int(seed)
        self.env_config = dict(env_config or {})
        self.train_config = dict(DEFAULT_TRAIN_CONFIG)
        if train_config:
            self.train_config.update(train_config)
        self.model_path = Path(model_path) if model_path else (
            MODEL_DIR / f"hybrid_{self.task_id}_ppo")
        self.env = None
        self.train_env = None
        self.model = None
        self.last_train_metrics: dict = {}

    # ------------------------------------------------------------------
    # 环境/模型
    # ------------------------------------------------------------------
    def build_env(self):
        if self.env is None:
            self.env = make_masked_env(self.task_id, self.env_config)
        return self.env

    def build_model(self, env=None):
        self.env = env if env is not None else self.build_env()
        self.train_env = self.env
        cfg = self.train_config
        self.model = PPO(
            HybridActorCriticPolicy, self.env,
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            n_epochs=int(cfg["n_epochs"]),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            clip_range=float(cfg["clip_range"]),
            ent_coef=float(cfg["ent_coef"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            seed=self.seed, verbose=0,
        )
        return self.model

    def _ensure_model(self):
        if self.model is None:
            self.build_model()
        return self.model

    def action_spec(self) -> dict:
        spec = action_spec(self.task_id)
        spec.update({
            "mission": self.task_id,
            "mission_name_zh": MISSION_NAMES_ZH[self.task_id],
            "policy": "hybrid_actor_critic",
            "observation": {"version": "bvr_obs_v1+mask_v1", "dim": 35,
                            "base_dim": 28, "mask_dim": ACTION_V1_DIM},
            "algorithm": "PPO",
            "model_path": str(self.model_path),
        })
        return spec

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_with_meta(self, obs35, deterministic: bool):
        model = self._ensure_model()
        obs_t = torch.as_tensor(obs35, dtype=torch.float32,
                                device=model.device).unsqueeze(0)
        dist = model.policy.get_distribution(obs_t)
        action = dist.get_actions(deterministic=deterministic)
        log_prob = float(dist.log_prob(action).reshape(-1)[0].cpu())
        value = float(model.policy.predict_values(obs_t).reshape(-1)[0].cpu())
        return action.reshape(-1).cpu().numpy(), log_prob, value

    def get_action(self, obs, *, info=None, deterministic=True, mask=None,
                   guide_action=None, guide_steps: int = 0) -> PolicyOutput:
        info = dict(info or {})
        mask_arr = (np.asarray(mask, dtype=bool).reshape(ACTION_V1_DIM) if mask is not None
                    else compute_action_mask(obs, info, self.env_config))
        meta = {"deterministic": bool(deterministic)}
        source = "drl"
        if guide_steps and int(guide_steps) > 0 and guide_action is not None:
            action = clip_action_v1(guide_action)
            source = "guide"
            meta["guide_steps"] = int(guide_steps)
            log_prob = value = None
        else:
            policy_obs = np.concatenate(
                [normalize_obs(obs).reshape(-1),
                 mask_arr.astype(np.float32)], axis=0).astype(np.float32)
            try:
                raw, log_prob, value = self._predict_with_meta(
                    policy_obs, deterministic=bool(deterministic))
                action = clip_action_v1(raw)
            except Exception as exc:
                from agents.policy_interface import SAFE_ACTION_V1
                action = SAFE_ACTION_V1.copy()
                source = "fallback"
                log_prob = value = None
                meta["error"] = f"{type(exc).__name__}: {exc}"
        action = apply_action_mask(action, mask_arr)
        return PolicyOutput(action=action, source=source, log_prob=log_prob,
                            value=value, masked=mask_arr,
                            action_version=ACTION_V1_VERSION,
                            task_id=self.task_id, info=meta)

    def predict(self, obs, deterministic: bool = True):
        return self.get_action(obs, deterministic=deterministic).action

    # ------------------------------------------------------------------
    # 训练/保存/加载
    # ------------------------------------------------------------------
    def train(self, timesteps: int = 2048, progress_bar: bool = False,
              tensorboard_log: str | None = None,
              checkpoint_freq: int = 0, checkpoint_dir: str | Path | None = None,
              verbose: int = 0) -> dict:
        model = self._ensure_model()
        if tensorboard_log:
            model.tensorboard_log = str(tensorboard_log)
        callbacks = []
        reward_stats = _RewardStatsCallback()
        callbacks.append(reward_stats)
        if checkpoint_freq and checkpoint_freq > 0:
            save_path = Path(checkpoint_dir or (self.model_path.parent / "checkpoints"))
            save_path.mkdir(parents=True, exist_ok=True)
            callbacks.append(CheckpointCallback(
                save_freq=max(int(checkpoint_freq) // max(int(model.n_envs), 1), 1),
                save_path=str(save_path),
                name_prefix=f"hybrid_{self.task_id}"))
        t0 = time.time()
        model.learn(total_timesteps=int(timesteps), progress_bar=progress_bar,
                    callback=callbacks or None)
        metrics = self._collect_metrics()
        metrics.update(reward_stats.summary())
        self.last_train_metrics = {
            "task_id": self.task_id, "policy": "hybrid", "timesteps": int(timesteps),
            "n_envs": int(model.n_envs), "seed": self.seed,
            "wall_s": round(time.time() - t0, 3),
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

    def save(self, path: str | Path | None = None) -> Path:
        model = self._ensure_model()
        path = Path(path) if path is not None else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        spec_path = Path(str(path) + ".spec.json")
        spec_path.write_text(json.dumps(self.action_spec(), ensure_ascii=False, indent=2),
                             encoding="utf-8")
        self.model_path = Path(str(path) + ".zip") if path.suffix != ".zip" else path
        return self.model_path

    def load(self, path: str | Path | None = None, build_env: bool = True):
        path = Path(path) if path is not None else self.model_path
        if not path.exists():
            path = Path(str(path) + ".zip")
        if not path.exists():
            raise FileNotFoundError(f"模型不存在: {path}")
        env = self.build_env() if build_env else None
        self.model = PPO.load(str(path), env=env, device="cpu")
        self.train_env = env
        self.model_path = path
        return self.model


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _make_vec_env(task: str, n_envs: int, seed: int):
    def thunk(i):
        return lambda: make_masked_env(task)
    return VecMonitor(DummyVecEnv([thunk(i) for i in range(int(n_envs))]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阶段三 P1 混合动作 PPO 训练入口")
    ap.add_argument("--task", default="strike", help="strike/patrol/defense/ew")
    ap.add_argument("--policy", default="hybrid", choices=["hybrid", "mlp"])
    ap.add_argument("--timesteps", type=int, default=2048)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="models/hybrid_smoke")
    ap.add_argument("--resume", default=None, help="从已有模型继续训练")
    ap.add_argument("--checkpoint-freq", type=int, default=0,
                    help="每 N 环境步保存 checkpoint（0=关闭）")
    ap.add_argument("--tensorboard", action="store_true", help="启用 TensorBoard")
    ap.add_argument("--tb-log", default="tensorboard/hybrid_smoke")
    ap.add_argument("--eval-steps", type=int, default=16, help="训练后确定性评估步数")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    task = validate_mission(args.task)
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_env = _make_vec_env(task, args.n_envs, seed)
    vec_env.seed(seed)

    if args.policy == "hybrid":
        agent = HybridPPOAgent(task=task, seed=seed, model_path=out_dir / "hybrid_ppo_final")
        agent.train_env = vec_env
        agent.env = vec_env
        if args.resume:
            if not args.quiet:
                print(f"[resume] {args.resume}")
            agent.model = PPO.load(args.resume, env=vec_env, device="cpu")
        else:
            agent.build_model(vec_env)
        # PPO 已带 seed；TensorBoard 在 learn 前设置
        tb = args.tb_log if args.tensorboard else None
        metrics = agent.train(timesteps=args.timesteps, tensorboard_log=tb,
                              checkpoint_freq=args.checkpoint_freq,
                              checkpoint_dir=out_dir / "checkpoints",
                              verbose=0)
        model_path = agent.save(out_dir / "hybrid_ppo_final")
        model = PPO.load(str(model_path), env=vec_env, device="cpu")
    else:
        from stable_baselines3 import PPO as SB3PPO
        model = (SB3PPO.load(args.resume, env=vec_env, device="cpu") if args.resume
                 else SB3PPO("MlpPolicy", vec_env,
                             policy_kwargs={"net_arch": [256, 128]},
                             learning_rate=3e-4, n_steps=128, batch_size=64,
                             n_epochs=4, gamma=0.99, gae_lambda=0.95,
                             clip_range=0.2, ent_coef=0.01, max_grad_norm=0.5,
                             tensorboard_log=(args.tb_log if args.tensorboard else None),
                             seed=seed, verbose=0))
        t0 = time.time()
        model.learn(total_timesteps=int(args.timesteps))
        metrics = {"wall_s": round(time.time() - t0, 3)}
        model_path = out_dir / "mlp_ppo_final"
        model.save(str(model_path))
        model_path = Path(str(model_path) + ".zip")
        model = SB3PPO.load(str(model_path), env=vec_env, device="cpu")

    # 训练后确定性评估：形状/界/掩码一致性
    obs = vec_env.reset()
    acts = []
    reward_sum = 0.0
    reward_n = 0
    for _ in range(int(args.eval_steps)):
        action, _ = model.predict(obs, deterministic=True)
        acts.append(action[0])
        obs, rewards, _, _ = vec_env.step(action)
        reward_sum += float(np.sum(rewards))
        reward_n += int(np.size(rewards))
    acts = np.asarray(acts, dtype=np.float32)
    finite = bool(np.all(np.isfinite(acts)))
    in_bounds = bool(np.all(acts[:, 0] >= 0) and np.all(acts[:, 0] <= 1)
                     and np.all(np.abs(acts[:, 1:4]) <= 1.0))
    summary = {
        "task": task, "policy": args.policy, "timesteps": int(args.timesteps),
        "n_envs": int(args.n_envs), "seed": seed,
        "model_path": str(model_path), "tensorboard": args.tb_log if args.tensorboard else None,
        "metrics": metrics,
        "eval": {"steps": int(args.eval_steps), "finite": finite,
                 "in_bounds": in_bounds,
                 "mean_reward": round(reward_sum / reward_n, 4) if reward_n else 0.0,
                 "action_min": acts.min(axis=0).tolist() if acts.size else [],
                 "action_max": acts.max(axis=0).tolist() if acts.size else []},
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        train_metrics = metrics.get("metrics", metrics)
        print(f"[完成] {args.policy} task={task} steps={args.timesteps} "
              f"wall={metrics.get('wall_s', train_metrics.get('wall_s', '?'))}s")
        for key in ("rollout/ep_rew_mean", "train/policy_gradient_loss",
                    "train/value_loss", "train/entropy_loss", "train/approx_kl",
                    "time/fps"):
            if key in train_metrics:
                print(f"  {key}: {train_metrics[key]:.6g}")
        print(f"[评估] finite={finite} in_bounds={in_bounds} "
              f"mean_reward={summary['eval']['mean_reward']} "
              f"action_min={summary['eval']['action_min']}")
        print(f"[模型] {model_path}")
        print(f"[摘要] {out_dir / 'train_summary.json'}")
    vec_env.close()
    return 0 if (finite and in_bounds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
