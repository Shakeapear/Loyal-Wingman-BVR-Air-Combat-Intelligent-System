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
import csv
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor, VecNormalize)

from common.paths import MODEL_DIR, ensure_writable_dir

from agents.hybrid_policy import HybridActorCriticPolicy, MaskedObsWrapper
from agents.model_spec import save_model_spec, validate_model_artifact
from agents.policy_interface import (ACTION_V1_DIM, ACTION_V1_VERSION,
                                     PolicyOutput, action_spec,
                                     apply_action_mask, clip_action_v1,
                                     compute_action_mask)
from agents.evaluate_strike import evaluate_model, parse_seeds
from agents.task_agents.mission_env import (MISSION_NAMES_ZH,
                                            FlatTaskMissionEnv, normalize_obs,
                                            scene_config, validate_mission,
                                            wrap_observation)

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


def make_policy_env(policy: str, task: str = "strike", scene: str = "default",
                    extra_config: dict | None = None):
    """按 policy 选择观测：P1 hybrid=35 维（含掩码），P0 mlp=28 维。"""
    cfg = scene_config(scene)
    if extra_config:
        cfg.update(extra_config)
    raw = FlatTaskMissionEnv(mission=task, config=cfg)
    scaled = wrap_observation(raw)
    return MaskedObsWrapper(scaled) if policy == "hybrid" else scaled


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


def _has_prior_artifacts(out_dir: str | Path) -> bool:
    """判断输出目录是否已有训练产物（用于阻止误覆盖）。"""
    out = Path(out_dir)
    for name in ("training_log.csv", "train_summary.json",
                 "hybrid_ppo_final.zip", "mlp_ppo_final.zip"):
        if (out / name).exists():
            return True
    for sub in ("checkpoints", "eval_periodic"):
        directory = out / sub
        if directory.is_dir() and any(p.is_file() for p in directory.rglob("*")):
            return True
    return False


def _last_logged_episodes(log_path: str | Path) -> int:
    """从已有 training_log.csv 读取最后一个 episode 计数（续训监控用）。"""
    path = Path(log_path)
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return 0
    for row in reversed(rows):
        try:
            return max(0, int(float(row.get("episodes", "") or 0)))
        except (TypeError, ValueError):
            continue
    return 0


class _TrainingMonitorCallback(BaseCallback):
    """长训练监控：CSV 学习曲线、每 N episode 固定测试集评估、每 M episode checkpoint。"""

    def __init__(self, policy: str, scene: str, eval_seeds: list,
                 log_path: str | Path, eval_dir: str | Path,
                 checkpoint_dir: str | Path, log_every_steps: int = 512,
                 eval_every_episodes: int = 500,
                 checkpoint_every_episodes: int = 1000,
                 append_log: bool = False, resume_episodes: int = 0,
                 compute_envelope: bool = True):
        super().__init__()
        self.policy = policy
        self.scene = scene
        self.eval_seeds = list(eval_seeds)
        self.log_path = Path(log_path)
        self.eval_dir = Path(eval_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every_steps = int(log_every_steps)
        self.eval_every_episodes = int(eval_every_episodes)
        self.checkpoint_every_episodes = int(checkpoint_every_episodes)
        self.append_log = bool(append_log)
        self.compute_envelope = bool(compute_envelope)
        self.episodes = int(resume_episodes)
        self._last_log_step = 0
        # 续训时以已有 episode 计数为基线，避免恢复后立即触发评估/checkpoint。
        self._last_eval_ep = self.episodes
        self._last_ckpt_ep = self.episodes
        self._reward_sum = 0.0
        self._reward_n = 0
        self._ep_returns = []
        self._csv = None
        self._writer = None
        self.eval_history = []

    def _on_training_start(self) -> None:
        self._t_start = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        resume_csv = (self.append_log and self.log_path.is_file()
                      and self.log_path.stat().st_size > 0)
        self._csv = self.log_path.open("a" if resume_csv else "w",
                                       newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv)
        if not resume_csv:
            self._writer.writerow(["step", "wall_s", "episodes", "mean_step_reward",
                                   "ep_rew_mean", "policy_gradient_loss",
                                   "value_loss", "entropy_loss", "approx_kl"])
        else:
            # 续训时保留已有评估历史，避免 eval_history.json 被覆盖。
            history = self.eval_dir / "eval_history.json"
            if history.is_file():
                try:
                    loaded = json.loads(history.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        self.eval_history = loaded
                except (OSError, ValueError):
                    pass
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        for info in infos:
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode:
                self.episodes += 1
                if "r" in episode:
                    self._ep_returns.append(float(episode["r"]))
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._reward_sum += float(np.sum(rewards))
            self._reward_n += int(np.size(rewards))
        step = int(self.model.num_timesteps)
        if step - self._last_log_step >= self.log_every_steps:
            vals = self.model.logger.name_to_value
            mean_reward = (self._reward_sum / self._reward_n
                           if self._reward_n else 0.0)
            ep_rew = (float(np.mean(self._ep_returns))
                      if self._ep_returns else "")
            self._writer.writerow([
                step, round(time.time() - self._t_start, 2), self.episodes,
                round(mean_reward, 6), ep_rew,
                vals.get("train/policy_gradient_loss", ""),
                vals.get("train/value_loss", ""),
                vals.get("train/entropy_loss", ""),
                vals.get("train/approx_kl", ""),
            ])
            self._csv.flush()
            self._last_log_step = step
            self._reward_sum, self._reward_n = 0.0, 0
            self._ep_returns = []
        if (self.eval_every_episodes > 0
                and self.episodes - self._last_eval_ep >= self.eval_every_episodes):
            from agents.evaluate_strike import evaluate_model
            # 文件名同时带 episode 与 num_timesteps：续训时即使 episode 计数
            # 从 0 重新开始，也不会覆盖上一段训练的评估/模型文件。
            tag = f"ep{self.episodes:05d}_step{step:09d}"
            agg = evaluate_model(self.model, self.policy, self.eval_seeds,
                                 scene=self.scene, deterministic=True,
                                 out_dir=self.eval_dir / tag,
                                 compute_envelope=self.compute_envelope)
            agg["train_step"] = step
            self.eval_history.append(agg)
            (self.eval_dir / "eval_history.json").write_text(
                json.dumps(self.eval_history, ensure_ascii=False, indent=2),
                encoding="utf-8")
            self._last_eval_ep = self.episodes
        if (self.checkpoint_every_episodes > 0
                and self.episodes - self._last_ckpt_ep >= self.checkpoint_every_episodes):
            self.model.save(str(self.checkpoint_dir
                                / f"ckpt_ep{self.episodes:05d}_step{step:09d}.zip"))
            self._last_ckpt_ep = self.episodes
        return True

    def _on_training_end(self) -> None:
        if self._csv is not None:
            self._csv.close()


class HybridPPOAgent:
    """P1 混合动作 PPO 智能体（PolicyProtocol 兼容）。"""

    def __init__(self, task: str = "strike", seed: int = 0,
                 env_config: dict | None = None, train_config: dict | None = None,
                 model_path: str | Path | None = None, scene: str = "default",
                 device: str = "cpu"):
        self.task_id = validate_mission(str(task))
        self.seed = int(seed)
        self.scene = str(scene)
        self.device = str(device)
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
            self.env = make_policy_env("hybrid", self.task_id, self.scene,
                                       self.env_config)
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
            seed=self.seed, device=self.device, verbose=0,
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
            "scene": self.scene,
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
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        base_obs = (obs_arr[:-ACTION_V1_DIM]
                    if obs_arr.size == 28 + ACTION_V1_DIM else obs_arr)
        mask_arr = (np.asarray(mask, dtype=bool).reshape(ACTION_V1_DIM) if mask is not None
                    else compute_action_mask(base_obs, info, self.env_config))
        meta = {"deterministic": bool(deterministic)}
        source = "drl"
        if guide_steps and int(guide_steps) > 0 and guide_action is not None:
            action = clip_action_v1(guide_action)
            source = "guide"
            meta["guide_steps"] = int(guide_steps)
            log_prob = value = None
        else:
            policy_obs = np.concatenate(
                [normalize_obs(base_obs).reshape(-1),
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
              verbose: int = 0, extra_callbacks=None,
              reset_num_timesteps: bool = True) -> dict:
        model = self._ensure_model()
        if tensorboard_log:
            model.tensorboard_log = str(tensorboard_log)
        callbacks = list(extra_callbacks or [])
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
                    callback=callbacks or None,
                    reset_num_timesteps=bool(reset_num_timesteps))
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
        self.model = PPO.load(str(path), env=None, device=self.device)
        validate_model_artifact(self.model, policy="hybrid", task=self.task_id,
                                scene=self.scene, path=path)
        if env is not None:
            self.model.set_env(env)
        self.train_env = env
        self.model_path = path
        return self.model


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _subproc_env_fn(policy: str, task: str, scene: str,
                    extra_config: dict | None = None):
    """SubprocVecEnv 子进程入口：避免每个 JSBSim 进程再开满 CPU 线程。"""
    import torch
    torch.set_num_threads(1)
    return make_policy_env(policy, task, scene, extra_config=extra_config)


def _choose_vec_env(kind: str) -> str:
    if kind != "auto":
        return kind
    # Windows 的 spawn 对闭包/JSBSim C++ 对象更敏感；本地/单机冒烟默认 dummy，
    # Linux 服务器默认 subproc。长训练仍可在命令行显式指定。
    return "dummy" if sys.platform.startswith("win") else "subproc"


def _make_vec_env(task: str, n_envs: int, seed: int, policy: str = "hybrid",
                  scene: str = "default", vec_env: str = "auto",
                  extra_config: dict | None = None):
    kind = _choose_vec_env(vec_env)
    if kind == "subproc":
        fns = [partial(_subproc_env_fn, policy, task, scene, extra_config)
               for _ in range(int(n_envs))]
        base = SubprocVecEnv(fns)
    else:
        fns = [partial(make_policy_env, policy, task, scene,
                       extra_config=extra_config)
               for _ in range(int(n_envs))]
        base = DummyVecEnv(fns)
    return VecMonitor(base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阶段三 P1 混合动作 PPO 训练入口")
    ap.add_argument("--task", default="strike", help="strike/patrol/defense/ew")
    ap.add_argument("--policy", default="hybrid", choices=["hybrid", "mlp"])
    ap.add_argument("--timesteps", type=int, default=2048)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--vec-env", default="auto",
                    choices=["auto", "subproc", "dummy"],
                    help="并行环境类型（auto：Linux=subproc，Windows=dummy）")
    ap.add_argument("--device", default="cpu",
                    help="torch 设备（默认 cpu；本项目不依赖 GPU）")
    ap.add_argument("--compute-envelope", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="reset 时解算攻击区（默认开启；--no-compute-envelope 提速）")
    ap.add_argument("--normalize-reward", action="store_true",
                    help="启用 reward/return 归一化（默认关闭，便于与旧基线对比）")
    ap.add_argument("--mask-actions", action="store_true",
                    help="在环境侧按 ActionMaskV1 清零非法武器动作（P0 可选基线）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="bvr80-120",
                    choices=["default", "bvr80-120"],
                    help="训练场景（P0/P1 使用 bvr80-120：80~120 km/600 步）")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--out-dir", default=str(MODEL_DIR / "hybrid_smoke"))
    ap.add_argument("--overwrite", action="store_true",
                    help="允许在已有训练产物的输出目录上重新开始（默认拒绝）")
    ap.add_argument("--resume", default=None, help="从已有模型继续训练")
    ap.add_argument("--checkpoint-freq", type=int, default=0,
                    help="每 N 环境步保存 checkpoint（0=关闭）")
    ap.add_argument("--checkpoint-episodes", type=int, default=1000,
                    help="每 N episode 保存 checkpoint（0=关闭）")
    ap.add_argument("--eval-freq-episodes", type=int, default=500,
                    help="每 N episode 用固定 20 seeds 评估（0=关闭）")
    ap.add_argument("--eval-seeds", default="1-20")
    ap.add_argument("--log-every-steps", type=int, default=512)
    ap.add_argument("--final-eval", action="store_true",
                    help="训练结束后跑固定测试集评估")
    ap.add_argument("--tensorboard", action="store_true", help="启用 TensorBoard")
    ap.add_argument("--tb-log", default=None)
    ap.add_argument("--eval-steps", type=int, default=16, help="训练后环境步快速检查")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    task = validate_mission(args.task)
    if args.n_envs <= 0 or args.n_steps <= 0 or args.batch_size <= 0:
        raise SystemExit("--n-envs/--n-steps/--batch-size 必须为正整数")
    rollout = int(args.n_envs) * int(args.n_steps)
    if rollout % int(args.batch_size) != 0:
        raise SystemExit(
            f"n_envs*n_steps={rollout} 必须能被 batch_size={args.batch_size} 整除；"
            "请调整 --n-envs/--n-steps/--batch-size。")
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = ensure_writable_dir(args.out_dir, label="训练输出目录")
    resume = bool(args.resume)
    if not resume and not args.overwrite and _has_prior_artifacts(out_dir):
        raise SystemExit(
            f"输出目录已有训练产物：{out_dir}\n"
            "续训请加 --resume <checkpoint.zip>；确需从头重跑请加 --overwrite，"
            "或换一个 --out-dir。")
    vec_env = _make_vec_env(
        task, args.n_envs, seed, args.policy, args.scene, args.vec_env,
        extra_config={"compute_envelope": bool(args.compute_envelope),
                      "mask_actions": bool(args.mask_actions)})
    if args.normalize_reward:
        stats_path = out_dir / "vecnormalize.pkl"
        if resume and stats_path.is_file():
            vec_env = VecNormalize.load(str(stats_path), vec_env)
        else:
            if resume and not args.quiet:
                print("[warn] 未找到 vecnormalize.pkl，续训从零开始统计")
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                                   clip_reward=10.0, gamma=0.99)
    vec_env.seed(seed)
    eval_seeds = parse_seeds(args.eval_seeds)
    tb_log = None
    if args.tensorboard:
        tb_log = str(ensure_writable_dir(args.tb_log or (out_dir / "tensorboard"),
                                         label="TensorBoard 目录"))
    resume_episodes = _last_logged_episodes(out_dir / "training_log.csv") if resume else 0
    monitor = _TrainingMonitorCallback(
        policy=args.policy, scene=args.scene, eval_seeds=eval_seeds,
        log_path=out_dir / "training_log.csv", eval_dir=out_dir / "eval_periodic",
        checkpoint_dir=out_dir / "checkpoints",
        log_every_steps=args.log_every_steps,
        eval_every_episodes=args.eval_freq_episodes,
        checkpoint_every_episodes=args.checkpoint_episodes,
        append_log=resume, resume_episodes=resume_episodes,
        compute_envelope=bool(args.compute_envelope))

    if args.policy == "hybrid":
        agent = HybridPPOAgent(
            task=task, seed=seed, model_path=out_dir / "hybrid_ppo_final",
            scene=args.scene, device=args.device,
            train_config={"n_steps": args.n_steps, "batch_size": args.batch_size,
                          "n_epochs": args.n_epochs,
                          "learning_rate": args.learning_rate})
        agent.train_env = vec_env
        agent.env = vec_env
        if args.resume:
            if not args.quiet:
                print(f"[resume] {args.resume}")
            agent.model = PPO.load(args.resume, env=None, device=args.device)
            try:
                validate_model_artifact(agent.model, policy="hybrid", task=task,
                                        scene=args.scene, path=args.resume)
            except ValueError as exc:
                raise SystemExit(f"[resume 错误] {exc}") from exc
            agent.model.set_env(vec_env)
            agent.train_env = vec_env
        else:
            agent.build_model(vec_env)
        # PPO 已带 seed；TensorBoard 在 learn 前设置。续训时保留累计步数，
        # 否则 checkpoint/eval 文件名会从头计数并覆盖上一段产物。
        metrics = agent.train(timesteps=args.timesteps, tensorboard_log=tb_log,
                              checkpoint_freq=args.checkpoint_freq,
                              checkpoint_dir=out_dir / "checkpoints",
                              verbose=0, extra_callbacks=[monitor],
                              reset_num_timesteps=not resume)
        model_path = agent.save(out_dir / "hybrid_ppo_final")
        model = PPO.load(str(model_path), env=vec_env, device=args.device)
        validate_model_artifact(model, policy="hybrid", task=task,
                                scene=args.scene, path=model_path)
    else:
        from stable_baselines3 import PPO as SB3PPO
        if args.resume:
            model = SB3PPO.load(args.resume, env=None, device=args.device)
            try:
                validate_model_artifact(model, policy="mlp", task=task,
                                        scene=args.scene, path=args.resume)
            except ValueError as exc:
                raise SystemExit(f"[resume 错误] {exc}") from exc
            model.set_env(vec_env)
        else:
            model = SB3PPO("MlpPolicy", vec_env,
                           policy_kwargs={"net_arch": [256, 128]},
                           learning_rate=args.learning_rate,
                           n_steps=args.n_steps, batch_size=args.batch_size,
                           n_epochs=args.n_epochs, gamma=0.99, gae_lambda=0.95,
                           clip_range=0.2, ent_coef=0.01, max_grad_norm=0.5,
                           tensorboard_log=tb_log, seed=seed,
                           device=args.device, verbose=0)
        callbacks = [monitor]
        if args.checkpoint_freq and args.checkpoint_freq > 0:
            cb_dir = ensure_writable_dir(out_dir / "checkpoints",
                                         label="checkpoint 目录")
            callbacks.append(CheckpointCallback(
                save_freq=max(int(args.checkpoint_freq) // max(int(args.n_envs), 1), 1),
                save_path=str(cb_dir), name_prefix=f"mlp_{task}"))
        t0 = time.time()
        model.learn(total_timesteps=int(args.timesteps), callback=callbacks,
                    reset_num_timesteps=not resume)
        metrics = {str(k): float(v) for k, v in model.logger.name_to_value.items()
                   if np.isscalar(v)} if model.logger is not None else {}
        metrics["wall_s"] = round(time.time() - t0, 3)
        model_path = out_dir / "mlp_ppo_final"
        model.save(str(model_path))
        model_path = Path(str(model_path) + ".zip")
        save_model_spec(model_path, {
            "task_id": task, "mission": task, "policy": "mlp",
            "scene": args.scene, "algorithm": "PPO",
            "observation": {"version": "bvr_obs_v1", "dim": 28},
        })
        model = SB3PPO.load(str(model_path), env=vec_env, device=args.device)
        validate_model_artifact(model, policy="mlp", task=task,
                                scene=args.scene, path=model_path)

    if args.normalize_reward:
        vec_env.training = False
        vec_env.save(str(out_dir / "vecnormalize.pkl"))

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
    if acts.size:
        finite = bool(np.all(np.isfinite(acts)))
        in_bounds = bool(np.all(acts[:, 0] >= 0) and np.all(acts[:, 0] <= 1)
                         and np.all(np.abs(acts[:, 1:4]) <= 1.0))
    else:
        finite, in_bounds = True, True
    final_eval = None
    if args.final_eval:
        final_eval = evaluate_model(
            model, args.policy, eval_seeds, scene=args.scene,
            deterministic=True, out_dir=out_dir / "eval_final",
            compute_envelope=bool(args.compute_envelope))
    summary = {
        "task": task, "policy": args.policy, "timesteps": int(args.timesteps),
        "n_envs": int(args.n_envs), "seed": seed, "scene": args.scene,
        "compute_envelope": bool(args.compute_envelope),
        "normalize_reward": bool(args.normalize_reward),
        "mask_actions": bool(args.mask_actions),
        "hyperparams": {"n_steps": args.n_steps, "batch_size": args.batch_size,
                        "n_epochs": args.n_epochs, "learning_rate": args.learning_rate,
                        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                        "ent_coef": 0.01},
        "model_path": str(model_path), "tensorboard": tb_log,
        "training_log": str(out_dir / "training_log.csv"),
        "metrics": metrics,
        "final_eval": final_eval,
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
        if final_eval:
            print(f"[固定测试集] hit={final_eval['hit_rate']:.0%} "
                  f"survival={final_eval['survival_rate']:.0%} "
                  f"misfire={final_eval['misfire_rate']:.0%} "
                  f"safety={final_eval['safety_violations']}")
        print(f"[模型] {model_path}")
        print(f"[摘要] {out_dir / 'train_summary.json'}")
    vec_env.close()
    return 0 if (finite and in_bounds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
