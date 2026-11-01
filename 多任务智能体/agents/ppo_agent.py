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
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from agents.hybrid_policy import (HybridActorCriticPolicy, MaskedObsWrapper,
                                  masked_obs)
from agents.policy_interface import (ACTION_V1_DIM, ACTION_V1_VERSION,
                                     PolicyOutput, action_spec,
                                     apply_action_mask, clip_action_v1,
                                     compute_action_mask)
from agents.evaluate_strike import evaluate_model, parse_seeds
from agents.task_agents.mission_env import (MISSION_NAMES_ZH,
                                            FlatTaskMissionEnv, normalize_obs,
                                            scene_config, validate_mission,
                                            wrap_observation)

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


class _TrainingMonitorCallback(BaseCallback):
    """长训练监控：CSV 学习曲线、每 N episode 固定测试集评估、每 M episode checkpoint。"""

    def __init__(self, policy: str, scene: str, eval_seeds: list,
                 log_path: str | Path, eval_dir: str | Path,
                 checkpoint_dir: str | Path, log_every_steps: int = 512,
                 eval_every_episodes: int = 500,
                 checkpoint_every_episodes: int = 1000):
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
        self.episodes = 0
        self._last_log_step = 0
        self._last_eval_ep = 0
        self._last_ckpt_ep = 0
        self._reward_sum = 0.0
        self._reward_n = 0
        self._ep_returns = []
        self._csv = None
        self._writer = None
        self.eval_history = []

    def _on_training_start(self) -> None:
        self._t_start = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["step", "wall_s", "episodes", "mean_step_reward",
                               "ep_rew_mean", "policy_gradient_loss",
                               "value_loss", "entropy_loss", "approx_kl"])
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
            tag = f"ep{self.episodes:05d}"
            agg = evaluate_model(self.model, self.policy, self.eval_seeds,
                                 scene=self.scene, deterministic=True,
                                 out_dir=self.eval_dir / tag)
            agg["train_step"] = step
            self.eval_history.append(agg)
            (self.eval_dir / "eval_history.json").write_text(
                json.dumps(self.eval_history, ensure_ascii=False, indent=2),
                encoding="utf-8")
            self._last_eval_ep = self.episodes
        if (self.checkpoint_every_episodes > 0
                and self.episodes - self._last_ckpt_ep >= self.checkpoint_every_episodes):
            self.model.save(str(self.checkpoint_dir
                                / f"ckpt_ep{self.episodes:05d}.zip"))
            self._last_ckpt_ep = self.episodes
        return True

    def _on_training_end(self) -> None:
        if self._csv is not None:
            self._csv.close()


class HybridPPOAgent:
    """P1 混合动作 PPO 智能体（PolicyProtocol 兼容）。"""

    def __init__(self, task: str = "strike", seed: int = 0,
                 env_config: dict | None = None, train_config: dict | None = None,
                 model_path: str | Path | None = None, scene: str = "default"):
        self.task_id = validate_mission(str(task))
        self.seed = int(seed)
        self.scene = str(scene)
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
              verbose: int = 0, extra_callbacks=None) -> dict:
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
def _make_vec_env(task: str, n_envs: int, seed: int, policy: str = "hybrid",
                  scene: str = "default", vec_env: str = "subproc"):
    def thunk(i):
        return lambda: make_policy_env(policy, task, scene)
    fns = [thunk(i) for i in range(int(n_envs))]
    base = SubprocVecEnv(fns) if vec_env == "subproc" else DummyVecEnv(fns)
    return VecMonitor(base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阶段三 P1 混合动作 PPO 训练入口")
    ap.add_argument("--task", default="strike", help="strike/patrol/defense/ew")
    ap.add_argument("--policy", default="hybrid", choices=["hybrid", "mlp"])
    ap.add_argument("--timesteps", type=int, default=2048)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--vec-env", default="subproc", choices=["subproc", "dummy"],
                    help="并行环境类型（Linux 长训练用 subproc；Windows 调试可 dummy）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="bvr80-120",
                    choices=["default", "bvr80-120"],
                    help="训练场景（P0/P1 使用 bvr80-120：80~120 km/600 步）")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--out-dir", default="models/hybrid_smoke")
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
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_env = _make_vec_env(task, args.n_envs, seed, args.policy, args.scene, args.vec_env)
    vec_env.seed(seed)
    eval_seeds = parse_seeds(args.eval_seeds)
    tb_log = (args.tb_log or str(out_dir / "tensorboard")) if args.tensorboard else None
    monitor = _TrainingMonitorCallback(
        policy=args.policy, scene=args.scene, eval_seeds=eval_seeds,
        log_path=out_dir / "training_log.csv", eval_dir=out_dir / "eval_periodic",
        checkpoint_dir=out_dir / "checkpoints",
        log_every_steps=args.log_every_steps,
        eval_every_episodes=args.eval_freq_episodes,
        checkpoint_every_episodes=args.checkpoint_episodes)

    if args.policy == "hybrid":
        agent = HybridPPOAgent(
            task=task, seed=seed, model_path=out_dir / "hybrid_ppo_final",
            scene=args.scene,
            train_config={"n_steps": args.n_steps, "batch_size": args.batch_size,
                          "n_epochs": args.n_epochs,
                          "learning_rate": args.learning_rate})
        agent.train_env = vec_env
        agent.env = vec_env
        if args.resume:
            if not args.quiet:
                print(f"[resume] {args.resume}")
            agent.model = PPO.load(args.resume, env=vec_env, device="cpu")
        else:
            agent.build_model(vec_env)
        # PPO 已带 seed；TensorBoard 在 learn 前设置
        metrics = agent.train(timesteps=args.timesteps, tensorboard_log=tb_log,
                              checkpoint_freq=args.checkpoint_freq,
                              checkpoint_dir=out_dir / "checkpoints",
                              verbose=0, extra_callbacks=[monitor])
        model_path = agent.save(out_dir / "hybrid_ppo_final")
        model = PPO.load(str(model_path), env=vec_env, device="cpu")
    else:
        from stable_baselines3 import PPO as SB3PPO
        model = (SB3PPO.load(args.resume, env=vec_env, device="cpu") if args.resume
                 else SB3PPO("MlpPolicy", vec_env,
                             policy_kwargs={"net_arch": [256, 128]},
                             learning_rate=args.learning_rate,
                             n_steps=args.n_steps, batch_size=args.batch_size,
                             n_epochs=args.n_epochs, gamma=0.99, gae_lambda=0.95,
                             clip_range=0.2, ent_coef=0.01, max_grad_norm=0.5,
                             tensorboard_log=tb_log, seed=seed, verbose=0))
        t0 = time.time()
        model.learn(total_timesteps=int(args.timesteps), callback=[monitor])
        metrics = {str(k): float(v) for k, v in model.logger.name_to_value.items()
                   if np.isscalar(v)} if model.logger is not None else {}
        metrics["wall_s"] = round(time.time() - t0, 3)
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
            deterministic=True, out_dir=out_dir / "eval_final")
    summary = {
        "task": task, "policy": args.policy, "timesteps": int(args.timesteps),
        "n_envs": int(args.n_envs), "seed": seed, "scene": args.scene,
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
