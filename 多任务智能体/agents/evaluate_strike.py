# -*- coding: utf-8 -*-
"""
agents/evaluate_strike.py
=========================
P0/P1 打击任务固定测试集评估（20 seeds，默认 80–120 km / 600 步）。

- P0（mlp）：28 维尺度化观测 → MlpPolicy；
- P1（hybrid）：28 维尺度化 + 7 维掩码 = 35 维 → HybridActorCriticPolicy；
- 同一 env/scene/reward/control_mode；不加边界守卫（训练未启用，评估保持一致）；
- 指标：命中率、存活率、安全违规、误发射、平均步数、平均回报、终止分布。

也可以作为库函数供训练回调使用：
    from agents.evaluate_strike import evaluate_model
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AGENT_LIB = _HERE.parent
if str(_AGENT_LIB) not in sys.path:
    sys.path.insert(0, str(_AGENT_LIB))

import numpy as np

from agents.model_spec import validate_model_artifact
from agents.policy_interface import clip_action_v1, compute_action_mask
from agents.task_agents.mission_env import (FlatTaskMissionEnv, normalize_obs,
                                            scene_config)
from common.constants import FT2M, wrap_deg180
from common.paths import ensure_writable_dir

POLICIES = ("mlp", "hybrid")


def parse_seeds(spec: str) -> list:
    seeds = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def policy_input(policy: str, obs, info, env_cfg) -> np.ndarray:
    base = normalize_obs(obs).reshape(-1)
    if policy == "hybrid":
        mask = compute_action_mask(obs, info, env_cfg).astype(np.float32)
        return np.concatenate([base, mask], axis=0).astype(np.float32)
    return base.astype(np.float32)


def initial_situation(env) -> dict:
    frame = env.get_viz_frame()
    own, enemy = frame["own"], frame["enemy"]
    dx = enemy["pos"][0] - own["pos"][0]
    dy = enemy["pos"][1] - own["pos"][1]
    bearing_abs = math.degrees(math.atan2(dx, dy))
    return {
        "dist_km": round(frame["dist_m"] / 1000.0, 2),
        "bearing_deg": round(wrap_deg180(bearing_abs - math.degrees(own["psi_rad"])), 1),
        "own_alt_ft": round(own["h_sl_m"] / FT2M, 0),
        "own_mach": round(float(own["mach"]), 3),
        "own_v_mps": round(float(own["vtrue_mps"]), 1),
        "enemy_alt_ft": round(enemy["h_sl_m"] / FT2M, 0),
        "enemy_mach": round(float(enemy["mach"]), 3),
    }


def run_episode(model, policy: str, seed: int, env_config: dict,
                deterministic: bool = True) -> dict:
    env = FlatTaskMissionEnv(mission="strike", config=env_config)
    obs, info = env.reset(seed=int(seed))
    situation = initial_situation(env)
    target_steps = int(env.cfg.get("max_steps", 600))
    steps = launches = invalid_launches = safety = 0
    total_reward = 0.0
    reason = None
    try:
        for step in range(target_steps):
            model_obs = policy_input(policy, obs, info, env.cfg)
            action, _ = model.predict(model_obs, deterministic=bool(deterministic))
            action = clip_action_v1(action)
            if int(action[4]) == 1:
                launches += 1
                valid = bool(compute_action_mask(obs, info, env.cfg)[4])
                if not valid:
                    invalid_launches += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps = step + 1
            if env._safety_violation(env.own.state_dict()):
                safety += 1
            if terminated or truncated:
                reason = info.get("terminated_reason")
                break
    finally:
        env.close()
    survived = reason not in ("own_hit", "own_crash")
    return {
        "seed": int(seed), "policy": policy, "steps": steps,
        "terminated_reason": reason, "survived": bool(survived),
        "enemy_hit": reason == "enemy_hit",
        "safety_violations": int(safety),
        "launch_commands": int(launches),
        "invalid_launches": int(invalid_launches),
        "misfire_rate": (invalid_launches / launches) if launches else 0.0,
        "total_reward": round(float(total_reward), 4),
        "mean_reward": round(float(total_reward) / steps, 4) if steps else 0.0,
        "situation": situation,
    }


def evaluate_model(model, policy: str, seeds: list, scene: str = "bvr80-120",
                   deterministic: bool = True, out_dir: str | Path | None = None,
                   compute_envelope: bool = True) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"policy 必须为 {POLICIES}")
    env_config = scene_config(scene)
    env_config["compute_envelope"] = bool(compute_envelope)
    rows = [run_episode(model, policy, seed, env_config, deterministic)
            for seed in seeds]
    n = len(rows)
    launches = sum(r["launch_commands"] for r in rows)
    invalid = sum(r["invalid_launches"] for r in rows)
    reasons = {}
    for r in rows:
        key = r["terminated_reason"] or "none"
        reasons[key] = reasons.get(key, 0) + 1
    aggregate = {
        "policy": policy, "scene": scene, "seeds": list(seeds),
        "n_episodes": n,
        "hit_rate": round(sum(r["enemy_hit"] for r in rows) / n, 4) if n else 0.0,
        "survival_rate": round(sum(r["survived"] for r in rows) / n, 4) if n else 0.0,
        "misfire_rate": round(invalid / launches, 4) if launches else 0.0,
        "launch_commands": launches, "invalid_launches": invalid,
        "safety_violations": sum(r["safety_violations"] for r in rows),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 1) if n else 0.0,
        "avg_return": round(sum(r["total_reward"] for r in rows) / n, 2) if n else 0.0,
        "avg_reward": round(sum(r["mean_reward"] for r in rows) / n, 4) if n else 0.0,
        "termination_counts": reasons,
    }
    if out_dir is not None:
        out = ensure_writable_dir(out_dir, label="评估输出目录")
        (out / "eval_metrics.json").write_text(
            json.dumps({"aggregate": aggregate, "episodes": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        cols = ["seed", "steps", "terminated_reason", "survived", "enemy_hit",
                "safety_violations", "launch_commands", "invalid_launches",
                "misfire_rate", "total_reward", "mean_reward",
                "dist_km", "bearing_deg", "own_alt_ft", "own_mach",
                "enemy_alt_ft", "enemy_mach"]
        lines = [",".join(cols)]
        for r in rows:
            vals = {**r, **r["situation"]}
            lines.append(",".join(str(int(vals[c])) if isinstance(vals[c], bool)
                                  else str(vals[c]) for c in cols))
        (out / "eval_metrics.csv").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    return aggregate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P0/P1 打击任务固定测试集评估")
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--model", required=True, help="SB3 模型 .zip")
    ap.add_argument("--scene", default="bvr80-120")
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stochastic", action="store_true", help="默认 deterministic")
    ap.add_argument("--no-envelope", action="store_true",
                    help="跳过攻击区解算（快速评估；默认与训练一致开启）")
    args = ap.parse_args(argv)

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    try:
        validate_model_artifact(model, policy=args.policy, task="strike",
                                scene=args.scene, path=args.model)
    except ValueError as exc:
        raise SystemExit(f"[模型校验失败] {exc}") from exc
    seeds = parse_seeds(args.seeds)
    out = Path(args.out_dir) if args.out_dir else (
        Path(args.model).with_suffix("").parent / f"eval_{args.policy}")
    agg = evaluate_model(model, args.policy, seeds, scene=args.scene,
                         deterministic=not args.stochastic, out_dir=out,
                         compute_envelope=not args.no_envelope)
    print(f"[{args.policy}] hit={agg['hit_rate']:.0%} survival={agg['survival_rate']:.0%} "
          f"misfire={agg['misfire_rate']:.0%} safety={agg['safety_violations']} "
          f"avg_steps={agg['avg_steps']} avg_return={agg['avg_return']}")
    print(f"[产物] {out / 'eval_metrics.json'} / eval_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
