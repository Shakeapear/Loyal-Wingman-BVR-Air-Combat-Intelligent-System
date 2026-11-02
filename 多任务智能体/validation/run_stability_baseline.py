# -*- coding: utf-8 -*-
"""
validation/run_stability_baseline.py（阶段三稳定飞行基线）
=========================================================
对 4 个任务级智能体（strike/patrol/defense/ew）做 240 步隔离飞行测试：

- `control_mode=assisted`（本机 AP 内环）；ActionV1 仍为 7 维；
- 隔离敌方发射（n_missiles=0 + enemy_fire_interval_s 极大），只测飞行稳定性；
- 策略：SAFE_ACTION_V1、固定 seed 小幅随机、已有 512 步 PPO 模型；
- 判定：240 步内不出现 `own_crash` 即通过（timeout 允许；被导弹命中不算
  飞行稳定性失败）；
- 另可选 `--combat` 用正常战斗配置跑 240 步，仅记录终止原因，不作硬性通过。

产物：
    validation/results/stage3_stability/stability_summary.json
    validation/results/stage3_stability/stability_report.txt
    validation/results/stage3_stability/<task>_<policy>_seed<seed>.csv
    validation/results/stage3_stability/combat_summary.json（--combat）

用法（DC 环境，仓库根目录）：
    python "多任务智能体\\validation\\run_stability_baseline.py"
    python ... --tasks strike,patrol --seeds 42 --policies safe,model --combat
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = _HERE.parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from agents.action_adapter import to_flat_action  # noqa: E402
from agents.policy_interface import SAFE_ACTION_V1  # noqa: E402
from agents.task_agents import TASK_AGENTS, FlatTaskMissionEnv  # noqa: E402
from agents.task_agents.base_task_agent import MODEL_DIR  # noqa: E402
from common.paths import ensure_writable_dir, results_dir  # noqa: E402

RESULTS_DIR = results_dir() / "stage3_stability"
CSV_FIELDS = ["task_id", "policy", "seed", "step", "h_m", "v_mps", "theta_rad",
              "phi_rad", "throttle", "roll", "pitch", "yaw", "fire", "switch",
              "ecm", "reward", "terminated", "truncated", "terminated_reason"]


def _safe_action() -> np.ndarray:
    return SAFE_ACTION_V1.copy()


def _random_action(rng: np.random.Generator) -> np.ndarray:
    """小幅随机基线：围绕安全动作扰动，避免人为制造大姿态。"""
    a = SAFE_ACTION_V1.copy()
    a[0] = np.clip(0.55 + rng.normal(0.0, 0.03), 0.40, 0.70)
    a[1:3] = np.clip(rng.normal(0.0, 0.03, size=2), -0.08, 0.08)
    a[3] = 0.0
    a[4:7] = 0.0
    return a.astype(np.float32)


def _make_env(task_id: str, max_steps: int, combat: bool,
              control_mode: str = "assisted") -> FlatTaskMissionEnv:
    config = {
        "max_steps": int(max_steps),
        "compute_envelope": False,
        "control_mode": control_mode,
    }
    if not combat:
        config.update({"n_missiles": 0, "enemy_fire_interval_s": 1e9})
    return FlatTaskMissionEnv(mission=task_id, config=config)


def run_episode(task_id: str, policy: str, seed: int, max_steps: int,
                combat: bool = False, model_agent=None, out_dir: Path | None = None,
                rng: np.random.Generator | None = None,
                control_mode: str = "assisted") -> tuple[dict, list[dict]]:
    env = _make_env(task_id, max_steps, combat, control_mode=control_mode)
    rng = rng or np.random.default_rng(seed)
    records: list[dict] = []
    try:
        obs, info = env.reset(seed=seed)
        for step in range(int(max_steps)):
            if policy == "model":
                if model_agent is None:
                    raise RuntimeError("model 策略缺少已加载模型")
                action = model_agent.get_action(obs, info=info, deterministic=True).action
            elif policy == "random":
                action = _random_action(rng)
            else:
                action = _safe_action()
            obs, reward, terminated, truncated, info = env.step(to_flat_action(action))
            records.append({
                "task_id": task_id, "policy": policy, "seed": int(seed),
                "step": step + 1, "h_m": float(info.get("h_sl_m", 0.0)),
                "v_mps": float(info.get("vtrue_mps", 0.0)),
                "theta_rad": float(obs[4]), "phi_rad": float(obs[3]),
                "throttle": float(action[0]), "roll": float(action[1]),
                "pitch": float(action[2]), "yaw": float(action[3]),
                "fire": int(action[4] > 0.5), "switch": int(action[5] > 0.5),
                "ecm": int(action[6] > 0.5), "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "terminated_reason": info.get("terminated_reason"),
            })
            if terminated or truncated:
                break
    finally:
        env.close()

    hs = np.asarray([r["h_m"] for r in records], dtype=float)
    vs = np.asarray([r["v_mps"] for r in records], dtype=float)
    thetas = np.asarray([r["theta_rad"] for r in records], dtype=float)
    phis = np.asarray([r["phi_rad"] for r in records], dtype=float)
    rewards = np.asarray([r["reward"] for r in records], dtype=float)
    reason = records[-1]["terminated_reason"] if records else None
    own_crash = any(r["terminated_reason"] == "own_crash" for r in records)
    reached = len(records) >= int(max_steps)
    summary = {
        "task_id": task_id, "policy": policy, "seed": int(seed),
        "combat": bool(combat), "steps": len(records), "max_steps": int(max_steps),
        "terminated_reason": reason, "own_crash": bool(own_crash),
        "reached_240": bool(reached),
        "pass": bool((not own_crash) and (reached or reason == "timeout")),
        "h_min_m": float(hs.min()) if hs.size else None,
        "h_final_m": float(hs[-1]) if hs.size else None,
        "v_min_mps": float(vs.min()) if vs.size else None,
        "v_max_mps": float(vs.max()) if vs.size else None,
        "max_abs_theta_rad": float(np.abs(thetas).max()) if thetas.size else None,
        "max_abs_phi_rad": float(np.abs(phis).max()) if phis.size else None,
        "reward_sum": float(rewards.sum()) if rewards.size else 0.0,
        "reward_finite": bool(np.all(np.isfinite(rewards))) if rewards.size else True,
        "control_mode": control_mode,
    }
    if out_dir is not None and records:
        tag = "combat_" if combat else ""
        csv_path = out_dir / f"{task_id}_{policy}_{tag}seed{seed}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        summary["csv"] = str(csv_path)
    return summary, records


def _load_model_agent(task_id: str, seed: int, model_dir: Path):
    path = model_dir / f"task_{task_id}_ppo"
    if not path.exists() and not path.with_suffix(".zip").exists():
        return None
    cls = TASK_AGENTS[task_id]
    agent = cls(env_config={"max_steps": 240, "compute_envelope": False,
                            "control_mode": "assisted"},
                seed=seed, model_path=path)
    agent.load(agent.model_path)
    return agent


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 assisted 稳定飞行基线（240 步）")
    ap.add_argument("--tasks", default="strike,patrol,defense,ew")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--policies", default="safe,model",
                    help="safe,random,model 的逗号组合")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--control-mode", default="assisted",
                    choices=("assisted", "raw_fcs"),
                    help="assisted=本机 AP 内环；raw_fcs=原人工舵面（诊断/对照）")
    ap.add_argument("--combat", action="store_true",
                    help="额外跑正常战斗配置（n_missiles=2，敌方会发射）")
    ap.add_argument("--combat-seeds", default=None,
                    help="战斗配置种子，默认取 --seeds 第一个")
    args = ap.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_AGENTS]
    if unknown:
        print(f"[错误] 未知任务: {unknown}", file=sys.stderr)
        return 2
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if any(p not in ("safe", "random", "model") for p in policies):
        print("[错误] policies 仅支持 safe/random/model", file=sys.stderr)
        return 2
    out_dir = ensure_writable_dir(args.out_dir, label="稳定飞行结果目录")
    model_dir = Path(args.model_dir)

    print(f"[稳定飞行基线] tasks={tasks} seeds={seeds} policies={policies} "
          f"steps={args.steps} control_mode={args.control_mode} 敌方发射=隔离")
    results = []
    all_records = []
    t0 = time.time()
    for task_id in tasks:
        model_agent = None
        if "model" in policies:
            model_agent = _load_model_agent(task_id, seeds[0], model_dir)
            if model_agent is None:
                print(f"  [跳过] {task_id}: 未找到模型 {model_dir / f'task_{task_id}_ppo'}")
        try:
            for policy in policies:
                if policy == "model" and model_agent is None:
                    continue
                for seed in seeds:
                    summary, records = run_episode(
                        task_id, policy, seed, args.steps, combat=False,
                        model_agent=model_agent, out_dir=out_dir,
                        rng=np.random.default_rng(seed + 1000),
                        control_mode=args.control_mode)
                    results.append(summary)
                    all_records.extend(records)
                    print(f"  {task_id:7s} {policy:6s} seed={seed:<4d} "
                          f"steps={summary['steps']:<3d} h_min={summary['h_min_m']:.1f} "
                          f"theta_max={summary['max_abs_theta_rad']:.3f} "
                          f"reason={summary['terminated_reason']} "
                          f"own_crash={summary['own_crash']} -> "
                          f"{'PASS' if summary['pass'] else 'FAIL'}")
        finally:
            if model_agent is not None:
                model_agent.close()

    combat_results = []
    if args.combat:
        combat_seeds = ([int(s) for s in args.combat_seeds.split(",")]
                        if args.combat_seeds else [seeds[0]])
        print(f"\n[正常战斗配置] seeds={combat_seeds}（仅记录，不作硬性通过）")
        for task_id in tasks:
            for seed in combat_seeds:
                summary, _ = run_episode(task_id, "safe", seed, args.steps,
                                         combat=True, out_dir=out_dir,
                                         rng=np.random.default_rng(seed + 2000),
                                         control_mode=args.control_mode)
                combat_results.append(summary)
                print(f"  {task_id:7s} seed={seed:<4d} steps={summary['steps']} "
                      f"reason={summary['terminated_reason']} "
                      f"own_crash={summary['own_crash']}")

    wall = time.time() - t0
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "steps": int(args.steps), "seeds": seeds, "tasks": tasks,
        "policies": policies, "control_mode": args.control_mode,
        "enemy_fire_isolated": True, "wall_s": round(wall, 2),
        "passed": sum(1 for r in results if r["pass"]), "total": len(results),
        "results": results, "combat_results": combat_results,
    }
    (out_dir / "stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if combat_results:
        (out_dir / "combat_summary.json").write_text(
            json.dumps(combat_results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"阶段三稳定飞行基线报告（{args.control_mode}，敌方发射隔离）",
        f"生成时间: {summary['generated_at']}",
        f"tasks={tasks} seeds={seeds} policies={policies} steps={args.steps}",
        "",
        "task | policy | seed | steps | h_min_m | v_range_mps | max|theta| | max|phi| | reason | own_crash | pass",
        "---|---|---|---|---|---|---|---|---|---|---",
    ]
    for r in results:
        lines.append(
            f"{r['task_id']} | {r['policy']} | {r['seed']} | {r['steps']} | "
            f"{r['h_min_m']:.1f} | {r['v_min_mps']:.1f}~{r['v_max_mps']:.1f} | "
            f"{r['max_abs_theta_rad']:.3f} | {r['max_abs_phi_rad']:.3f} | "
            f"{r['terminated_reason']} | {r['own_crash']} | {r['pass']}")
    lines += ["", f"汇总: {summary['passed']}/{summary['total']} 通过，"
                  f"总耗时 {summary['wall_s']}s",
              f"JSON: {out_dir / 'stability_summary.json'}"]
    if combat_results:
        lines += ["", "正常战斗配置（仅记录）:"]
        for r in combat_results:
            lines.append(f"{r['task_id']} seed={r['seed']} steps={r['steps']} "
                         f"reason={r['terminated_reason']} own_crash={r['own_crash']}")
    report = "\n".join(lines) + "\n"
    (out_dir / "stability_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
