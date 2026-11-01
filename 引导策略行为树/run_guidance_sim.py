# -*- coding: utf-8 -*-
"""
引导策略行为树/run_guidance_sim.py（步骤 2.6 D2.6-4 联调运行器）
================================================================
把【边界约束守卫 + 战术执行子树 + GuideInterface】接入 BVRCombatEnv：

    GuideInterface.get_action(obs, policy)
        └─ policy → executor.tick_once(..., boundary=BoundaryConstraints)
                        ├─ 战术执行子树（py_trees，OODA）
                        └─ 边界约束覆盖/限幅（最高优先级）

用法（DC 环境，在 引导策略行为树/ 目录下）：
    python -X utf8 run_guidance_sim.py                       # 默认 seeds 11,22,33
    python -X utf8 run_guidance_sim.py --seeds 11,22,33,44   # 自定义种子
    python -X utf8 run_guidance_sim.py --max-steps 120 --fast

产物（默认 outputs/guidance/）：
    tactical_tree.txt / .dot / .png        战术树可视化（dot 可选）
    guidance_seed<N>.csv                   逐帧全量数据
    guidance_seed<N>.acmi                  TacView 回放
    guidance_seed<N>.summary.json          指标 + 约束计数 + 引导比例
    guidance_seed<N>.events.json           边界约束触发/清除日志
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_HERE, _REPO_ROOT / "多任务智能体", _REPO_ROOT / "可视化工具"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import numpy as np

from common.bvr_combat_env import BVRCombatEnv                      # noqa: E402
from common.constants import GEO_LAT0_DEG, GEO_LON0_DEG             # noqa: E402
from llm_bt.boundary_constraints import BoundaryConstraints         # noqa: E402
from llm_bt.executor import Blackboard, tick_once                   # noqa: E402
from llm_bt.guide_interface import GuideInterface                   # noqa: E402
from llm_bt.schema import OODA_ALL                                  # noqa: E402
from llm_bt.tactical_execution_tree import (build_tactical_tree,    # noqa: E402
                                            tactical_coverage,
                                            tactical_tree_json,
                                            validate_tactical_tree)
from llm_bt.tester import is_safety_violation                       # noqa: E402
from llm_bt.tree_io import text_tree, tree_to_dot                   # noqa: E402
from visualization.acmi import export_acmi                          # noqa: E402
from visualization.recorder import EpisodeRecorder                  # noqa: E402


def _default_action() -> dict:
    """引导耗尽且无 DRL 时的安全平飞占位（联调用）。"""
    return {"flight": [0.55, 0.0, 0.0, 0.0], "weapon": [0, 0, 0]}


def run_episode(seed: int, max_steps: int = 240,
                compute_envelope: bool = True, guard_config: dict | None = None) -> tuple:
    """执行一局：战术树 + 边界守卫 + GuideInterface。"""
    tree = tactical_tree_json()
    env = BVRCombatEnv(config={"compute_envelope": bool(compute_envelope)})
    bb = Blackboard(env)
    root = build_tactical_tree(bb)
    guard = BoundaryConstraints(config=guard_config, env_cfg=env.cfg)
    guide = GuideInterface(max_guide_steps=int(max_steps))
    rec = EpisodeRecorder(meta={"lat0_deg": GEO_LAT0_DEG, "lon0_deg": GEO_LON0_DEG})

    obs, info = env.reset(seed=int(seed))
    rec.capture_env(env)
    logic_errors = safety = in_zone_steps = 0
    own_fired = enemy_fired = guided_steps = 0
    reason, steps, t0 = None, 0, time.time()
    try:
        for step in range(int(max_steps)):
            bb.set_frame(obs, info, step)

            def policy(_obs):
                return tick_once(root, bb, _obs, info, step, boundary=guard)

            action, is_guided = guide.get_action(obs, policy)
            if not is_guided:
                action = guard.apply(_default_action(), bb)
            elif is_guided:
                guided_steps += 1

            flight = np.asarray(action["flight"], dtype=float)
            if flight.shape != (4,) or not np.all(np.isfinite(flight)):
                logic_errors += 1
                flight = np.array([0.55, 0.0, 0.0, 0.0])
            flight[0] = float(np.clip(flight[0], 0.0, 1.0))
            flight[1:] = np.clip(flight[1:], -1.0, 1.0)
            weapon = np.clip(np.asarray(action["weapon"], dtype=int), 0, 1)
            obs, _reward, term, trunc, info = env.step(
                {"flight": flight.astype(np.float32), "weapon": weapon})
            rec.capture_env(env)
            steps = step + 1
            if info.get("in_zone"):
                in_zone_steps += 1
            own_fired = int(info.get("own_fired", own_fired))
            enemy_fired = int(info.get("enemy_fired", enemy_fired))
            if is_safety_violation(obs, env.cfg):
                safety += 1
            if term or trunc:
                reason = info.get("terminated_reason")
                break
    finally:
        env.close()

    executed = set(bb.tick_counts)
    ooda_executed = [n for n in OODA_ALL if n in executed]
    summary = {
        "seed": int(seed), "steps": steps, "terminated_reason": reason,
        "logic_errors": logic_errors, "safety_violations": safety,
        "own_fired": own_fired, "enemy_fired": enemy_fired,
        "in_zone_steps": in_zone_steps, "frames": len(rec),
        "ooda_executed_coverage": len(ooda_executed) / len(OODA_ALL),
        "ooda_executed_missing": [n for n in OODA_ALL if n not in executed],
        "guided_steps": guided_steps,
        "is_guided_ratio": (guided_steps / steps) if steps else 0.0,
        "wall_s": round(time.time() - t0, 1),
        "tick_counts": dict(bb.tick_counts),
        "tactical_ooda_coverage": tactical_coverage()[0],
        "boundary": guard.summary(),
    }
    return rec, summary, guard.log, tree


def _save_tree_assets(tree: dict, out_dir: Path):
    (out_dir / "tactical_tree.txt").write_text(text_tree(tree), encoding="utf-8")
    dot_path = out_dir / "tactical_tree.dot"
    dot_path.write_text(tree_to_dot(tree, "bt_tactical_2_6"), encoding="utf-8")
    png_path = out_dir / "tactical_tree.png"
    dot_exe = shutil.which("dot")
    if dot_exe:
        try:
            subprocess.run([dot_exe, "-Tpng", str(dot_path), "-o", str(png_path)],
                           check=True, capture_output=True, timeout=30)
        except Exception:
            png_path = None
    else:
        png_path = None
    return {"text": str(out_dir / "tactical_tree.txt"), "dot": str(dot_path),
            "png": str(png_path) if png_path else None}


def _guard_config(args) -> dict | None:
    """可选：在原点放置圆形禁飞区，用于真实环境中的边界守卫联调演示。"""
    if not getattr(args, "guard_demo", False):
        return None
    return {"no_fly_zones": [{"name": "demo_origin", "type": "circle",
                              "center": [0.0, 0.0],
                              "radius_m": float(args.guard_demo_radius)}]}


def _run_one(seed: int, args, out_dir: Path) -> dict:
    rec, summary, events, tree = run_episode(
        seed, max_steps=args.max_steps, compute_envelope=not args.fast,
        guard_config=_guard_config(args))
    csv_path = out_dir / f"guidance_seed{seed}.csv"
    rec.save_csv(csv_path)
    acmi_path = None
    if not args.no_acmi:
        acmi_path = export_acmi(
            rec.frames, out_dir / f"guidance_seed{seed}.acmi",
            lat0_deg=GEO_LAT0_DEG, lon0_deg=GEO_LON0_DEG,
            title=f"Guidance BT 1v1 (seed {seed})")
    events_path = out_dir / f"guidance_seed{seed}.events.json"
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    summary.update({"csv": str(csv_path),
                    "acmi": str(acmi_path) if acmi_path else None,
                    "events": str(events_path),
                    "tree_id": tree.get("bt_id", "")})
    summary_path = out_dir / f"guidance_seed{seed}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="2.6 边界约束 + 战术树 + GuideInterface 联调")
    ap.add_argument("--seeds", default="11,22,33", help="逗号分隔种子（默认 11,22,33）")
    ap.add_argument("--max-steps", type=int, default=240, help="最大决策步（1 Hz）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/guidance/）")
    ap.add_argument("--fast", action="store_true", help="跳过攻击区解算（冒烟用）")
    ap.add_argument("--no-acmi", action="store_true", help="不导出 ACMI")
    ap.add_argument("--guard-demo", action="store_true",
                    help="在原点放置圆形禁飞区，联调演示边界守卫触发（默认关闭）")
    ap.add_argument("--guard-demo-radius", type=float, default=20000.0,
                    help="--guard-demo 的禁飞区半径 m（默认 20000）")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总行")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else _HERE / "outputs" / "guidance"
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = tactical_tree_json()
    errors = validate_tactical_tree()
    coverage, missing = tactical_coverage()
    if errors:
        print(f"[错误] 战术树校验失败：{errors}", file=sys.stderr)
        return 2
    assets = _save_tree_assets(tree, out_dir)
    if not args.quiet:
        print(f"[战术树] OODA 覆盖 {coverage:.0%}（缺失：{'、'.join(missing) or '无'}）")
        print(text_tree(tree))
        print(f"[运行] seeds={seeds} max_steps={args.max_steps} 输出={out_dir}")
        if assets["png"]:
            print(f"[可视化] {assets['png']}")
        else:
            print("[可视化] graphviz 不可用，保留 .txt/.dot")

    summaries = []
    for seed in seeds:
        summary = _run_one(seed, args, out_dir)
        summaries.append(summary)
        b = summary["boundary"]
        print(f"[seed {seed}] 步数={summary['steps']:<3d} 结局={summary['terminated_reason'] or '未终止'} "
              f"违规={summary['safety_violations']} 逻辑错误={summary['logic_errors']} "
              f"OODA执行={summary['ooda_executed_coverage']:.0%} "
              f"引导={summary['is_guided_ratio']:.0%} "
              f"边界触发={b['boundary_trigger_count']}（{b['per_constraint']}）")

    total_steps = sum(s["steps"] for s in summaries)
    total_safety = sum(s["safety_violations"] for s in summaries)
    total_trigger = sum(s["boundary"]["boundary_trigger_count"] for s in summaries)
    print(f"[汇总] {len(summaries)} 局，总步数 {total_steps}，安全违规 {total_safety}，"
          f"边界触发 {total_trigger}，产物目录 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
