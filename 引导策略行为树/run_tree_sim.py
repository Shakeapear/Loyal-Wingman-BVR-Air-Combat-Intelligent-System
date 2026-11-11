# -*- coding: utf-8 -*-
"""
引导策略行为树/run_tree_sim.py（行为树 → 仿真 → 录制 集成运行器）
================================================================
把 LLM 生成的行为树接入 BVRCombatEnv 完整跑一局 1v1 双机对抗，并录制
全量数据（CSV + TacView ACMI），用于人工复盘与"接口/数据统一适配"验证：

    bt_final.json ──▶ llm_bt.executor（JSON→py_trees）──▶ BVRCombatEnv 1 Hz
                                                              │ get_viz_frame()
                                                              ▼
                                     可视化工具/visualization.EpisodeRecorder
                                                              │
                                              ├── CSV（逐帧全量数据，可离线回放）
                                              └── ACMI（TacView 3D 复盘）

用法（DC 环境，在 引导策略行为树/ 目录下）：
    # 默认树（examples/bt_final_离线样例.json）+ 默认 seed
    python -X utf8 run_tree_sim.py

    # 指定种子 / 树文件 / 输出目录
    python -X utf8 run_tree_sim.py --seed 22
    python -X utf8 run_tree_sim.py --tree outputs/<时间戳>/bt_final.json --seed 42

    # 批量跑多个种子做稳定性扫描（输出每局步数/结局汇总）
    python -X utf8 run_tree_sim.py --seeds 11,22,33,44,55,66,77,88

产物（默认 outputs/sim/）：
    bt_ep_seed<seed>.csv            逐帧全量数据（EpisodeRecorder 格式）
    bt_ep_seed<seed>.acmi           TacView 3D 回放文件（双击即看）
    bt_ep_seed<seed>.summary.json   本局指标与节点执行统计
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_HERE, _REPO_ROOT / "多任务智能体", _REPO_ROOT / "可视化工具"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import numpy as np

from common.bvr_combat_env import BVRCombatEnv                      # noqa: E402
from common.contracts import (EpisodeResult,                        # noqa: E402
                              ensure_within, sanitize_tag)
from common.errors import (BT_SCHEMA_INVALID, ProjectError, ValidationError,  # noqa: E402
                           exit_code_for)
from llm_bt.executor import Blackboard, build_tree, tick_once       # noqa: E402
from llm_bt.schema import validate_tree_json                        # noqa: E402
from llm_bt.sim_common import (export_episode_acmi, make_recorder,   # noqa: E402
                               parse_seeds)
from llm_bt.tester import is_safety_violation                       # noqa: E402
from llm_bt.tree_io import text_tree                                # noqa: E402


def default_tree_path() -> Path | None:
    """默认树：examples 归档样例 → 最近一次迭代的 bt_final.json。"""
    example = _HERE / "examples" / "bt_final_离线样例.json"
    if example.is_file():
        return example
    finals = sorted((_HERE / "outputs").glob("*/bt_final.json"))
    return finals[-1] if finals else None


def load_tree(path: Path) -> dict:
    tree = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_tree_json(tree)
    if errors:
        raise ValidationError(BT_SCHEMA_INVALID, "行为树 JSON 校验失败",
                              detail={"path": str(path), "errors": errors})
    return tree


def run_episode(tree: dict, seed: int, max_steps: int = 240,
                compute_envelope: bool = True) -> tuple:
    """执行一局完整对抗，返回 (recorder, summary)。"""
    env = BVRCombatEnv(config={"compute_envelope": bool(compute_envelope)})
    bb = Blackboard(env)
    root = build_tree(tree, bb)
    rec = make_recorder()
    obs, info = env.reset(seed=int(seed))
    rec.capture_env(env)                 # t=0 初始态势（TacView 首帧）

    logic_errors = safety = in_zone_steps = 0
    own_fired = enemy_fired = 0
    reason, steps, t0 = None, 0, time.time()
    try:
        for step in range(int(max_steps)):
            action = tick_once(root, bb, obs, info, step)
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
    summary = {
        "seed": int(seed), "steps": steps, "terminated_reason": reason,
        "logic_errors": logic_errors, "safety_violations": safety,
        "own_fired": own_fired, "enemy_fired": enemy_fired,
        "in_zone_steps": in_zone_steps, "frames": len(rec),
        "wall_s": round(time.time() - t0, 1),
        "tick_counts": dict(bb.tick_counts),
    }
    return rec, summary


def _run_one(tree: dict, seed: int, args, out_dir: Path, multi: bool = False) -> dict:
    rec, summary = run_episode(tree, seed, max_steps=args.max_steps,
                               compute_envelope=not args.fast)
    tag = sanitize_tag(args.tag, fallback="") if args.tag else ""
    tag = f"{tag}_seed{seed}" if (tag and multi) else (tag or f"seed{seed}")
    csv_path = ensure_within(out_dir / f"bt_ep_{tag}.csv", out_dir)
    rec.save_csv(csv_path)
    acmi_path = None
    if not args.no_acmi:
        acmi_path = export_episode_acmi(
            rec.frames, ensure_within(out_dir / f"bt_ep_{tag}.acmi", out_dir),
            title=f"BT {tree.get('bt_id', 'bt')} 1v1 (seed {seed})")
    record = EpisodeResult(**summary, tree_id=str(tree.get("bt_id", "")), tag=tag,
                           csv=str(csv_path),
                           acmi=str(acmi_path) if acmi_path else None).to_dict()
    record["tree"] = str(args.tree_resolved)
    summary_path = ensure_within(out_dir / f"bt_ep_{tag}.summary.json", out_dir)
    summary_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    summary.update({"csv": str(csv_path), "acmi": str(acmi_path),
                    "summary": str(summary_path)})
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="行为树接入仿真并录制双机对抗数据")
    ap.add_argument("--tree", default=None, help="行为树 JSON（默认 examples 样例）")
    ap.add_argument("--seed", type=int, default=22, help="单局种子")
    ap.add_argument("--seeds", default=None, help="批量种子，如 11,22,33（覆盖 --seed）")
    ap.add_argument("--max-steps", type=int, default=240, help="最大决策步（1 Hz）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/sim/）")
    ap.add_argument("--tag", default=None, help="文件名标签（单局时；默认 seedN）")
    ap.add_argument("--fast", action="store_true", help="跳过攻击区解算（冒烟用）")
    ap.add_argument("--no-acmi", action="store_true", help="不导出 ACMI")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总行")
    args = ap.parse_args(argv)

    tree_path = Path(args.tree) if args.tree else default_tree_path()
    if tree_path is None or not tree_path.is_file():
        print("[错误] 未找到行为树 JSON；请先运行 llm_bt_generator.py 或指定 --tree",
              file=sys.stderr)
        return 2
    args.tree_resolved = tree_path
    try:
        tree = load_tree(tree_path)
    except ProjectError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return exit_code_for(e)

    out_dir = Path(args.out_dir) if args.out_dir else _HERE / "outputs" / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        seeds = parse_seeds(args.seeds) if args.seeds else [args.seed]
    except ValueError:
        print(f"[错误] --seeds 需为逗号分隔整数或范围，实际 {args.seeds!r}",
              file=sys.stderr)
        return 2
    multi = len(seeds) > 1

    if not args.quiet:
        print(f"[树] {tree_path}")
        print(text_tree(tree))
        print(f"[运行] seeds={seeds}  max_steps={args.max_steps}  "
              f"envelope={'off' if args.fast else 'on'}  输出={out_dir}")

    summaries = []
    for seed in seeds:
        try:
            summary = _run_one(tree, seed, args, out_dir, multi=multi)
        except ProjectError as e:
            print(f"[错误] seed={seed}: {e}", file=sys.stderr)
            return exit_code_for(e)
        summaries.append(summary)
        print(f"[seed {seed}] 步数={summary['steps']:<3d} 结局={summary['terminated_reason'] or '未终止'} "
              f"违规={summary['safety_violations']} 逻辑错误={summary['logic_errors']} "
              f"本机发射={summary['own_fired']} 我弹{summary['enemy_fired']} 攻击区步={summary['in_zone_steps']}")
        print(f"          CSV : {summary['csv']}")
        if summary["acmi"]:
            print(f"          ACMI: {summary['acmi']}")
    if len(summaries) > 1:
        avg = sum(s["steps"] for s in summaries) / len(summaries)
        print(f"[汇总] {len(summaries)} 局，平均 {avg:.0f} 步，"
              f"最长 {max(s['steps'] for s in summaries)} 步，"
              f"可追溯统计见各 .summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
