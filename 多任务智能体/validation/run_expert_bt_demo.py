# -*- coding: utf-8 -*-
"""
validation/run_expert_bt_demo.py（阶段三最小演示：行为树选择专家智能体）
=======================================================================
用 py_trees 任务选择树按任务名调用任务级专家智能体（默认 strike + patrol），
每个任务跑一小段 BVRCombatEnv 仿真（不训练），保存：
    <out>/<task>.csv              逐帧 ActionV1/奖励/态势
    <out>/bt_expert_demo.csv      合并记录
    <out>/demo_summary.json       每任务步数/奖励/状态/模型加载
    <out>/demo_log.json           行为树事件日志
    <out>/demo_screenshot.png     轨迹指标图（中期检查汇报用）

用法（DC 环境，仓库根目录）：
    python "多任务智能体\\validation\\run_expert_bt_demo.py"
    python ... --tasks strike,patrol,defense,ew --steps 24
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = _HERE.parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from agents.bt_expert import (build_expert_tree, records_to_csv,  # noqa: E402
                              run_missions)
from agents.expert_registry import DEFAULT_REGISTRY  # noqa: E402
from common.paths import (DOCS_ASSETS_DIR, MODEL_DIR, RESULTS_DIR,  # noqa: E402
                          ensure_writable_dir)


def _default_out() -> str:
    """优先写 docs/assets；代码目录只读时回退到 validation/results。"""
    preferred = DOCS_ASSETS_DIR / "stage3_expert_demo"
    try:
        ensure_writable_dir(preferred, label="行为树演示输出目录")
        return str(preferred)
    except RuntimeError as exc:
        fallback = RESULTS_DIR / "stage3_expert_demo"
        print(f"[提示] {exc}\n[提示] 改用可写目录：{fallback}", file=sys.stderr)
        return str(fallback)


def _plot(results: dict, out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_ids = [k for k, v in results.items() if v.get("result")]
    if not task_ids:
        return out_png
    fig, axes = plt.subplots(len(task_ids), 2, figsize=(12, 3.2 * len(task_ids)),
                             squeeze=False)
    for row, task_id in enumerate(task_ids):
        recs = results[task_id]["result"]["records"]
        steps = [r["step"] for r in recs]
        actions = np.asarray([r["action_v1"] for r in recs], dtype=float)
        rewards = np.asarray([r["reward"] for r in recs], dtype=float)
        ax = axes[row][0]
        for idx, label in enumerate(("throttle", "roll", "pitch", "yaw")):
            ax.plot(steps, actions[:, idx], label=label, linewidth=1.2)
        ax.set_title(f"{task_id}: ActionV1 continuous dims")
        ax.set_xlabel("step")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

        ax2 = axes[row][1]
        ax2.plot(steps, rewards, color="#c0392b", label="reward", linewidth=1.2)
        ax2.set_title(f"{task_id}: step reward (total {rewards.sum():.2f})")
        ax2.set_xlabel("step")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7)
    fig.suptitle("Stage-3 Expert Agent BT Demo (ActionV1)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser(description="行为树选择任务级专家智能体最小演示")
    ap.add_argument("--tasks", default="strike,patrol")
    ap.add_argument("--steps", type=int, default=24, help="每个任务最大决策步")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out-dir", default=None,
                    help="产物目录；缺省优先 docs/assets，不可写时回退 results/")
    ap.add_argument("--no-load-models", action="store_true",
                    help="不加载已训练模型（用随机初始化策略演示接口）")
    args = ap.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    try:
        canonical = [DEFAULT_REGISTRY.resolve(t) for t in tasks]
    except KeyError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    if any(DEFAULT_REGISTRY.get(t).tier != "task" for t in canonical):
        print("[错误] 演示仅支持任务级智能体（strike/patrol/defense/ew）", file=sys.stderr)
        return 2

    out_dir = ensure_writable_dir(args.out_dir or _default_out(),
                                  label="行为树演示输出目录")
    root, bb = build_expert_tree(tasks=tuple(canonical), max_steps=args.steps,
                                 seed=args.seed, load_models=not args.no_load_models,
                                 model_dir=args.model_dir)
    print(f"[行为树] 根节点={root.name} 任务={canonical} 每任务≤{args.steps}步 "
          f"模型加载={'否' if args.no_load_models else '是'}")
    t0 = datetime.now()
    results = run_missions(root, bb, canonical,
                           max_ticks_per_mission=args.steps + 8,
                           registry=DEFAULT_REGISTRY)
    wall = (datetime.now() - t0).total_seconds()

    all_records = []
    for task_id, row in results.items():
        result = row.get("result") or {}
        records = result.get("records", [])
        all_records.extend(records)
        if records:
            records_to_csv(records, out_dir / f"{task_id}.csv")
        print(f"  [{'成功' if row['status'] == 'SUCCESS' else '失败'}] {task_id}: "
              f"{result.get('steps', 0)} 步, 累计奖励 {result.get('total_reward', 0):.3f}, "
              f"结局 {result.get('terminated_reason')}")
    if all_records:
        records_to_csv(all_records, out_dir / "bt_expert_demo.csv")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "wall_s": round(wall, 2),
        "tasks": canonical,
        "max_steps": args.steps,
        "seed": args.seed,
        "models_loaded": not args.no_load_models,
        "results": results,
    }
    (out_dir / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "demo_log.json").write_text(
        json.dumps(bb.logs, ensure_ascii=False, indent=2), encoding="utf-8")
    png = _plot(results, out_dir / "demo_screenshot.png")
    print(f"[产物] {out_dir}")
    print(f"  CSV: {out_dir / 'bt_expert_demo.csv'}")
    print(f"  截图: {png}")
    ok = all(row["status"] == "SUCCESS" for row in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
