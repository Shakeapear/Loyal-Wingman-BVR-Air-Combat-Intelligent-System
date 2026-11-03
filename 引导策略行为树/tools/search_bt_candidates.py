# -*- coding: utf-8 -*-
"""LLM 候选行为树闭环搜索（外部 harness 评分，不依赖 LLM 自评）。

流程：
  父树评分 → LLM 生成候选 → 静态校验 → 固定 seeds 仿真评分
  → 失败 seed/指标反馈 → 下一代候选 → 50 seeds 复评 → 选最优

用法（在 引导策略行为树/ 目录下）：
  python -X utf8 tools/search_bt_candidates.py \
    --provider deepseek --model deepseek-flash \
    --scene bvr80-120 --generations 3 --candidates 8 \
    --seeds 1-20 --final-seeds 1-50 \
    --parent docs/../assets/.../bt_final.json \
    --out-dir outputs/search_80_120

API Key 只从环境变量读取；不得写入命令、日志或产物。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "引导策略行为树"
for _p in (HERE, ROOT / "多任务智能体", ROOT / "可视化工具"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm_bt.llm_client import LLMClient, extract_json          # noqa: E402
from llm_bt.maneuvers import ManeuverRegistry                 # noqa: E402
from llm_bt.prompts import SYSTEM_CODER, render_coder_prompt   # noqa: E402
from llm_bt.scenes import SCENES, scene_max_steps              # noqa: E402
from llm_bt.schema import ooda_coverage, validate_tree_json    # noqa: E402
from llm_bt.tree_io import text_tree                          # noqa: E402
from llm_bt.validator_bridge import validate_tree_static        # noqa: E402
from run_2_7_validation import run_episode                    # noqa: E402

DEFAULT_MISSION = (
    "超视距空战拦截：远距搜索并闭合接敌，雷达锁定后在 NEZ 内发射中距弹；"
    "导弹来袭时优先防御，燃油不足返航；全程满足安全包线。"
)


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _metrics(rows: list[dict], scene: str) -> dict:
    n = len(rows) or 1
    launches = sum(int(r.get("launch_commands", 0)) for r in rows)
    invalid = sum(int(r.get("invalid_launches", 0)) for r in rows)
    misfire = (invalid / launches) if launches else 0.0
    survival = sum(bool(r.get("survived")) for r in rows) / n
    hit = sum(bool(r.get("enemy_hit")) for r in rows) / n
    ooda = sum(float(r.get("ooda_executed_coverage", 0.0)) for r in rows) / n
    safety = sum(int(r.get("safety_violations", 0)) for r in rows)
    trace_ok = all(bool(r.get("node_trace_ok")) for r in rows)
    hard = (safety == 0 and misfire == 0.0 and trace_ok and ooda >= 0.95)
    score = 0.45 * survival + 0.45 * hit + 0.10 * ooda
    if not hard:
        score -= 1.0
    worst = sorted(
        rows,
        key=lambda r: (bool(r.get("survived")), bool(r.get("enemy_hit")),
                       float(r.get("ooda_executed_coverage", 0.0))),
    )[:5]
    return {
        "scene": scene,
        "n": len(rows),
        "survival_rate": round(survival, 4),
        "hit_rate": round(hit, 4),
        "misfire_rate": round(misfire, 4),
        "safety_violations": safety,
        "ooda_executed_coverage": round(ooda, 4),
        "node_trace_ok": trace_ok,
        "hard_constraints_ok": hard,
        "score": round(score, 4),
        "worst_seeds": [
            {"seed": r.get("seed"), "reason": r.get("terminated_reason"),
             "survived": bool(r.get("survived")),
             "enemy_hit": bool(r.get("enemy_hit")),
             "steps": r.get("steps"),
             "ooda": r.get("ooda_executed_coverage")}
            for r in worst
        ],
    }


def evaluate(tree: dict, seeds: list[int], scene: str) -> dict:
    rows = []
    for seed in seeds:
        result = run_episode(tree, seed, max_steps=scene_max_steps(scene),
                             compute_envelope=True, scene=scene)
        rows.append(result["summary"])
    metrics = _metrics(rows, scene)
    metrics["seeds"] = list(seeds)
    return metrics


def _feedback(best: dict, recent: list[dict]) -> str:
    lines = [
        f"外部 harness 评分（{best['scene']}，{best['n']} seeds）：",
        f"- 存活 {best['survival_rate']:.0%}；命中 {best['hit_rate']:.0%}；"
        f"误发射 {best['misfire_rate']:.0%}；安全违规 {best['safety_violations']}；"
        f"OODA {best['ooda_executed_coverage']:.0%}；score={best['score']}",
        "- 失败/最差 seed：",
    ]
    for row in best["worst_seeds"]:
        lines.append(
            f"  * seed={row['seed']} reason={row['reason']} "
            f"survived={row['survived']} hit={row['enemy_hit']} "
            f"steps={row['steps']} ooda={row['ooda']}")
    if recent:
        lines.append("- 最近候选（不要重复低分结构）：")
        for item in recent[-4:]:
            lines.append(
                f"  * score={item['metrics']['score']} "
                f"survival={item['metrics']['survival_rate']:.0%} "
                f"hit={item['metrics']['hit_rate']:.0%}")
    lines += [
        "- 只做小步结构修改，保留已验证的搜索/闭合/防御/发射分支；",
        "- 必须使用 execute_maneuver 注册招式，禁止发明动作；",
        "- 80–120 km 场景必须远距搜索（search_pattern）并 NEZ 内发射；",
        "- 防御默认 defensive_spiral；不要用 notch；",
        "- 不得让 fire_missile 与 launch_and_crank 在同一 Sequence 串联。",
    ]
    return "\n".join(lines)


def _load_tree(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLM 候选行为树外部评分搜索")
    ap.add_argument("--provider", default="deepseek",
                    choices=["deepseek", "openai", "mock"])
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--mission", default=DEFAULT_MISSION)
    ap.add_argument("--scene", default="bvr80-120", choices=sorted(SCENES))
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--final-seeds", default="1-50")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--parent", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--archive-dir", default=None)
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    final_seeds = parse_seeds(args.final_seeds)
    run_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "outputs" / f"search_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = ManeuverRegistry.from_csv()
    maneuver_ids = set(registry.ids())

    parents = [Path(p) for p in args.parent]
    if not parents:
        parents = [
            ROOT / "docs" / "assets" / "3.4_llm_tree_80_120" / "bt_final.json",
        ]
    parent_trees = [_load_tree(p) for p in parents if p.is_file()]
    if not parent_trees:
        raise SystemExit("没有可用 --parent 树")

    print(f"[baseline] {len(parent_trees)} parent(s), seeds={seeds}, scene={args.scene}",
          flush=True)
    baseline = []
    for path, tree in zip([p for p in parents if p.is_file()], parent_trees):
        metrics = evaluate(tree, seeds, args.scene)
        baseline.append({"parent": str(path), "metrics": metrics})
        print(f"[baseline] {path.name}: score={metrics['score']} "
              f"survival={metrics['survival_rate']:.0%} "
              f"hit={metrics['hit_rate']:.0%}", flush=True)
    _save_json(run_dir / "baseline.json", baseline)
    best_item = max(baseline, key=lambda x: x["metrics"]["score"])
    best_tree = parent_trees[baseline.index(best_item)]
    best_metrics = best_item["metrics"]

    llm = LLMClient(provider=args.provider, model=args.model,
                    temperature=args.temperature)
    recent = []
    for gen in range(1, args.generations + 1):
        feedback = _feedback(best_metrics, recent)
        mission = (args.mission + "\n\n当前最优树（只做小步修改）：\n"
                   + text_tree(best_tree) + "\n")
        candidates = []
        for i in range(args.candidates):
            prompt = render_coder_prompt(mission, feedback)
            try:
                raw = llm.complete(SYSTEM_CODER, prompt, task="coder",
                                   context={"mission": mission,
                                            "feedback": feedback,
                                            "scene": args.scene,
                                            "generation": gen, "candidate": i})
                tree = extract_json(raw)
                tree["bt_id"] = f"search_g{gen}_c{i}"
                tree.setdefault("mission", args.mission)
                errors, _warnings = validate_tree_static(
                    tree, maneuver_ids=maneuver_ids)
                if errors:
                    print(f"[g{gen}c{i}] 静态失败: {errors[:2]}", flush=True)
                    continue
                metrics = evaluate(tree, seeds, args.scene)
                item = {"generation": gen, "candidate": i, "tree": tree,
                        "metrics": metrics, "raw": raw}
                candidates.append(item)
                recent.append(item)
                print(f"[g{gen}c{i}] score={metrics['score']} "
                      f"survival={metrics['survival_rate']:.0%} "
                      f"hit={metrics['hit_rate']:.0%}", flush=True)
            except Exception as exc:
                print(f"[g{gen}c{i}] 跳过: {type(exc).__name__}: {exc}",
                      flush=True)
        if not candidates:
            print(f"[g{gen}] 无有效候选", flush=True)
            continue
        candidates.sort(key=lambda x: x["metrics"]["score"], reverse=True)
        if candidates[0]["metrics"]["score"] > best_metrics["score"]:
            best_tree = candidates[0]["tree"]
            best_metrics = candidates[0]["metrics"]
        _save_json(run_dir / f"generation_{gen}.json",
                   [{k: v for k, v in c.items() if k != "tree"}
                    for c in candidates])
        _save_json(run_dir / f"best_g{gen}.json", best_tree)
        print(f"[g{gen}] best score={best_metrics['score']} "
              f"survival={best_metrics['survival_rate']:.0%} "
              f"hit={best_metrics['hit_rate']:.0%}", flush=True)

    # 最终 50 seeds 复核：对当前 best 与最后一代 top-k 重评。
    finalists = [{"name": "best", "tree": best_tree, "metrics": best_metrics}]
    if recent:
        recent.sort(key=lambda x: x["metrics"]["score"], reverse=True)
        for idx, item in enumerate(recent[:max(0, args.top_k - 1)], 1):
            finalists.append({"name": f"top{idx}", "tree": item["tree"],
                              "metrics": item["metrics"]})
    for item in finalists:
        item["final_metrics"] = evaluate(item["tree"], final_seeds, args.scene)
        print(f"[final] {item['name']} score={item['final_metrics']['score']} "
              f"survival={item['final_metrics']['survival_rate']:.0%} "
              f"hit={item['final_metrics']['hit_rate']:.0%}", flush=True)
    winner = max(finalists, key=lambda x: x["final_metrics"]["score"])
    _save_json(run_dir / "bt_best.json", winner["tree"])
    _save_json(run_dir / "final_metrics.json",
               {x["name"]: x["final_metrics"] for x in finalists})
    if args.archive_dir:
        target = Path(args.archive_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "bt_best.json", target / "bt_best.json")
        shutil.copy2(run_dir / "final_metrics.json", target / "final_metrics.json")
    print(f"[done] winner={winner['name']} output={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
