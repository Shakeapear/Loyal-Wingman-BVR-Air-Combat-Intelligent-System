# -*- coding: utf-8 -*-
"""
引导策略行为树/run_2_7_validation.py（步骤 2.7 D2.7-1 验证运行器）
================================================================
对固定 20 组态势分别执行【手写战术树】与【LLM 生成树】，统一使用：
- 同一 BVRCombatEnv 随机初始态势（seed 复现）；
- 同一 BoundaryConstraints 边界守卫；
- 同一 GuideInterface（max_guide_steps = max_steps）；
- 同一 max_steps=240、录制/日志口径。

用法（在 引导策略行为树/ 目录下）：
    # 手写树（2.6 tactical_tree_json）
    python -X utf8 run_2_7_validation.py --tree handwritten --seeds 1-20 --out-dir outputs/validation_2_7/handwritten
    # LLM 树（2.5 在线 Round 1 收敛产物）
    python -X utf8 run_2_7_validation.py --tree llm --seeds 1-20 --out-dir outputs/validation_2_7/llm
    # 汇总对比 + 图表（写入 docs/assets/2.7_validation）
    python -X utf8 run_2_7_validation.py --summarize ^
        --handwritten-dir outputs/validation_2_7/handwritten ^
        --llm-dir outputs/validation_2_7/llm --out-dir ..\\docs\\assets\\2.7_validation

产物（每棵树）：
    metrics.csv / situation_table.csv / aggregate.json
    ep_seed<N>.csv / .acmi / .summary.json / .node_trace.json / .events.json
汇总模式：
    comparison.json / comparison.md / situation_compare.csv / *.png
"""
from __future__ import annotations

import argparse
import json
import math
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
from common.constants import (FT2M, GEO_LAT0_DEG, GEO_LON0_DEG,     # noqa: E402
                              wrap_deg180)
from llm_bt.boundary_constraints import BoundaryConstraints         # noqa: E402
from llm_bt.executor import Blackboard, build_tree, tick_once       # noqa: E402
from llm_bt.guide_interface import GuideInterface                   # noqa: E402
from llm_bt.maneuvers import ManeuverRegistry                       # noqa: E402
from llm_bt.schema import (ACTIONS, CONDITIONS, OODA_ALL,            # noqa: E402
                           ooda_coverage, validate_tree_json)
from llm_bt.tactical_execution_tree import (tactical_tree_80_120_json,
                                            tactical_tree_json)     # noqa: E402
from llm_bt.tester import is_safety_violation                       # noqa: E402
from visualization.acmi import export_acmi                          # noqa: E402
from visualization.recorder import EpisodeRecorder                  # noqa: E402

LLM_TREE_DEFAULT = _REPO_ROOT / "docs" / "assets" / "2.5_online_r2" / "bt_final.json"
LEGAL_NODES = set(ACTIONS) | set(CONDITIONS)
SCENES = {
    "default": {"max_steps": 240},
    "bvr80-120": {"scene_dist_min_m": 80000.0,
                  "scene_dist_max_m": 120000.0, "max_steps": 600},
}


# ----------------------------------------------------------------------
# 参数/树/种子
# ----------------------------------------------------------------------
def parse_seeds(spec: str) -> list:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def load_tree(tree: str, tree_json: str | None, tree_label: str | None):
    path = tree_json or (str(LLM_TREE_DEFAULT) if tree == "llm" else None)
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"树 JSON 不存在: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        label = tree_label or ("llm" if tree == "llm" else p.stem)
        source = str(p)
    elif tree in ("handwritten", "manual"):
        data = tactical_tree_json()
        label = tree_label or "handwritten"
        source = "llm_bt/tactical_execution_tree.py::tactical_tree_json()"
    elif tree in ("handwritten_80_120", "maneuver"):
        data = tactical_tree_80_120_json()
        label = tree_label or "handwritten_80_120"
        source = "llm_bt/tactical_execution_tree.py::tactical_tree_80_120_json()"
    else:
        p = Path(tree)
        if not p.is_file():
            raise FileNotFoundError(f"未知树或路径不存在: {tree}")
        data = json.loads(p.read_text(encoding="utf-8"))
        label = tree_label or p.stem
        source = str(p)
    try:
        maneuver_ids = set(ManeuverRegistry.from_csv().ids())
    except Exception:
        maneuver_ids = None
    errors = validate_tree_json(data, maneuver_ids=maneuver_ids)
    if errors:
        raise ValueError(f"行为树 JSON 校验失败: {errors}")
    return data, label, source


# ----------------------------------------------------------------------
# 单局
# ----------------------------------------------------------------------
def _initial_situation(env, obs) -> dict:
    frame = env.get_viz_frame()
    own, enemy = frame["own"], frame["enemy"]
    dx = enemy["pos"][0] - own["pos"][0]
    dy = enemy["pos"][1] - own["pos"][1]
    bearing_abs = math.degrees(math.atan2(dx, dy))
    rel_bearing = wrap_deg180(bearing_abs - math.degrees(own["psi_rad"]))
    return {
        "dist_km": round(frame["dist_m"] / 1000.0, 2),
        "bearing_deg": round(rel_bearing, 1),
        "own_alt_ft": round(own["h_sl_m"] / FT2M, 0),
        "own_mach": round(float(own["mach"]), 3),
        "own_v_mps": round(float(own["vtrue_mps"]), 1),
        "enemy_alt_ft": round(enemy["h_sl_m"] / FT2M, 0),
        "enemy_mach": round(float(enemy["mach"]), 3),
    }


def validate_node_trace(log: list, steps: int, tick_counts: dict) -> dict:
    """校验节点日志：step 单调/无缺失、节点名合法、OODA 执行覆盖。"""
    entries = [{"step": int(s), "node": str(n), "status": str(st)} for s, n, st in log]
    step_seq = [e["step"] for e in entries]
    step_set = set(step_seq)
    missing_steps = [s for s in range(int(steps)) if s not in step_set]
    illegal = sorted({e["node"] for e in entries if e["node"] not in LEGAL_NODES})
    executed = set(tick_counts)
    ooda_hit = [n for n in OODA_ALL if n in executed]
    monotonic = all(a <= b for a, b in zip(step_seq, step_seq[1:]))
    integrity = (bool(entries) and monotonic and not missing_steps and not illegal)
    ooda_complete = len(ooda_hit) == len(OODA_ALL)
    return {
        "ok": bool(integrity),
        "ooda_complete": bool(ooda_complete),
        "entries": len(entries),
        "first_step": step_seq[0] if step_seq else None,
        "last_step": step_seq[-1] if step_seq else None,
        "monotonic": bool(monotonic),
        "missing_steps": missing_steps,
        "illegal_nodes": illegal,
        "ooda_executed_coverage": len(ooda_hit) / len(OODA_ALL),
        "ooda_executed_missing": [n for n in OODA_ALL if n not in executed],
        "trace": entries,
    }


def run_episode(tree: dict, seed: int, max_steps: int = 240,
                compute_envelope: bool = True,
                scene: str = "default") -> dict:
    env_cfg = {"compute_envelope": bool(compute_envelope),
               "max_steps": int(max_steps)}
    env_cfg.update({k: v for k, v in SCENES.get(scene, {}).items()
                    if k != "max_steps"})
    env = BVRCombatEnv(config=env_cfg)
    bb = Blackboard(env)
    root = build_tree(tree, bb)
    guard = BoundaryConstraints(env_cfg=env.cfg)
    guide = GuideInterface(max_guide_steps=int(max_steps))
    rec = EpisodeRecorder(meta={"lat0_deg": GEO_LAT0_DEG, "lon0_deg": GEO_LON0_DEG})

    obs, info = env.reset(seed=int(seed))
    situation = _initial_situation(env, obs)
    rec.capture_env(env)

    logic_errors = safety = in_zone_steps = 0
    own_fired = enemy_fired = guided_steps = launches = invalid_launches = 0
    reason, steps, t0 = None, 0, time.time()
    try:
        for step in range(int(max_steps)):
            bb.set_frame(obs, info, step)

            def policy(_obs):
                return tick_once(root, bb, _obs, info, step, boundary=guard)

            action, is_guided = guide.get_action(obs, policy)
            if not is_guided:
                action = guard.apply({"flight": [0.55, 0.0, 0.0, 0.0],
                                      "weapon": [0, 0, 0]}, bb)
            else:
                guided_steps += 1

            flight = np.asarray(action["flight"], dtype=float)
            if flight.shape != (4,) or not np.all(np.isfinite(flight)):
                logic_errors += 1
                flight = np.array([0.55, 0.0, 0.0, 0.0])
            flight[0] = float(np.clip(flight[0], 0.0, 1.0))
            flight[1:] = np.clip(flight[1:], -1.0, 1.0)
            weapon = np.clip(np.asarray(action["weapon"], dtype=int), 0, 1)

            # 误发射判定：本步下达发射指令时，检查攻击区/余弹/目标探测
            if int(weapon[0]) == 1:
                launches += 1
                valid = (bool(info.get("in_zone", False))
                         and float(obs[12]) > 0.5 and float(obs[21]) > 0.5)
                if not valid:
                    invalid_launches += 1

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

    trace = validate_node_trace(bb.log, steps, dict(bb.tick_counts))
    survived = reason not in ("own_hit", "own_crash")
    summary = {
        "seed": int(seed), "steps": steps,
        "terminated_reason": reason, "survived": bool(survived),
        "enemy_hit": reason == "enemy_hit",
        "logic_errors": logic_errors, "safety_violations": safety,
        "own_fired": own_fired, "enemy_fired": enemy_fired,
        "in_zone_steps": in_zone_steps,
        "launch_commands": launches, "invalid_launches": invalid_launches,
        "misfire_rate": (invalid_launches / launches) if launches else 0.0,
        "guided_steps": guided_steps,
        "is_guided_ratio": (guided_steps / steps) if steps else 0.0,
        "ooda_executed_coverage": trace["ooda_executed_coverage"],
        "ooda_executed_missing": trace["ooda_executed_missing"],
        "ooda_complete": trace["ooda_complete"],
        "node_trace_ok": trace["ok"],
        "node_trace_entries": trace["entries"],
        "frames": len(rec),
        "wall_s": round(time.time() - t0, 1),
        "initial_situation": situation,
        "boundary": guard.summary(),
        "tick_counts": dict(bb.tick_counts),
    }
    return {"summary": summary, "trace": trace, "rec": rec,
            "events": guard.log, "tree": tree}


# ----------------------------------------------------------------------
# 批量/聚合
# ----------------------------------------------------------------------
def _write_metrics(rows: list, out_dir: Path):
    cols = ["seed", "survived", "terminated_reason", "enemy_hit", "steps",
            "safety_violations", "logic_errors", "own_fired", "enemy_fired",
            "launch_commands", "invalid_launches", "misfire_rate",
            "ooda_executed_coverage", "node_trace_ok", "is_guided_ratio",
            "boundary_trigger_count", "wall_s", "dist_km", "bearing_deg",
            "own_alt_ft", "own_mach", "enemy_alt_ft", "enemy_mach"]
    lines = [",".join(cols)]
    for s in rows:
        sit = s["initial_situation"]
        vals = {
            **s, "boundary_trigger_count": s["boundary"]["boundary_trigger_count"],
            **sit,
        }
        lines.append(",".join(str(int(vals[c])) if isinstance(vals[c], bool)
                              else str(vals[c]) for c in cols))
    (out_dir / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sit_cols = ["seed", "dist_km", "bearing_deg", "own_alt_ft", "own_mach",
                "own_v_mps", "enemy_alt_ft", "enemy_mach"]
    lines = [",".join(sit_cols)]
    for s in rows:
        vals = {"seed": s["seed"], **s["initial_situation"]}
        lines.append(",".join(str(vals[c]) for c in sit_cols))
    (out_dir / "situation_table.csv").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def aggregate(rows: list, tree_label: str, tree_id: str, tree_source: str,
              seeds: list, max_steps: int) -> dict:
    n = len(rows)

    def rate(key):
        return sum(1 for r in rows if r[key]) / n if n else 0.0

    reasons = {}
    for r in rows:
        reasons[r["terminated_reason"] or "none"] = reasons.get(
            r["terminated_reason"] or "none", 0) + 1
    launches = sum(r["launch_commands"] for r in rows)
    invalid = sum(r["invalid_launches"] for r in rows)
    return {
        "tree_label": tree_label, "tree_id": tree_id, "tree_source": tree_source,
        "seeds": seeds, "max_steps": int(max_steps), "n_episodes": n,
        "survival_rate": round(rate("survived"), 4),
        "hit_rate": round(rate("enemy_hit"), 4),
        "misfire_rate": round((invalid / launches) if launches else 0.0, 4),
        "launch_commands": launches, "invalid_launches": invalid,
        "safety_violations": sum(r["safety_violations"] for r in rows),
        "logic_errors": sum(r["logic_errors"] for r in rows),
        "ooda_static_coverage": 1.0,
        "avg_ooda_executed": round(sum(r["ooda_executed_coverage"] for r in rows) / n, 4) if n else 0.0,
        "full_ooda_episodes": sum(1 for r in rows if r["ooda_executed_coverage"] >= 1.0),
        "trace_ok_episodes": sum(1 for r in rows if r["node_trace_ok"]),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 1) if n else 0.0,
        "avg_guided_ratio": round(sum(r["is_guided_ratio"] for r in rows) / n, 4) if n else 0.0,
        "avg_wall_s": round(sum(r["wall_s"] for r in rows) / n, 2) if n else 0.0,
        "boundary_trigger_count": sum(r["boundary"]["boundary_trigger_count"] for r in rows),
        "termination_counts": reasons,
        "own_fired": sum(r["own_fired"] for r in rows),
        "enemy_fired": sum(r["enemy_fired"] for r in rows),
    }


def run_tree(tree: dict, label: str, source: str, seeds: list, max_steps: int,
             out_dir: Path, compute_envelope: bool = True,
             no_acmi: bool = False, scene: str = "default") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, all_events = [], {}
    for seed in seeds:
        result = run_episode(tree, seed, max_steps=max_steps,
                             compute_envelope=compute_envelope, scene=scene)
        s, rec = result["summary"], result["rec"]
        s["tree_label"] = label
        rows.append(s)
        csv_path = out_dir / f"ep_seed{seed}.csv"
        rec.save_csv(csv_path)
        acmi = None
        if not no_acmi:
            acmi = export_acmi(rec.frames, out_dir / f"ep_seed{seed}.acmi",
                               lat0_deg=GEO_LAT0_DEG, lon0_deg=GEO_LON0_DEG,
                               title=f"{label} 1v1 (seed {seed})")
        (out_dir / f"ep_seed{seed}.node_trace.json").write_text(
            json.dumps(result["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
        all_events[str(seed)] = result["events"]
        s.update({"csv": str(csv_path), "acmi": str(acmi) if acmi else None,
                  "node_trace": str(out_dir / f"ep_seed{seed}.node_trace.json"),
                  "events": str(out_dir / f"ep_seed{seed}.events.json")})
        (out_dir / f"ep_seed{seed}.summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"ep_seed{seed}.events.json").write_text(
            json.dumps(result["events"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{label} seed {seed:>2}] {s['terminated_reason'] or '未终止':<10} "
              f"存活={int(s['survived'])} 命中={int(s['enemy_hit'])} "
              f"步数={s['steps']:<3d} 违规={s['safety_violations']} "
              f"OODA={s['ooda_executed_coverage']:.0%} "
              f"trace={'OK' if s['node_trace_ok'] else 'FAIL'} "
              f"发射={s['launch_commands']}(非法{s['invalid_launches']})", flush=True)
    _write_metrics(rows, out_dir)
    agg = aggregate(rows, label, str(tree.get("bt_id", "")), source, seeds, max_steps)
    (out_dir / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "events_all.json").write_text(
        json.dumps(all_events, ensure_ascii=False, indent=2), encoding="utf-8")
    return agg


# ----------------------------------------------------------------------
# 汇总对比
# ----------------------------------------------------------------------
def _read_metrics(d: Path) -> dict:
    import csv
    with (d / "metrics.csv").open(encoding="utf-8") as f:
        return {int(r["seed"]): r for r in csv.DictReader(f)}


def summarize(hw_dir: Path, llm_dir: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hw = json.loads((hw_dir / "aggregate.json").read_text(encoding="utf-8"))
    llm = json.loads((llm_dir / "aggregate.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = ["survival_rate", "hit_rate", "misfire_rate", "safety_violations",
            "ooda_static_coverage", "avg_ooda_executed", "full_ooda_episodes",
            "trace_ok_episodes", "avg_steps", "avg_guided_ratio", "avg_wall_s",
            "launch_commands", "invalid_launches", "boundary_trigger_count"]
    comparison = {"handwritten": hw, "llm": llm,
                  "delta_llm_minus_handwritten": {
                      k: round(llm[k] - hw[k], 4) if isinstance(hw[k], (int, float)) else None
                      for k in keys}}
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐 seed 对比表
    hm, lm = _read_metrics(hw_dir), _read_metrics(llm_dir)
    lines = ["seed,dist_km,bearing_deg,own_alt_ft,own_mach,"
             "hw_reason,hw_survived,hw_hit,hw_steps,hw_safety,hw_ooda,"
             "llm_reason,llm_survived,llm_hit,llm_steps,llm_safety,llm_ooda"]
    for seed in sorted(hm):
        h, l = hm[seed], lm[seed]
        lines.append(",".join(str(x) for x in [
            seed, h["dist_km"], h["bearing_deg"], h["own_alt_ft"], h["own_mach"],
            h["terminated_reason"], h["survived"], h["enemy_hit"], h["steps"],
            h["safety_violations"], h["ooda_executed_coverage"],
            l["terminated_reason"], l["survived"], l["enemy_hit"], l["steps"],
            l["safety_violations"], l["ooda_executed_coverage"]]))
    (out_dir / "situation_compare.csv").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")

    # 图 1：存活率/命中率
    labels = ["survival", "hit"]
    hw_vals = [hw["survival_rate"], hw["hit_rate"]]
    llm_vals = [llm["survival_rate"], llm["hit_rate"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, hw_vals, 0.36, label="handwritten")
    ax.bar(x + 0.18, llm_vals, 0.36, label="LLM tree")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    for i, (a, b) in enumerate(zip(hw_vals, llm_vals)):
        ax.text(i - 0.18, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
        ax.text(i + 0.18, b + 0.02, f"{b:.0%}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_survival_hit.png", dpi=120)
    plt.close(fig)

    # 图 2：终止原因分布
    reasons = sorted(set(hw["termination_counts"]) | set(llm["termination_counts"]))
    x = np.arange(len(reasons))
    hv = [hw["termination_counts"].get(r, 0) for r in reasons]
    lv = [llm["termination_counts"].get(r, 0) for r in reasons]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, hv, 0.36, label="handwritten")
    ax.bar(x + 0.18, lv, 0.36, label="LLM tree")
    ax.set_xticks(x, reasons, rotation=15)
    ax.set_ylabel("episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_termination.png", dpi=120)
    plt.close(fig)

    # 图 3：逐 seed 步数
    seeds = sorted(hm)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(seeds, [int(hm[s]["steps"]) for s in seeds], marker="o", label="handwritten")
    ax.plot(seeds, [int(lm[s]["steps"]) for s in seeds], marker="s", label="LLM tree")
    ax.set_xlabel("seed")
    ax.set_ylabel("steps")
    ax.set_xticks(seeds)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_per_seed_steps.png", dpi=120)
    plt.close(fig)

    # comparison.md
    md = ["# 2.7 两树对比（自动生成）", "",
          "| 指标 | 手写树 | LLM 树 | Δ(LLM-手写) |", "|---|---:|---:|---:|"]
    names = {
        "survival_rate": "存活率", "hit_rate": "命中率",
        "misfire_rate": "误发射率", "safety_violations": "安全违规步数",
        "ooda_static_coverage": "OODA 静态覆盖", "avg_ooda_executed": "OODA 执行覆盖均值",
        "full_ooda_episodes": "OODA 8/8 局数", "trace_ok_episodes": "trace 校验通过局数",
        "avg_steps": "平均步数", "avg_guided_ratio": "平均引导比例",
        "avg_wall_s": "平均 wall time (s)", "launch_commands": "发射指令数",
        "invalid_launches": "非法发射数", "boundary_trigger_count": "边界触发数",
    }
    for k in keys:
        md.append(f"| {names[k]} | {hw[k]} | {llm[k]} | "
                  f"{comparison['delta_llm_minus_handwritten'][k]} |")
    md += ["", f"- 手写树终止分布：`{hw['termination_counts']}`",
           f"- LLM 树终止分布：`{llm['termination_counts']}`",
           "- 生成/开发耗时（参考，非严格人工工时）：LLM 在线 Round 1 约 24 s；"
           "手写树实现按 2.6 commit 间隔约 7 min（2.5 `30165e1` → 2.6 `86dd2c8`）。"]
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return comparison


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="2.7 20 组态势行为树验证/对比")
    ap.add_argument("--tree", default="handwritten",
                    help="handwritten | handwritten_80_120 | llm | 树 JSON 路径")
    ap.add_argument("--tree-json", default=None, help="树 JSON 路径（覆盖 --tree）")
    ap.add_argument("--tree-label", default=None, help="结果标签（默认 handwritten/llm）")
    ap.add_argument("--seeds", default="1-20", help="种子，如 1-20 或 1,2,3")
    ap.add_argument("--scene", default="default", choices=sorted(SCENES),
                    help="场景：default(30–80km/240) 或 bvr80-120(80–120km/600)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 时使用场景默认步数")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fast", action="store_true", help="不解算攻击区（冒烟）")
    ap.add_argument("--no-acmi", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="汇总两棵树并出图")
    ap.add_argument("--handwritten-dir", default=None)
    ap.add_argument("--llm-dir", default=None)
    args = ap.parse_args(argv)

    if args.summarize:
        if not args.handwritten_dir or not args.llm_dir:
            print("[错误] --summarize 需要 --handwritten-dir 与 --llm-dir", file=sys.stderr)
            return 2
        out = Path(args.out_dir) if args.out_dir else _HERE / "outputs" / "validation_2_7"
        comp = summarize(Path(args.handwritten_dir), Path(args.llm_dir), out)
        print(f"[对比] 手写存活率 {comp['handwritten']['survival_rate']:.0%}，"
              f"LLM 存活率 {comp['llm']['survival_rate']:.0%}；输出 {out}")
        return 0

    try:
        tree, label, source = load_tree(args.tree, args.tree_json, args.tree_label)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 2
    seeds = parse_seeds(args.seeds)
    max_steps = int(args.max_steps or SCENES.get(args.scene, {}).get("max_steps", 240))
    out = Path(args.out_dir) if args.out_dir else _HERE / "outputs" / "validation_2_7" / label
    print(f"[树] {label} ({source}) 静态 OODA={ooda_coverage(tree)[0]:.0%}  "
          f"scene={args.scene} steps={max_steps} seeds={seeds}")
    agg = run_tree(tree, label, source, seeds, max_steps, out,
                   compute_envelope=not args.fast, no_acmi=args.no_acmi,
                   scene=args.scene)
    print(f"[汇总] 存活率 {agg['survival_rate']:.0%}，命中率 {agg['hit_rate']:.0%}，"
          f"误发射率 {agg['misfire_rate']:.0%}，安全违规 {agg['safety_violations']}，"
          f"OODA 执行均值 {agg['avg_ooda_executed']:.0%}，trace 通过 "
          f"{agg['trace_ok_episodes']}/{agg['n_episodes']}；输出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
