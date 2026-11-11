# -*- coding: utf-8 -*-
"""
visualize_bt_vs_bt.py（从 BT-vs-BT 结果中挑最长对局并可视化）
================================================================
默认从 outputs/bt_vs_bt/{default_1-20,bvr80-120_1-20} 的结果里，为每个场景、
每棵树各挑一局“流程最长”（steps 最大）的对局重新录制并输出：

- 单局 2x2 仪表板 PNG：航迹 / 距离 / 高度 / 速度；
- 每个场景 2x4 汇总图 PNG（两局上下排列）；
- TacView ACMI（双击 3D 回放）与逐帧 CSV。

用法（在 引导策略行为树/ 目录下）：
    python -X utf8 visualize_bt_vs_bt.py
    python -X utf8 visualize_bt_vs_bt.py --out-dir outputs/bt_vs_bt/visualization
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_HERE, _REPO_ROOT / "多任务智能体", _REPO_ROOT / "可视化工具"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common.constants import GEO_LAT0_DEG, GEO_LON0_DEG
from llm_bt.scenes import env_config, scene_max_steps
from visualization.acmi import export_acmi
from visualization.recorder import EpisodeRecorder

from run_bt_vs_bt import load_named_tree, run_episode
RESULT_FILES = {
    "default": _HERE / "outputs" / "bt_vs_bt" / "default_1-20" / "result.json",
    "bvr80-120": _HERE / "outputs" / "bt_vs_bt" / "bvr80-120_1-20" / "result.json",
}
TREE_COLORS = {"handwritten": "#1f77b4", "bt_best": "#d62728"}
TREE_NAMES = {"handwritten": "handwritten", "bt_best": "LLM g1c2"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 选择“最长”对局
# ----------------------------------------------------------------------
def select_longest(results: dict) -> list[dict]:
    picked = []
    for scene, path in results.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        own_labels = sorted({r["own_tree"] for r in data["episodes"]})
        order = ["handwritten", "bt_best"]
        own_labels.sort(key=lambda x: order.index(x) if x in order else 99)
        for own in own_labels:
            eps = [r for r in data["episodes"] if r["own_tree"] == own]
            if not eps:
                continue
            best = max(eps, key=lambda r: (int(r["steps"]), -int(r["seed"])))
            picked.append({"scene": scene, "row": best})
    return picked


def select_longest_overall(results: dict) -> dict:
    """跨场景选 steps 最大的一局。"""
    best = None
    for scene, path in results.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["episodes"]:
            item = {"scene": scene, "row": row}
            if best is None or int(row["steps"]) > int(best["row"]["steps"]):
                best = item
    return best


# ----------------------------------------------------------------------
# 画图辅助
# ----------------------------------------------------------------------
def _pos_km(frames, side):
    return np.array([[f[side]["pos"][0] / 1000.0, f[side]["pos"][1] / 1000.0]
                     for f in frames])


def _missile_tracks(frames, key, match_km=4.0):
    """按最近邻跨帧关联导弹位置，返回若干条轨迹（km）。"""
    tracks, active = [], []
    for f in frames:
        cur = [np.array(m["pos"][:2]) / 1000.0 for m in f.get(key, [])]
        matched = [False] * len(active)
        nxt = []
        for p in cur:
            best, best_d = None, match_km
            for i, pts in enumerate(active):
                if matched[i]:
                    continue
                d = float(np.linalg.norm(p - pts[-1]))
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                nxt.append([p])
            else:
                matched[best] = True
                active[best].append(p)
                nxt.append(active[best])
        tracks.extend([pts for i, pts in enumerate(active)
                       if not matched[i] and len(pts) >= 2])
        active = nxt
    tracks.extend([pts for pts in active if len(pts) >= 2])
    return tracks


def _event_points(frames, side, ev_key):
    pts = []
    for f in frames:
        if f.get("events", {}).get(ev_key):
            pts.append((f[side]["pos"][0] / 1000.0,
                        f[side]["pos"][1] / 1000.0, f["t"]))
    return pts


def plot_episode(axes, rec: dict, out_label: str = "",
                 set_suptitle: bool = True):
    frames = rec["frames"]
    own_label, enemy_label = rec["own_tree"], rec["enemy_tree"]
    own_color = TREE_COLORS.get(own_label, "#1f77b4")
    enemy_color = TREE_COLORS.get(enemy_label, "#d62728")
    own = _pos_km(frames, "own")
    enemy = _pos_km(frames, "enemy")
    t = np.array([f["t"] for f in frames])
    dist = np.array([f["dist_m"] / 1000.0 for f in frames])
    alt_own = np.array([f["own"]["h_sl_m"] / 1000.0 for f in frames])
    alt_enemy = np.array([f["enemy"]["h_sl_m"] / 1000.0 for f in frames])
    v_own = np.array([f["own"]["vtrue_mps"] for f in frames])
    v_enemy = np.array([f["enemy"]["vtrue_mps"] for f in frames])

    # 1) 航迹
    ax = axes[0]
    ax.plot(own[:, 0], own[:, 1], color=own_color, lw=2.0,
            label=f"{TREE_NAMES.get(own_label, own_label)} (own)")
    ax.plot(enemy[:, 0], enemy[:, 1], color=enemy_color, lw=2.0,
            label=f"{TREE_NAMES.get(enemy_label, enemy_label)} (enemy)")
    for pts in _missile_tracks(frames, "own_missiles"):
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], color=own_color, lw=0.9, ls="--", alpha=0.65)
    for pts in _missile_tracks(frames, "enemy_missiles"):
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], color=enemy_color, lw=0.9, ls="--", alpha=0.65)
    ax.scatter([own[0, 0]], [own[0, 1]], marker="o", s=60, color=own_color,
               edgecolors="k", zorder=5)
    ax.scatter([enemy[0, 0]], [enemy[0, 1]], marker="o", s=60, color=enemy_color,
               edgecolors="k", zorder=5)
    ax.annotate("S", own[0], xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.annotate("S", enemy[0], xytext=(4, 4), textcoords="offset points", fontsize=9)
    for e_key, side, color in (("fired_own", "own", own_color),
                               ("fired_enemy", "enemy", enemy_color)):
        for x, y, tt in _event_points(frames, side, e_key):
            ax.scatter([x], [y], marker="*", s=180, color=color,
                       edgecolors="k", zorder=6)
    for e_key, side in (("hit_enemy", "enemy"), ("hit_own", "own")):
        for x, y, tt in _event_points(frames, side, e_key):
            ax.scatter([x], [y], marker="X", s=160, color="k", zorder=7)
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title("Trajectory (solid=aircraft, dashed=missile)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # 2) 距离 + 攻击区
    ax = axes[1]
    ax.plot(t, dist, color="k", lw=1.8)
    in_zone = np.array([bool(f.get("in_zone")) for f in frames])
    if in_zone.any():
        ax.fill_between(t, 0, dist, where=in_zone, color="tab:green",
                        alpha=0.18, label="own in launch zone")
    rz = frames[0].get("r_nez_own_m")
    rmax_o = frames[0].get("r_max_own_m")
    rmax_e = frames[0].get("r_max_enemy_m")
    if rmax_o:
        ax.axhline(rmax_o / 1000.0, color=own_color, ls="--", lw=1,
                   label="own Rmax")
    if rmax_e:
        ax.axhline(rmax_e / 1000.0, color=enemy_color, ls=":", lw=1,
                   label="enemy Rmax")
    if rz:
        ax.axhline(rz / 1000.0, color="tab:green", ls="-.", lw=1,
                   label="own Rnez")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance (km)")
    ax.set_title("Range & launch envelope")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    # 3) 高度
    ax = axes[2]
    ax.plot(t, alt_own, color=own_color, lw=1.8, label="own")
    ax.plot(t, alt_enemy, color=enemy_color, lw=1.8, label="enemy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("altitude (km)")
    ax.set_title("Altitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # 4) 速度
    ax = axes[3]
    ax.plot(t, v_own, color=own_color, lw=1.8, label="own")
    ax.plot(t, v_enemy, color=enemy_color, lw=1.8, label="enemy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("true airspeed (m/s)")
    ax.set_title("Speed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    own_name = TREE_NAMES.get(own_label, own_label)
    enemy_name = TREE_NAMES.get(enemy_label, enemy_label)
    reason = rec.get("terminated_reason") or "none"
    winner = rec.get("winner_tree", "draw")
    title = (f"{out_label} | own={own_name} vs enemy={enemy_name} | "
             f"winner={winner} | reason={reason} | steps={rec['steps']} | "
             f"min range={rec['min_dist_m'] / 1000:.1f} km")
    fig = axes[0].figure
    if set_suptitle:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    else:
        axes[0].set_title("Trajectory (solid=aircraft, dashed=missile)\n" + title,
                          fontsize=8.5)
    return title


def plot_composite(scene: str, recs: list[dict], out_path: Path):
    fig, axes = plt.subplots(len(recs), 4, figsize=(20, 4.6 * len(recs)))
    axes = np.atleast_2d(axes)
    for row, rec in enumerate(recs):
        label = (f"{scene} seed {rec['seed']} "
                 f"[{TREE_NAMES.get(rec['own_tree'], rec['own_tree'])} own]")
        plot_episode(axes[row], rec, out_label=label, set_suptitle=False)
    handles = [
        Line2D([0], [0], color=TREE_COLORS["handwritten"], lw=2,
               label="handwritten"),
        Line2D([0], [0], color=TREE_COLORS["bt_best"], lw=2,
               label="LLM g1c2 (bt_best)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=13, label="missile launch"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="k",
               markersize=11, label="hit"),
    ]
    fig.suptitle(f"{scene}: two longest BT-vs-BT episodes",
                 fontsize=14, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="挑选最长 BT-vs-BT 对局并可视化")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--own-mode", default="raw_fcs",
                    choices=("raw_fcs", "assisted"))
    ap.add_argument("--acmi-only", action="store_true",
                    help="只导出跨场景最长一局的 ACMI")
    args = ap.parse_args(argv)

    out_dir = (Path(args.out_dir) if args.out_dir
               else _HERE / "outputs" / "bt_vs_bt" / "visualization")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = ([select_longest_overall(RESULT_FILES)] if args.acmi_only
                else select_longest(RESULT_FILES))
    by_scene: dict[str, list[dict]] = {}
    index = []

    for item in selected:
        scene = item["scene"]
        stored = item["row"]
        cfg = env_config(scene, compute_envelope=True)
        max_steps = scene_max_steps(scene)
        own_tree, own_label = load_named_tree(stored["own_tree"])
        enemy_tree, enemy_label = load_named_tree(stored["enemy_tree"])

        print(f"[录制] {scene} seed {stored['seed']} "
              f"{own_label} vs {enemy_label} ...", flush=True)
        rec = run_episode(own_tree, enemy_tree, int(stored["seed"]), max_steps,
                          cfg, compute_envelope=True, own_mode=args.own_mode,
                          record_frames=True)
        rec["own_tree"] = own_label
        rec["enemy_tree"] = enemy_label
        rec["scene"] = scene
        rec["winner_tree"] = (
            "draw" if rec["winner_side"] == "draw"
            else (own_label if rec["winner_side"] == "own" else enemy_label))
        if (rec["steps"] != stored["steps"]
                or rec["terminated_reason"] != stored["terminated_reason"]):
            print(f"  [注意] 重放与归档结果不同：{rec['steps']}步/"
                  f"{rec['terminated_reason']} vs {stored['steps']}步/"
                  f"{stored['terminated_reason']}")

        tag = (f"{scene}_seed{stored['seed']}_{own_label}_vs_{enemy_label}")
        frames = rec["frames"]

        # ACMI（3D 回放）
        acmi_path = out_dir / f"{tag}.acmi"
        export_acmi(frames, acmi_path, lat0_deg=GEO_LAT0_DEG,
                    lon0_deg=GEO_LON0_DEG,
                    title=f"{scene} seed {stored['seed']} "
                          f"{own_label} vs {enemy_label}")

        csv_path = png_path = None
        if not args.acmi_only:
            # CSV（逐帧）
            recorder = EpisodeRecorder(meta={"lat0_deg": GEO_LAT0_DEG,
                                             "lon0_deg": GEO_LON0_DEG})
            recorder.frames = frames
            csv_path = out_dir / f"{tag}.csv"
            recorder.save_csv(csv_path)

            # 单局图
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            plot_episode(axes.reshape(-1), rec,
                         out_label=f"{scene} seed {stored['seed']}")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            png_path = out_dir / f"{tag}.png"
            fig.savefig(png_path, dpi=140)
            plt.close(fig)

            by_scene.setdefault(scene, []).append(rec)

        index.append({
            "scene": scene, "seed": int(stored["seed"]),
            "own_tree": own_label, "enemy_tree": enemy_label,
            "steps": rec["steps"], "terminated_reason": rec["terminated_reason"],
            "winner_tree": rec["winner_tree"], "min_dist_m": rec["min_dist_m"],
            "png": str(png_path) if png_path else None,
            "acmi": str(acmi_path),
            "csv": str(csv_path) if csv_path else None,
        })

    if not args.acmi_only:
        for scene, recs in by_scene.items():
            recs.sort(key=lambda r: r["seed"])
            png = out_dir / f"{scene}_longest2.png"
            plot_composite(scene, recs, png)
            index.append({"scene": scene, "composite_png": str(png)})

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[输出] {out_dir}")
    for item in index:
        if "composite_png" in item:
            print(f"  汇总图: {item['composite_png']}")
        elif args.acmi_only:
            print(f"  ACMI: {item['acmi']}")
        else:
            print(f"  {item['scene']} seed {item['seed']}: {item['png']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
