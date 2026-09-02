# -*- coding: utf-8 -*-
"""
validation/make_launch_envelope_charts.py（交付物 D2.2-3 出图脚本）
====================================================================
生成攻击区（LAE）/不可逃逸区（NEZ）曲线图到 validation/results/：
  1. D2.2-3_launch_envelope_altitude.png  不同发射高度下 R_max vs 相对方位角
  2. D2.2-3_launch_envelope_speed.png     不同发射速度下 R_max vs 相对方位角
  3. D2.2-3_launch_envelope_rmin_nez.png  R_max / R_min / R_nez 对比（NEZ ⊂ LAE）
  4. D2.2-3_launch_envelope_polar.png     极坐标攻击区（附加图）

场景约定：发射机位于原点沿 +X 飞行，目标始终朝向发射机飞行（迎头进入）；
横轴为目标相对发射机的方位角（0°=正前方，±90°=正侧方，±180°=正后方）。
解算结果缓存到 launch_envelope_cache.npz（再次运行只重算缺失工况）。
用法（DC 环境）：python validation/make_launch_envelope_charts.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.missile_model import launch_envelope

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent / "results"
CACHE_FILE = RESULTS / "launch_envelope_cache.npz"
BEARINGS = np.arange(-180.0, 181.0, 15.0)

_cache = {}


def _load_cache():
    if not CACHE_FILE.exists():
        return
    z = np.load(CACHE_FILE, allow_pickle=True)
    for name in z.files:
        kind, h, va = name.rsplit("_", 2)
        arr = z[name]
        for b, val in zip(BEARINGS, arr):
            if np.isnan(val):
                continue
            _cache.setdefault((float(h), float(va), float(b)), {})[kind] = float(val)


def _save_cache():
    data = {}
    for (h, va, b), vals in _cache.items():
        for kind in ("r_max", "r_min", "r_nez"):
            data.setdefault(f"{kind}_{h:.0f}_{va:.0f}", {})[int(b)] = vals.get(kind, np.nan)
    out = {}
    for name, d in data.items():
        arr = np.full(BEARINGS.size, np.nan)
        for b, v in d.items():
            arr[int(np.where(BEARINGS == b)[0][0])] = v
        out[name] = arr
    np.savez(CACHE_FILE, **out)


def env_at(h, va, bearing):
    """带缓存（内存 + npz 持久化）的攻击区解算。"""
    key = (round(h), round(va), round(bearing))
    if key in _cache and _cache[key].get("r_max") is not None:
        return _cache[key]
    env = launch_envelope(launcher_h=h, launcher_speed=va,
                          target_h=h, target_speed=va, bearing_deg=bearing)
    _cache[key] = {"r_max": env.r_max, "r_min": env.r_min, "r_nez": env.r_nez}
    print(f"  [解算] h={h:.0f} m  v={va:.0f} m/s  bearing={bearing:+4.0f}°  "
          f"R_max={env.r_max/1e3 if env.r_max else 0:5.1f} km  "
          f"R_nez={env.r_nez/1e3 if env.r_nez else 0:5.1f} km  "
          f"R_min={env.r_min/1e3 if env.r_min else 0:4.1f} km", flush=True)
    return _cache[key]


def sweep(h, va):
    """扫描全部方位角，返回 (r_max, r_min, r_nez) km 数组（解算失败处为 nan）。"""
    rm, rmin, rnz = [], [], []
    for b in BEARINGS:
        env = env_at(h, va, b)
        rm.append(env["r_max"] / 1e3 if env["r_max"] else np.nan)
        rmin.append(env["r_min"] / 1e3 if env["r_min"] else np.nan)
        rnz.append(env["r_nez"] / 1e3 if env["r_nez"] else np.nan)
    return np.array(rm), np.array(rmin), np.array(rnz)


def smooth(y, half=1):
    """3 点加权平滑（[1,2,1]/4），仅用于 NEZ 曲线呈现（端部重复边缘值）。"""
    w = np.array([1.0, 2.0, 1.0]) / 4.0
    y2 = np.pad(y, (half, half), mode="edge")
    return np.convolve(y2, w, mode="valid")


def style_ax(ax):
    ax.set_xlabel("目标相对方位角 (deg)")
    ax.set_ylabel("距离 (km)")
    ax.grid(True, alpha=0.35)
    ax.set_xlim(-185, 185)
    ax.legend(loc="upper right", fontsize=9)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    _load_cache()
    t0 = time.time()

    # ---- 图 1：不同发射高度 ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    alts = [(3048.0, "10000 ft (3048 m)"), (4572.0, "15000 ft (4572 m)"), (6096.0, "20000 ft (6096 m)")]
    for h, label in alts:
        print(f"== 图1 高度 {label} ==", flush=True)
        rm, _, _ = sweep(h, 280.0)
        ax.plot(BEARINGS, rm, "o-", label=label)
    ax.set_title("攻击区 R_max vs 相对方位角（发射机/目标速度 280 m/s，不同发射高度）")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "D2.2-3_launch_envelope_altitude.png", dpi=150)
    plt.close(fig)
    print(f"[图1 完成] {time.time()-t0:.0f} s", flush=True)

    # ---- 图 2：不同发射速度 ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    speeds = [(240.0, "240 m/s (M0.8)"), (280.0, "280 m/s (M0.9)"), (340.0, "340 m/s (M1.1)")]
    for va, label in speeds:
        print(f"== 图2 速度 {label} ==", flush=True)
        rm, _, _ = sweep(4572.0, va)
        ax.plot(BEARINGS, rm, "s-", label=label)
    ax.set_title("攻击区 R_max vs 相对方位角（高度 15000 ft，不同发射速度）")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "D2.2-3_launch_envelope_speed.png", dpi=150)
    plt.close(fig)
    print(f"[图2 完成] {time.time()-t0:.0f} s", flush=True)

    # ---- 图 3：R_max / R_min / R_nez（NEZ ⊂ LAE）----
    rm, rmin, rnz = sweep(4572.0, 280.0)   # 已缓存/已算
    rnz_s = smooth(rnz)                     # NEZ 曲线 3 点平滑（二分边界存在 ±2 km 抖动）
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(BEARINGS, rm, "o-", label="R_max（直飞目标，LAE 远界）")
    ax.plot(BEARINGS, rnz_s, "s-", label="R_nez（9g 规避目标，不可逃逸区，3 点平滑）")
    ax.plot(BEARINGS, rmin, "^-", label="R_min（引信保险+运动学，LAE 近界）")
    ax.fill_between(BEARINGS, rmin, rm, alpha=0.10, color="tab:blue", label="攻击区 LAE")
    ax.fill_between(BEARINGS, rmin, np.minimum(rnz_s, rm), alpha=0.15, color="tab:orange",
                    label="不可逃逸区 NEZ")
    ax.set_title("攻击区/不可逃逸区（高度 15000 ft，发射机/目标 280 m/s）")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "D2.2-3_launch_envelope_rmin_nez.png", dpi=150)
    plt.close(fig)
    print(f"[图3 完成] {time.time()-t0:.0f} s", flush=True)

    # ---- 图 4（附加）：极坐标攻击区 ----
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    theta = np.deg2rad(BEARINGS)
    ax.plot(theta, rm, "o-", label="R_max")
    ax.plot(theta, rnz_s, "s-", label="R_nez（平滑）")
    ax.plot(theta, rmin, "^-", label="R_min")
    ax.set_theta_zero_location("N")     # 0° 朝上（机头方向）
    ax.set_theta_direction(-1)          # 顺时针
    ax.set_title("攻击区极坐标视图（15000 ft / 280 m/s）", pad=20)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "D2.2-3_launch_envelope_polar.png", dpi=150)
    plt.close(fig)

    _save_cache()
    print(f"===== 全部出图完成，总耗时 {time.time()-t0:.0f} s =====")


if __name__ == "__main__":
    main()
