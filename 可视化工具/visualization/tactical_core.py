# -*- coding: utf-8 -*-
"""
可视化工具/visualization/tactical_core.py（绘图内核，后端无关）
================================================================
2D 战术显示的共用绘制逻辑：俯视态势图（双方位置/历史轨迹/导弹/攻击区
半透明扇形/告警）+ 态势面板。不 import pyplot，实时（TkAgg）与离屏
（FigureCanvasAgg）两条路径共用同一套绘制，保证所见一致。

输入帧格式：多任务智能体/common/bvr_combat_env.py 的
BVRCombatEnv.get_viz_frame() 帧字典（ENU + SI；本模块内部换算为 km 显示），
或 EpisodeRecorder.load_csv() 加载的同构帧。帧自包含绘制所需全部信息
（含攻击区扇形扫描角 radar_az_limit_deg），本模块不依赖智能体库。
坐标约定：x=东、y=北，航向 psi 0=正北、顺时针为正（JSBSim 约定）。
"""
from __future__ import annotations

import numpy as np
from matplotlib import rcParams, transforms
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, Wedge

# ---- 配色（与 validation 出图 tab 系一致）----
C_OWN = "#0057b8"        # 本机 蓝
C_ENEMY = "#d62728"      # 敌机 红
C_OWN_MSL = "#17becf"    # 本机导弹 青
C_ENEMY_MSL = "#ff7f0e"  # 敌方导弹 橙
C_ZONE = "#1f77b4"       # 本机攻击区
C_NEZ = "#ff7f0e"        # 不可逃逸区
C_ALARM = "#cc0000"

# 攻击区扇形半角默认值：仅当帧未携带 radar_az_limit_deg（旧录制文件）时使用；
# 新帧由 get_viz_frame() 携带该值，绘图不再 import 智能体库的雷达模型。
RADAR_AZ_LIMIT_DEG = 60.0

_TERMINATED_CN = {
    "enemy_hit": "命中敌机 —— 胜", "own_hit": "本机被命中",
    "own_crash": "本机坠毁", "enemy_crash": "敌机坠毁",
    "fuel_out": "燃油耗尽", "escape": "逃逸出界", "timeout": "超时截断",
}


def apply_chinese_font():
    """中文字体配置（与 validation/make_launch_envelope_charts.py 一致）。"""
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    rcParams["axes.unicode_minus"] = False


def _wrap180(deg):
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def ac_marker(psi_rad):
    """机头朝向 psi（0=北、顺时针为正）的三角标记（'^' 默认指北，顺时针旋转 psi）。"""
    return MarkerStyle("^", transform=transforms.Affine2D().rotate_deg(-np.rad2deg(psi_rad)))


class TrailHistory:
    """历史轨迹累积器：双方航迹 + 导弹尾迹点 + 命中标记（跨帧保持，供动画/实时显示）。"""

    def __init__(self, max_pts=20000):
        self.own, self.enemy = [], []
        self.own_msl, self.enemy_msl = [], []
        self.hit_marks = []          # [(e_m, n_m)] 命中位置（后续帧持续显示星标）
        self.max_pts = int(max_pts)

    def push(self, frm):
        self.own.append(frm["own"]["pos"][:2])
        self.enemy.append(frm["enemy"]["pos"][:2])
        for m in frm["own_missiles"]:
            self.own_msl.append(m["pos"][:2])
        for m in frm["enemy_missiles"]:
            self.enemy_msl.append(m["pos"][:2])
        ev = frm.get("events", {})
        if ev.get("hit_enemy"):
            self.hit_marks.append(tuple(frm["enemy"]["pos"][:2]))
        if ev.get("hit_own"):
            self.hit_marks.append(tuple(frm["own"]["pos"][:2]))
        for lst in (self.own, self.enemy, self.own_msl, self.enemy_msl):
            if len(lst) > self.max_pts:
                del lst[: len(lst) - self.max_pts]

    def clear(self):
        self.__init__(self.max_pts)


# ----------------------------------------------------------------------
# 俯视态势图
# ----------------------------------------------------------------------
def _envelope_wedge(ax, e_km, n_km, r_m, psi_rad, color, alpha, az_limit_deg, zorder=2):
    """以机头为中心 ±az_limit_deg 的攻击区/扫描扇形（半透明）。"""
    if r_m is None or r_m <= 0.0:
        return
    nose_deg = 90.0 - np.rad2deg(psi_rad)   # 航向（0=北顺时针）→ matplotlib 角（自+x逆时针）
    ax.add_patch(Wedge((e_km, n_km), r_m / 1e3,
                       nose_deg - az_limit_deg, nose_deg + az_limit_deg,
                       facecolor=color, alpha=alpha, edgecolor=color, lw=0.8, zorder=zorder))


def _trail(ax, pts, color):
    if len(pts) >= 2:
        arr = np.asarray(pts, dtype=float) / 1e3
        ax.plot(arr[:, 0], arr[:, 1], ls="--", lw=1.2, color=color, alpha=0.65, zorder=3)


def _dots(ax, pts, color):
    if pts:
        arr = np.asarray(pts, dtype=float) / 1e3
        ax.plot(arr[:, 0], arr[:, 1], ".", ms=2.5, color=color, alpha=0.55, zorder=3)


def draw_map(ax, frm, hist=None):
    """在 ax 上重绘一帧俯视态势（km 单位，北向上）。"""
    own, ene = frm["own"], frm["enemy"]
    oe, on = own["pos"][0] / 1e3, own["pos"][1] / 1e3
    ee, en = ene["pos"][0] / 1e3, ene["pos"][1] / 1e3
    psi_o, psi_e = own["psi_rad"], ene["psi_rad"]
    az_limit = float(frm.get("radar_az_limit_deg") or RADAR_AZ_LIMIT_DEG)
    ax.clear()

    # ---- 攻击区/不可逃逸区（半透明扇形，本机 R_max/R_nez/R_min + 敌方 R_max）----
    _envelope_wedge(ax, oe, on, frm.get("r_max_own"), psi_o, C_ZONE, 0.10, az_limit)
    _envelope_wedge(ax, oe, on, frm.get("r_nez_own"), psi_o, C_NEZ, 0.18, az_limit)
    _envelope_wedge(ax, ee, en, frm.get("r_max_enemy"), psi_e, C_ENEMY, 0.07, az_limit)
    if frm.get("r_min_own"):
        ax.add_patch(Circle((oe, on), frm["r_min_own"] / 1e3, fill=False, ls=":",
                            color=C_ZONE, lw=1.0, alpha=0.8, zorder=2))

    # ---- 历史轨迹与导弹尾迹 ----
    if hist is not None:
        _trail(ax, hist.own, C_OWN)
        _trail(ax, hist.enemy, C_ENEMY)
        _dots(ax, hist.own_msl, C_OWN_MSL)
        _dots(ax, hist.enemy_msl, C_ENEMY_MSL)
        for me, mn in hist.hit_marks:
            ax.scatter([me / 1e3], [mn / 1e3], marker="*", s=700, color="red",
                       edgecolors="black", linewidths=0.8, zorder=7)

    # ---- 弹目连线与距离标注 ----
    ax.plot([oe, ee], [on, en], ls="--", lw=0.9, color="gray", alpha=0.8, zorder=3)
    ax.annotate(f"{frm['dist_m'] / 1e3:.1f} km", ((oe + ee) / 2, (on + en) / 2),
                textcoords="offset points", xytext=(6, -12), fontsize=8, color="dimgray")

    # ---- 双方飞机（机头朝向三角 + 高度/速度标注）----
    ax.scatter([oe], [on], marker=ac_marker(psi_o), s=300, color=C_OWN,
               edgecolors="white", linewidths=1.0, zorder=5)
    ax.scatter([ee], [en], marker=ac_marker(psi_e), s=300, color=C_ENEMY,
               edgecolors="white", linewidths=1.0, zorder=5)
    ax.annotate(f"蓝 {own['h_sl_m'] / 1e3:.1f}km M{own['mach']:.2f}", (oe, on),
                textcoords="offset points", xytext=(12, 10), fontsize=8.5,
                color=C_OWN, fontweight="bold")
    ax.annotate(f"红 {ene['h_sl_m'] / 1e3:.1f}km M{ene['mach']:.2f}", (ee, en),
                textcoords="offset points", xytext=(12, -16), fontsize=8.5,
                color=C_ENEMY, fontweight="bold")

    # ---- 在飞导弹（圆点 + 速度向短线）----
    for m, c in ((frm["own_missiles"], C_OWN_MSL), (frm["enemy_missiles"], C_ENEMY_MSL)):
        for msl in m:
            me, mn = msl["pos"][0] / 1e3, msl["pos"][1] / 1e3
            ax.scatter([me], [mn], marker="o", s=55, color=c,
                       edgecolors="black", linewidths=0.6, zorder=6)

    # ---- 告警叠加 ----
    if frm.get("rwr_alarm"):
        # RWR 锁定告警：相对方位（相对本机机头）→ 绝对方位扇形
        bd = 90.0 - np.rad2deg(psi_o + frm.get("rwr_bearing", 0.0))
        ax.add_patch(Wedge((oe, on), 4.0, bd - 7.5, bd + 7.5,
                           facecolor=C_ALARM, alpha=0.55, zorder=4))
        ax.annotate("RWR 锁定!", (oe, on), textcoords="offset points",
                    xytext=(-58, 14), fontsize=9, color=C_ALARM, fontweight="bold")
    if frm.get("maws_alarm") and frm["enemy_missiles"]:
        # MAWS：最近来袭导弹 → 本机红色虚线 + TTA
        msl = min(frm["enemy_missiles"],
                  key=lambda m: (m["pos"][0] - own["pos"][0]) ** 2
                                + (m["pos"][1] - own["pos"][1]) ** 2)
        me, mn = msl["pos"][0] / 1e3, msl["pos"][1] / 1e3
        ax.plot([oe, me], [on, mn], ls="--", lw=1.6, color=C_ALARM, zorder=6)
        ax.annotate(f"MAWS TTA {frm.get('maws_tta', 0.0):.1f}s",
                    ((oe + me) / 2, (on + mn) / 2), textcoords="offset points",
                    xytext=(4, 6), fontsize=9, color=C_ALARM, fontweight="bold")

    # ---- 视野（覆盖双方 + 攻击区余量）----
    rmax_km = (frm.get("r_max_own") or 0.0) / 1e3
    half = max(abs(oe - ee), abs(on - en)) / 2.0 + max(rmax_km * 0.85, 15.0)
    ce, cn = (oe + ee) / 2.0, (on + en) / 2.0
    ax.set_xlim(ce - half, ce + half)
    ax.set_ylim(cn - half, cn + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("东向 (km)", fontsize=9)
    ax.set_ylabel("北向 (km)", fontsize=9)
    ax.set_title(f"BVR 空战态势   t = {frm['t']:.0f} s（第 {frm['steps']} 步）",
                 fontsize=11)

    # ---- 终止原因叠加 ----
    reason = frm.get("terminated_reason")
    if reason:
        ax.text(0.5, 0.94, _TERMINATED_CN.get(reason, reason), transform=ax.transAxes,
                ha="center", va="center", fontsize=15, fontweight="bold", color=C_ALARM,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_ALARM, alpha=0.9),
                zorder=8)


# ----------------------------------------------------------------------
# 态势面板
# ----------------------------------------------------------------------
def draw_panel(ax, frm, cum_reward=0.0):
    """右侧态势面板：高度/速度/航向/油量/武器/距离/告警/攻击区/事件。"""
    ax.clear()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    own, ene = frm["own"], frm["enemy"]
    rel_e = ene["pos"][0] - own["pos"][0]
    rel_n = ene["pos"][1] - own["pos"][1]
    brg = float(_wrap180(np.rad2deg(np.arctan2(rel_e, rel_n) - own["psi_rad"])))
    psi_deg = float(np.rad2deg(own["psi_rad"])) % 360.0
    fuel_pct = 100.0 * own.get("fuel_lbs", 0.0) / 4700.0
    ev = frm.get("events", {})

    lines = [
        (f"步 {frm['steps']:>3d}   本步奖励 {frm.get('reward', 0.0):+.2f}   累计 {cum_reward:+.1f}",
         "black", True),
        ("── 本机（蓝）──────────────", C_OWN, True),
        (f"高度 {own['h_sl_m']:7.0f} m    速度 {own['vtrue_mps']:5.1f} m/s  M {own['mach']:.2f}", "black", False),
        (f"航向 {psi_deg:5.1f}°     过载 {own.get('nz_g', 0.0):+.1f} g    油门 {own.get('throttle', 0.0):.2f}", "black", False),
        (f"油量 {own.get('fuel_lbs', 0.0):5.0f} lb ({fuel_pct:2.0f}%)   导弹 {own['n_left']}/2   "
         f"ECM {'开' if own.get('ecm_on') else '关'}", "black", False),
        (f"雷达 {frm.get('radar_state') or '—'}   攻击区内 {'√' if frm.get('in_zone') else '×'}",
         C_OWN if frm.get("in_zone") else "black", frm.get("in_zone", False)),
        ("── 敌机（红）──────────────", C_ENEMY, True),
        (f"高度 {ene['h_sl_m']:7.0f} m    速度 {ene['vtrue_mps']:5.1f} m/s  M {ene['mach']:.2f}", "black", False),
        (f"距离 {frm['dist_m'] / 1e3:5.1f} km   相对方位 {brg:+.0f}°   敌导弹 {ene['n_left']}/2", "black", False),
        (f"敌雷达 {frm.get('enemy_radar_state') or '—'}", "black", False),
        ("── 告警 ──────────────────", "black", True),
        (("RWR 锁定告警！方位 " + f"{np.rad2deg(frm.get('rwr_bearing', 0.0)):+.0f}°")
         if frm.get("rwr_alarm") else "RWR 无告警",
         C_ALARM if frm.get("rwr_alarm") else "gray", frm.get("rwr_alarm", False)),
        ((f"MAWS 导弹来袭！TTA {frm.get('maws_tta', 0.0):.1f} s")
         if frm.get("maws_alarm") else "MAWS 无告警",
         C_ALARM if frm.get("maws_alarm") else "gray", frm.get("maws_alarm", False)),
        ("── 攻击区（本机）──────────", "black", True),
        (f"R_max {_km(frm.get('r_max_own'))} km   R_nez {_km(frm.get('r_nez_own'))} km   "
         f"R_min {_km(frm.get('r_min_own'))} km", "black", False),
        (f"R_max（敌） {_km(frm.get('r_max_enemy'))} km", "black", False),
        ("── 事件 ──────────────────", "black", True),
        (_events_text(ev), C_ALARM if any(ev.values()) else "gray", any(ev.values())),
        (f"终止: {_TERMINATED_CN.get(frm['terminated_reason'], frm['terminated_reason'])}"
         if frm.get("terminated_reason") else "终止: —",
         C_ALARM if frm.get("terminated_reason") else "gray", bool(frm.get("terminated_reason"))),
    ]
    dy = 0.96 / (len(lines) + 1)
    y = 0.985
    for text, color, bold in lines:
        ax.text(0.02, y, text, fontsize=9.5, color=color, va="top",
                fontweight="bold" if bold else "normal")
        y -= dy


def _km(r_m):
    return f"{r_m / 1e3:.1f}" if r_m is not None else "—"


def _events_text(ev):
    parts = []
    if ev.get("fired_own"):
        parts.append("本机发射!")
    if ev.get("fired_enemy"):
        parts.append("敌机发射!")
    if ev.get("hit_enemy"):
        parts.append("命中敌机!")
    if ev.get("hit_own"):
        parts.append("本机被命中!")
    return "  ".join(parts) if parts else "—"


def draw_tactical_frame(ax_map, ax_panel, frm, hist=None, cum_reward=0.0):
    """绘制完整一帧（态势图 + 面板）。"""
    draw_map(ax_map, frm, hist)
    draw_panel(ax_panel, frm, cum_reward)
