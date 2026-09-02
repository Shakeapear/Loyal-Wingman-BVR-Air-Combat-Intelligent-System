# -*- coding: utf-8 -*-
"""
common/missile_model.py（交付物 D2.2-1）
========================================
中距空空导弹 3DOF 质点动力学模型 + 攻击区（LAE）/不可逃逸区（NEZ）快速解算 + 命中判定

模型依据（详见 common/README_战场要素.md 的参数来源表）：
- 运动学/动力学：参照 CloseAirCombat 开源项目 `docs/missile_engine.md` 的
  导弹质点动力学方程组（推力 T = g·Isp·ṁ、阻力 D = ½·cD·S·ρ·v²、比例导引）；
- 导引律：三维真比例导引（TPN）矢量形式 a_cmd = N·(Ω_LOS × V_m)，
  Ω_LOS = (e × V_rel) / r，比例系数 N 默认 4（文献典型 3~4，
  CloseAirCombat/BVRGym 取 K=2~3）；末端 3 s 切换增广比例导引（APN），
  叠加 (N/2)·目标加速度在视线法向分量；
- 发动机：双推力固体火箭（助推段 25 kN/3 s + 续航段 9 kN/8 s），
  依据 AIM-120C / PL-12(SD-10) 公开性能描述（助推 2~4 s + 续航 6~10 s）；
- 攻击区解算：以本模型的 3DOF 仿真为核心，对发射距离做批量探测 +
  区间二分，直飞目标给出 R_max / R_min，按最大过载规避（默认 9g）目标给出 R_nez；
- 命中判定：全程最小距离（脱靶量）< 杀伤半径 10 m。

所有量均为 SI 单位（m、m/s、kg、N、s），角度在 API 边界用 deg 便于使用，
内部统一 rad。

典型用法：
    from common.missile_model import Missile, launch_envelope

    env = launch_envelope(launcher_h=4572.0, launcher_speed=280.0,
                          target_h=4572.0, target_speed=280.0, bearing_deg=0.0)
    print(env.r_max, env.r_min, env.r_nez)

    m = Missile(pos=np.array([0.0, 0.0, 6000.0]), vel=np.array([300.0, 0.0, 0.0]))
    for _ in range(1000):
        m.step(0.05, target_pos, target_vel, target_acc)
        if not m.alive:
            break
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# 常量与模型参数（来源见 README_战场要素.md §参数来源表）
# ----------------------------------------------------------------------------
G0 = 9.80665  # 重力加速度 m/s²

# 国际标准大气（ISA，用于空气密度，公式见 README）
RHO0, T0, L, G_ISA, R_AIR = 1.225, 288.15, -0.0065, 9.80665, 287.05
H_TROP = 11000.0  # 对流层顶 m
RHO_TROP = 0.3639175  # 11 km 处密度 kg/m³（由 ISA 计算）

# AIM-120C / PL-12 量级的中距弹参数（常量字典，供 Missile 与 launch_envelope 共用）
MISSILE_PARAMS: dict = {
    # ---- 弹体/气动 ----
    "mass0": 160.0,         # 发射质量 kg（AIM-120A/B ≈157 kg、C-5 ≈161.5 kg 公开值）
    "diameter": 0.178,      # 弹径 m（AIM-120 公开 7 英寸）
    "s_ref": None,          # 计算面积 m²，默认按 π(d/2)² 计算（与速度正交的截面积）
    "cd": 0.6,              # 阻力系数（细长体导弹跨/超声速典型 0.4~0.8，常数近似）
    # ---- 双推力固体火箭发动机 ----
    "thr_boost": 25000.0,   # 助推段推力 N（公开无精确值，按总冲匹配估算）
    "t_boost": 3.0,         # 助推段时长 s（公开描述 2~4 s）
    "thr_sustain": 9000.0,  # 续航段推力 N（估算）
    "t_sustain": 11.0,      # 续航段结束时刻 s（公开描述助推+续航共约 8~14 s）
    "isp": 250.0,           # 比冲 s（固体推进剂典型 240~260 s）
    # ---- 制导 ----
    "n_gain": 4.0,          # 比例导引系数 N（文献典型 3~4）
    "t_terminal": 3.0,      # 末端制导切换剩余飞行时间 s（t_go < 3 s 启用 APN）
    "pp_gain": 3.0,         # 纯追踪段转向增益（将速度矢量拉向视线，中制导近似，仅后半球启用）
    # ---- 机动/杀伤 ----
    "n_max_g": 30.0,        # 侧向可用过载上限 g（AIM-120C/PL-12 公开 30~40G，取保守 30）
    "kill_radius": 10.0,    # 杀伤半径 m（高爆破片战斗部+近炸引信仿真惯例）
    "arm_dist": 1000.0,     # 最小发射距离下限（引信解除保险距离，公开约数百米~1 km）
    # ---- 仿真终止 ----
    "v_min": 200.0,         # 速度低于该值判定导弹失能（无操纵能力）m/s
}


def air_density(h: np.ndarray | float) -> np.ndarray | float:
    """ISA 空气密度 kg/m³（h 为海拔 m，支持标量或 ndarray）。

    h < 11 km:  ρ = ρ0·(T/T0)^(g/(R·L) - 1)，T = T0 + L·h
    h ≥ 11 km:  等温层指数衰减 ρ = ρ(11km)·exp(-(h-11000)·g/(R·T11))
    """
    h = np.clip(np.asarray(h, dtype=float), -2000.0, 40000.0)
    rho = np.where(
        h < H_TROP,
        RHO0 * ((T0 + L * h) / T0) ** (G_ISA / (R_AIR * (-L)) - 1.0),
        RHO_TROP * np.exp(-(h - H_TROP) * G_ISA / (R_AIR * (T0 + L * H_TROP))),
    )
    return rho.item() if rho.ndim == 0 else rho


def sound_speed(h: np.ndarray | float) -> np.ndarray | float:
    """ISA 声速 m/s（用于马赫数参考输出）。"""
    h = np.asarray(h, dtype=float)
    t_k = np.where(h < H_TROP, T0 + L * h, T0 + L * H_TROP)
    a = np.sqrt(1.4 * R_AIR * t_k)
    return a.item() if a.ndim == 0 else a


# ----------------------------------------------------------------------------
# 仿真结果
# ----------------------------------------------------------------------------
@dataclass
class EngagementResult:
    """单次交战（发射→命中/脱靶）仿真结果。"""
    hit: bool = False                  # 是否命中（最小距离 ≤ 杀伤半径）
    miss_distance: float = np.inf      # 脱靶量（全程最小弹目距离）m
    t_impact: float = np.inf           # 命中/终止时刻 s
    max_lat_g: float = 0.0             # 侧向过载峰值（含限幅后实际值）g
    max_lat_cmd_g: float = 0.0         # 侧向过载指令峰值（限幅前）g
    max_speed_mps: float = 0.0         # 最大速度 m/s
    terminal_reason: str = ""          # 终止原因: hit / energy / ground / timeout
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    a_lat_g: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ----------------------------------------------------------------------------
# Missile 类：3DOF 质点动力学 + 比例导引
# ----------------------------------------------------------------------------
class Missile:
    """中距空空导弹 3DOF 质点模型（可被 BVRCombatEnv 逐帧调用，也可整体仿真）。

    - 位置/速度在东北天（ENU）直角坐标系，z 向上为正，单位 m、m/s；
    - 侧向过载限幅 n_max_g（默认 30g），轴向为发动机推力-阻力；
    - 制导：三维 TPN（N=4）→ 末端 t_go<3s 切 APN（叠加目标加速度法向分量）；
    - 命中判定：弹目距离 ≤ kill_radius（默认 10 m）即命中，miss_distance 记录
      全程最小弹目距离（脱靶量）。
    """

    def __init__(self, pos, vel, params: dict | None = None):
        p = dict(MISSILE_PARAMS)
        if params:
            p.update(params)
        self.p = p
        if p.get("s_ref") is None:
            p["s_ref"] = np.pi * (p["diameter"] / 2.0) ** 2
        self.pos = np.asarray(pos, dtype=float).reshape(3).copy()
        self.vel = np.asarray(vel, dtype=float).reshape(3).copy()
        self.t = 0.0
        self.alive = True
        self.hit = False
        self.miss_distance = np.inf
        self.max_lat_g = 0.0
        self.max_lat_cmd_g = 0.0
        self.max_speed_mps = float(np.linalg.norm(self.vel))
        self.terminal_reason = ""
        # 燃料质量流量（两段常值），由 T = g·Isp·ṁ 反推
        self.mdot_boost = p["thr_boost"] / (p["isp"] * G0)
        self.mdot_sustain = p["thr_sustain"] / (p["isp"] * G0)

    # ---- 发动机/质量 ----
    def thrust(self, t: float | None = None) -> float:
        """t 时刻推力 N（双推力曲线：助推 [0,t_boost) + 续航 [t_boost,t_sustain)）。"""
        if t is None:
            t = self.t
        p = self.p
        if t < p["t_boost"]:
            return p["thr_boost"]
        if t < p["t_sustain"]:
            return p["thr_sustain"]
        return 0.0

    def mass(self, t: float | None = None) -> float:
        """t 时刻质量 kg（m(t) = m0 - ∫ṁ dt，两段常值质量流量）。"""
        if t is None:
            t = self.t
        p = self.p
        tb, ts = p["t_boost"], p["t_sustain"]
        return p["mass0"] - self.mdot_boost * min(t, tb) - self.mdot_sustain * max(0.0, min(t, ts) - tb)

    # ---- 制导 ----
    def guidance(self, target_pos, target_vel, target_acc=None):
        """计算侧向过载指令 m/s²（已按 n_max_g 限幅）。

        返回 (a_lat, a_lat_cmd)：a_lat 为限幅后实际施加值，a_lat_cmd 为限幅前指令。
        """
        p = self.p
        rv = np.asarray(target_pos, dtype=float).reshape(3) - self.pos
        rn = float(np.linalg.norm(rv))
        v_rel = np.asarray(target_vel, dtype=float).reshape(3) - self.vel
        v_close = -float(np.dot(v_rel, rv)) / rn if rn > 1e-9 else 0.0
        if rn < 1e-9:
            return np.zeros(3), np.zeros(3)
        e = rv / rn
        omega = np.cross(e, v_rel) / rn            # 视线角速度矢量 Ω_LOS = e × V_rel / r
        a_pn = p["n_gain"] * np.cross(omega, self.vel)  # TPN: a = N·Ω × V_m
        # 中制导段：速度矢量与视线夹角较大时用纯追踪先将速度拉向目标
        # （避免正后方等退化几何下 TPN 无指令；真实导弹为 INS/数据链中制导）
        u = self.vel / max(float(np.linalg.norm(self.vel)), 1e-6)
        cos_ue = float(np.dot(u, e))
        if cos_ue < 0.0:  # 目标位于后半球（视线与速度夹角 > 90°）时先用纯追踪转向
            e_perp = e - cos_ue * u              # 视线在速度法向的分量（转向方向）
            if np.linalg.norm(e_perp) < 1e-6:
                e_perp = np.cross(np.array([0.0, 0.0, 1.0]), u)  # 正后方退化：默认右转
            a_pn = p["pp_gain"] * float(np.linalg.norm(self.vel)) * e_perp
        # 重力补偿：指令侧向过载含 +1g 竖直分量
        # （对应 CloseAirCombat 的 n_z = (Kv/g)·ε̇ + cosθ 中 cosθ 项，避免长距离稳态下沉）
        a_pn = a_pn + np.array([0.0, 0.0, G0])
        # 末端制导：t_go < t_terminal 时切换 APN（叠加目标加速度视线法向分量）
        a_cmd = a_pn
        t_go = rn / max(v_close, 1.0) if v_close > 1.0 else np.inf
        if t_go < p["t_terminal"] and target_acc is not None:
            a_t = np.asarray(target_acc, dtype=float).reshape(3)
            a_cmd = a_pn + 0.5 * p["n_gain"] * (a_t - np.dot(a_t, e) * e)
        a_lim = p["n_max_g"] * G0
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm > a_lim and a_norm > 1e-12:
            return a_cmd * (a_lim / a_norm), a_cmd
        return a_cmd.copy(), a_cmd.copy()

    # ---- 单步积分（BVRCombatEnv 逐帧调用入口）----
    def step(self, dt: float, target_pos, target_vel, target_acc=None):
        """推进 dt 秒。调用后检查 self.hit / self.alive / self.miss_distance。"""
        if not self.alive:
            return
        p = self.p
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        target_vel = np.asarray(target_vel, dtype=float).reshape(3)
        rn = float(np.linalg.norm(target_pos - self.pos))
        # 连续碰撞检测：相对位移线段到目标的最小距离（防止大 dt 跨步跳过杀伤半径）
        p_rel = self.pos - target_pos
        v_rel = self.vel - target_vel
        v2 = float(np.dot(v_rel, v_rel))
        if v2 > 1e-12:
            t_ca = float(np.clip(-np.dot(p_rel, v_rel) / v2, 0.0, dt))
            d_min = float(np.linalg.norm(p_rel + v_rel * t_ca))
        else:
            d_min = rn
        self.miss_distance = min(self.miss_distance, d_min)
        if d_min <= p["kill_radius"]:
            self.hit, self.alive = True, False
            self.terminal_reason = "hit"
            return
        speed = float(np.linalg.norm(self.vel))
        self.max_speed_mps = max(self.max_speed_mps, speed)
        if speed < p["v_min"]:
            self.alive = False
            self.terminal_reason = "energy"
            return
        if self.pos[2] <= 0.0:
            self.alive = False
            self.terminal_reason = "ground"
            return

        a_lat, a_cmd = self.guidance(target_pos, target_vel, target_acc)
        self.max_lat_g = max(self.max_lat_g, float(np.linalg.norm(a_lat)) / G0)
        self.max_lat_cmd_g = max(self.max_lat_cmd_g, float(np.linalg.norm(a_cmd)) / G0)

        thr = self.thrust(self.t)
        m = self.mass(self.t)
        rho = float(air_density(self.pos[2]))
        drag = 0.5 * rho * p["cd"] * p["s_ref"] * speed ** 2
        a_ax = (thr - drag) / m
        u = self.vel / max(speed, 1e-6)
        acc = a_ax * u + a_lat + np.array([0.0, 0.0, -G0])
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt
        self.t += dt

    # ---- 完整交战仿真（测试/分析用，记录轨迹）----
    def simulate(self, target_motion, dt: float = 0.05, t_max: float = 120.0) -> EngagementResult:
        """对给定目标运动规律整体仿真。

        target_motion(t) -> (target_pos(3), target_vel(3), target_acc(3) 或 None)。
        """
        ts, poss, vels, a_g = [], [], [], []
        n_steps = int(t_max / dt) + 1
        for i in range(n_steps):
            t = i * dt
            tp, tv, ta = target_motion(t)
            ts.append(t)
            poss.append(self.pos.copy())
            vels.append(self.vel.copy())
            a_g.append(self.max_lat_g)
            self.step(dt, tp, tv, ta)
            if not self.alive:
                break
        return EngagementResult(
            hit=self.hit,
            miss_distance=self.miss_distance,
            t_impact=self.t,
            max_lat_g=self.max_lat_g,
            max_lat_cmd_g=self.max_lat_cmd_g,
            max_speed_mps=self.max_speed_mps,
            terminal_reason=self.terminal_reason,
            t=np.asarray(ts),
            pos=np.asarray(poss),
            vel=np.asarray(vels),
            a_lat_g=np.asarray(a_g),
        )


# ----------------------------------------------------------------------------
# 批量交战仿真核心（launch_envelope 用，向量化加速）
# ----------------------------------------------------------------------------
def _simulate_batch(pos_m0, vel_m0, pos_t0, vel_t0,
                    mode="straight", turn_dir=1.0, evasive_g=9.0, evade_delay=1.0,
                    dt=0.2, t_max=100.0, params=None):
    """批量仿真 K 条交战轨迹（相同目标机动规律、不同初始距离）。

    返回 (hit[K]bool, miss_dist[K], max_lat_g[K], terminal_reason[K])。
    mode: 'straight' 目标直飞；'evade' 目标按 evasive_g 做水平定速盘旋规避。
    """
    p = dict(MISSILE_PARAMS)
    if params:
        p.update(params)
    if p.get("s_ref") is None:
        p["s_ref"] = np.pi * (p["diameter"] / 2.0) ** 2

    K = pos_m0.shape[0]
    pos_m = pos_m0.copy()
    vel_m = vel_m0.copy()
    pos_t = pos_t0.copy()
    vel_t = vel_t0.copy()
    speed_t = np.linalg.norm(vel_t0, axis=1)
    psi_t = np.arctan2(vel_t0[:, 1], vel_t0[:, 0])  # 目标水平航向
    omega_t = evasive_g * G0 / np.maximum(speed_t, 1e-6)

    alive = np.ones(K, dtype=bool)
    hit = np.zeros(K, dtype=bool)
    min_r = np.full(K, np.inf)
    t_hit = np.full(K, np.inf)
    max_lat = np.zeros(K)
    reason = np.array(["timeout"] * K, dtype=object)
    md_boost = p["thr_boost"] / (p["isp"] * G0)
    md_sustain = p["thr_sustain"] / (p["isp"] * G0)
    tb, ts = p["t_boost"], p["t_sustain"]
    g_vec = np.array([0.0, 0.0, -G0])

    n_steps = int(t_max / dt) + 1
    for i in range(n_steps):
        t = i * dt
        if not alive.any():
            break
        # ---- 几何/命中判定（连续碰撞检测：相对位移线段到原点最小距离）----
        rv = pos_t - pos_m
        rn = np.linalg.norm(rv, axis=1)
        np.maximum(rn, 1e-9, out=rn)
        p_rel = pos_m - pos_t
        v_rel_c = vel_m - vel_t
        v2 = np.einsum("ij,ij->i", v_rel_c, v_rel_c)
        approaching = np.einsum("ij,ij->i", p_rel, v_rel_c) < 0.0
        t_ca = np.where(approaching & (v2 > 1e-12),
                        np.clip(-np.einsum("ij,ij->i", p_rel, v_rel_c) / np.maximum(v2, 1e-12), 0.0, dt),
                        0.0)
        d_min = np.linalg.norm(p_rel + v_rel_c * t_ca[:, None], axis=1)
        min_r = np.minimum(min_r, d_min)
        new_hit = alive & (d_min <= p["kill_radius"])
        hit |= new_hit
        t_hit = np.where(new_hit, t, t_hit)
        reason = np.where(new_hit, "hit", reason)
        alive &= ~new_hit
        # ---- 失能判定 ----
        speed_m = np.linalg.norm(vel_m, axis=1)
        dead = alive & ((speed_m < p["v_min"]) | (pos_m[:, 2] <= 0.0))
        reason = np.where(dead & (speed_m < p["v_min"]), "energy", reason)
        reason = np.where(dead & (pos_m[:, 2] <= 0.0), "ground", reason)
        alive &= ~dead
        if not alive.any():
            break
        # ---- 目标运动（机动规避）----
        a_t = np.zeros_like(pos_t)
        if mode == "evade" and t >= evade_delay:
            psi_t = psi_t + turn_dir * omega_t * dt
            vel_t[:, 0] = speed_t * np.cos(psi_t)
            vel_t[:, 1] = speed_t * np.sin(psi_t)
            a_t[:, 0] = -speed_t * omega_t * turn_dir * np.sin(psi_t)
            a_t[:, 1] = speed_t * omega_t * turn_dir * np.cos(psi_t)
        pos_t = pos_t + vel_t * dt
        # ---- 制导 ----
        e = rv / rn[:, None]
        v_rel = vel_t - vel_m
        omega = np.cross(e, v_rel) / rn[:, None]          # Ω_LOS
        a_pn = p["n_gain"] * np.cross(omega, vel_m)      # TPN
        # 中制导段纯追踪交接（速度矢量拉向视线，处理正后方退化几何）
        u = vel_m / np.maximum(np.linalg.norm(vel_m, axis=1), 1e-6)[:, None]
        cos_ue = np.einsum("ij,ij->i", u, e)
        use_pp = cos_ue < 0.0  # 仅后半球启用纯追踪（正侧方/前半球一律 TPN）
        e_perp = e - cos_ue[:, None] * u          # 视线在速度法向的分量（转向方向）
        degn = np.linalg.norm(e_perp, axis=1) < 1e-6
        e_perp = np.where(degn[:, None], np.cross(np.array([0.0, 0.0, 1.0]), u), e_perp)
        a_pp = p["pp_gain"] * np.linalg.norm(vel_m, axis=1)[:, None] * e_perp
        a_pn = np.where(use_pp[:, None], a_pp, a_pn)
        # 重力补偿（对应 CloseAirCombat n_z 中的 cosθ 项）
        a_pn = a_pn + np.array([0.0, 0.0, G0])
        v_close = -np.einsum("ij,ij->i", v_rel, e)
        t_go = np.where(v_close > 1.0, rn / np.maximum(v_close, 1e-3), np.inf)
        if mode == "evade":
            a_t_perp = a_t - np.einsum("ij,ij->i", a_t, e)[:, None] * e
            use_apn = t_go < p["t_terminal"]
            a_cmd = a_pn + use_apn[:, None] * (0.5 * p["n_gain"] * a_t_perp)
        else:
            a_cmd = a_pn
        a_lim = p["n_max_g"] * G0
        a_norm = np.linalg.norm(a_cmd, axis=1)
        scale = np.minimum(1.0, a_lim / np.maximum(a_norm, 1e-12))
        a_lat = a_cmd * scale[:, None]
        max_lat = np.maximum(max_lat, np.linalg.norm(a_lat, axis=1) / G0)
        # ---- 轴向/积分 ----
        m_t = p["mass0"] - md_boost * np.minimum(t, tb) - md_sustain * np.maximum(0.0, np.minimum(t, ts) - tb)
        thr = np.where(t < tb, p["thr_boost"], np.where(t < ts, p["thr_sustain"], 0.0))
        rho = air_density(pos_m[:, 2])
        drag = 0.5 * rho * p["cd"] * p["s_ref"] * speed_m ** 2
        a_ax = (thr - drag) / m_t
        u = vel_m / np.maximum(speed_m, 1e-6)[:, None]
        new_vel = vel_m + (a_ax[:, None] * u + a_lat + g_vec) * dt
        new_pos = pos_m + new_vel * dt
        vel_m = np.where(alive[:, None], new_vel, vel_m)
        pos_m = np.where(alive[:, None], new_pos, pos_m)
    return hit, min_r, t_hit, max_lat, reason


def _probe_hits(ranges, launcher_pos, launcher_vel, target_h, target_speed,
                target_heading_rad, bearing_rad, mode, turn_dir,
                evasive_g, evade_delay, dt, t_max, params):
    """对一组发射距离 R 批量仿真，返回命中布尔数组。"""
    ranges = np.asarray(ranges, dtype=float)
    K = ranges.size
    cb, sb = np.cos(bearing_rad), np.sin(bearing_rad)
    pos_t0 = np.column_stack([ranges * cb, ranges * sb, np.full(K, target_h)])
    vel_t0 = np.tile(target_speed * np.array([np.cos(target_heading_rad),
                                              np.sin(target_heading_rad), 0.0]), (K, 1))
    pos_m0 = np.tile(launcher_pos, (K, 1))
    vel_m0 = np.tile(launcher_vel, (K, 1))
    hit, _, _, _, _ = _simulate_batch(
        pos_m0, vel_m0, pos_t0, vel_t0, mode=mode, turn_dir=turn_dir,
        evasive_g=evasive_g, evade_delay=evade_delay, dt=dt, t_max=t_max, params=params)
    return hit


def _boundary(ranges, kind, probe_fn, n_refine=2, refine_pts=9):
    """在 ranges 上批量探测命中区间，并细化求边界。

    kind='right' 求命中区间的右边界（R_max/R_nez），'left' 求左边界（R_min）。
    返回边界值；若全区间无命中返回 None；若右边界超出探测上界则返回上界（截断）。
    """
    hits = probe_fn(ranges)
    if not hits.any():
        return None
    idxs = np.flatnonzero(hits)
    if kind == "right":
        idx = idxs[-1]
        if idx == ranges.size - 1:          # 命中直到上界：上界截断
            return float(ranges[-1])
        lo, hi = float(ranges[idx]), float(ranges[idx + 1])
    else:
        idx = idxs[0]
        lo = float(ranges[idx - 1]) if idx > 0 else float(max(ranges[0] * 0.5, 50.0))
        hi = float(ranges[idx])
    for _ in range(n_refine):
        rs = np.linspace(lo, hi, refine_pts + 2)[1:-1]
        hits = probe_fn(rs)
        for r, h in zip(rs, hits):
            if kind == "right":
                if h:
                    lo = max(lo, float(r))
                else:
                    hi = min(hi, float(r))
            else:
                if h:
                    hi = min(hi, float(r))
                else:
                    lo = max(lo, float(r))
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# 攻击区（LAE）/不可逃逸区（NEZ）快速解算
# ----------------------------------------------------------------------------
@dataclass
class LaunchEnvelope:
    """发射包线解算结果（SI 单位 m）。"""
    r_max: float | None   # 最大发射距离（对直飞目标）m
    r_min: float | None   # 最小发射距离（引信解除保险 + 运动学最小）m
    r_nez: float | None   # 不可逃逸距离（对 9g 规避目标）m
    bearing_deg: float
    launcher_h: float
    target_h: float
    launcher_speed: float
    target_speed: float
    evasive_g: float
    target_heading_deg: float
    r_max_capped: bool = False   # r_max 是否达到探测上界被截断


def launch_envelope(launcher_h, launcher_speed, target_h, target_speed,
                    bearing_deg, target_heading_deg=None, launcher_heading_deg=0.0,
                    evasive_g=9.0, evade_delay=1.0, dt=0.2, t_max=100.0,
                    params=None, r_hi=180e3) -> LaunchEnvelope:
    """解算攻击区（LAE）与不可逃逸区（NEZ）。

    参数（SI 单位）：
        launcher_h/target_h      : 发射机/目标高度 m
        launcher_speed/target_speed: 发射机/目标速度大小 m/s
        bearing_deg              : 目标相对发射机的方位角 deg（相对发射机航向，
                                   0=正前方，±180=正后方；水平面内）
        launcher_heading_deg     : 发射机航向 deg（默认 0，沿 +X）
        target_heading_deg       : 目标航向 deg；默认 None = 目标朝向发射机飞行
        evasive_g                : NEZ 解算时目标规避过载 g（默认 9，战斗机典型值）
    返回 LaunchEnvelope(r_max, r_min, r_nez)。

    解算方法：以 3DOF 比例导引导弹模型批量仿真为基准——直飞目标→R_max/R_min，
    双方向最大过载盘旋规避目标→R_nez（取两种规避方向中更不利者）。
    """
    p = dict(MISSILE_PARAMS)
    if params:
        p.update(params)
    bearing = np.deg2rad(bearing_deg)
    heading_a = np.deg2rad(launcher_heading_deg)
    launcher_pos = np.array([0.0, 0.0, launcher_h])
    launcher_vel = launcher_speed * np.array([np.cos(heading_a), np.sin(heading_a), 0.0])
    # 目标航向（水平面）：默认朝向发射机；注意相对方位角叠加发射机航向
    if target_heading_deg is None:
        target_heading = np.arctan2(-np.sin(heading_a + bearing),
                                    -np.cos(heading_a + bearing))
    else:
        target_heading = np.deg2rad(target_heading_deg)
    target_heading_deg_out = float(np.rad2deg(target_heading)) % 360.0

    def probe(ranges, mode, turn_dir=1.0):
        return _probe_hits(ranges, launcher_pos, launcher_vel, target_h, target_speed,
                           target_heading, heading_a + bearing, mode, turn_dir,
                           evasive_g, evade_delay, dt, t_max, p)

    # R_max：直飞目标（右边界）
    r_scan = np.geomspace(1e3, r_hi, 33)
    r_max = _boundary(r_scan, "right", lambda rs: probe(rs, "straight"))
    capped = r_max is not None and r_max >= r_hi * 0.999
    # R_nez：9g 规避目标，左右两个规避方向取更小值（更保守）
    r_nez_candidates = []
    for turn_dir in (+1.0, -1.0):
        r = _boundary(r_scan, "right", lambda rs, d=turn_dir: probe(rs, "evade", d))
        if r is not None:
            r_nez_candidates.append(r)
    r_nez = min(r_nez_candidates) if r_nez_candidates else None
    if r_nez is not None and r_max is not None:
        r_nez = min(r_nez, r_max)
    # R_min：直飞目标（左边界），且不小于引信解除保险距离
    r_min_scan = np.geomspace(100.0, min(60e3, r_hi), 33)
    r_min = _boundary(r_min_scan, "left", lambda rs: probe(rs, "straight"))
    if r_min is None and r_max is not None:
        r_min = r_max  # 退化情形：整个探测区间均命中
    elif r_min is not None:
        r_min = max(r_min, p["arm_dist"])
        if r_max is not None:
            r_min = min(r_min, r_max)

    return LaunchEnvelope(r_max=r_max, r_min=r_min, r_nez=r_nez,
                          bearing_deg=float(bearing_deg), launcher_h=float(launcher_h),
                          target_h=float(target_h), launcher_speed=float(launcher_speed),
                          target_speed=float(target_speed), evasive_g=float(evasive_g),
                          target_heading_deg=target_heading_deg_out,
                          r_max_capped=capped)
