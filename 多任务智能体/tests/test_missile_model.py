# -*- coding: utf-8 -*-
"""
tests/test_missile_model.py（交付物 D2.2-4 单元测试：导弹模型）
============================================================
覆盖：3DOF 动力学（迎头/尾追/侧向三种态势命中）、脱靶（超射程/大离轴角）、
过载限幅、攻击区（LAE/NEZ）解算趋势、模型参数一致性。
运行（DC 环境）：python -m pytest 多任务智能体/tests -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from common.missile_model import (
    G0, MISSILE_PARAMS, Missile, EngagementResult, LaunchEnvelope,
    air_density, launch_envelope,
)

G_LIM = MISSILE_PARAMS["n_max_g"]
KILL = MISSILE_PARAMS["kill_radius"]


def straight_target(pos0, vel0):
    """常速直线飞行目标（返回 target_motion 可调用对象）。"""
    pos0 = np.asarray(pos0, dtype=float)
    vel0 = np.asarray(vel0, dtype=float)

    def motion(t):
        return pos0 + vel0 * t, vel0, None

    return motion


# ---------------------------------------------------------------------------
# 基础工具与参数
# ---------------------------------------------------------------------------
def test_import_pure_numpy():
    """模块仅依赖 numpy，可被 `from common.missile_model import Missile` 直接导入。"""
    import common.missile_model as mm
    assert hasattr(mm, "Missile")
    assert hasattr(mm, "launch_envelope")


def test_isa_air_density_monotonic_and_values():
    """ISA 密度随高度单调递减，且与标准值一致（0/6/11 km）。"""
    assert air_density(0.0) == pytest.approx(1.225, rel=1e-3)
    assert air_density(6000.0) == pytest.approx(0.660, rel=1e-2)
    assert air_density(11000.0) == pytest.approx(0.364, rel=1e-2)
    hs = np.linspace(0.0, 20000.0, 41)
    rho = air_density(hs)
    assert np.all(np.diff(rho) < 0)


def test_thrust_and_mass_profile():
    """双推力曲线与质量消耗满足 T = g·Isp·ṁ（两段常值质量流量）。"""
    p = dict(MISSILE_PARAMS)
    m = Missile(np.zeros(3), np.array([300.0, 0.0, 0.0]))
    assert m.thrust(0.0) == p["thr_boost"]
    assert m.thrust(p["t_boost"] - 1e-9) == p["thr_boost"]
    assert m.thrust(p["t_boost"]) == p["thr_sustain"]
    assert m.thrust(p["t_sustain"] - 1e-9) == p["thr_sustain"]
    assert m.thrust(p["t_sustain"]) == 0.0
    # 质量守恒：燃料全部烧完后的质量 = m0 - (Tb·tb + Ts·ts)/(Isp·g0)
    fuel = (p["thr_boost"] * p["t_boost"] + p["thr_sustain"] * (p["t_sustain"] - p["t_boost"])) / (p["isp"] * G0)
    assert m.mass(0.0) == p["mass0"]
    assert m.mass(p["t_sustain"]) == pytest.approx(p["mass0"] - fuel, rel=1e-12)
    assert m.mass(1000.0) == pytest.approx(p["mass0"] - fuel, rel=1e-12)


# ---------------------------------------------------------------------------
# 三种态势命中（迎头 / 尾追 / 侧向）
# ---------------------------------------------------------------------------
def test_headon_hit():
    """迎头态势：20 km 同高度对头，必须命中且脱靶量 < 杀伤半径。"""
    r0 = 20000.0
    m = Missile(np.array([0.0, 0.0, 6000.0]), np.array([340.0, 0.0, 0.0]))
    r = m.simulate(straight_target(np.array([r0, 0.0, 6000.0]),
                                   np.array([-340.0, 0.0, 0.0])), dt=0.05, t_max=60.0)
    assert r.hit, f"迎头发射未命中（脱靶量 {r.miss_distance:.1f} m）"
    assert r.miss_distance <= KILL
    # 解析估计：初始接近率 680 m/s、末段 >1300 m/s，命中时间应在 10~25 s 量级
    assert 8.0 < r.t_impact < 25.0
    # 速度合理：超过 M2（600 m/s）、不超过 M4.2（1400 m/s）
    assert 600.0 < r.max_speed_mps < 1400.0
    # 过载不超上限（含重力补偿，迎头接近纯直线弹道，侧向过载应很小）
    assert r.max_lat_g <= G_LIM + 1e-3
    assert r.max_lat_cmd_g <= G_LIM + 1e-3
    assert r.max_lat_cmd_g < 5.0, "纯迎头弹道侧向指令过载应接近 1g（重力补偿）"


def test_tail_chase_hit():
    """尾追态势：8 km 追击逃逸目标（目标 M0.9），必须命中。"""
    r0 = 8000.0
    m = Missile(np.array([0.0, 0.0, 6000.0]), np.array([340.0, 0.0, 0.0]))
    r = m.simulate(straight_target(np.array([r0, 0.0, 6000.0]),
                                   np.array([300.0, 0.0, 0.0])), dt=0.05, t_max=120.0)
    assert r.hit, f"尾追发射未命中（脱靶量 {r.miss_distance:.1f} m）"
    assert r.miss_distance <= KILL
    assert r.max_lat_g <= G_LIM + 1e-3


def test_flank_hit():
    """侧向态势：12 km 正横方穿越目标（M1.0），比例导引应命中。"""
    m = Missile(np.array([0.0, 0.0, 6000.0]), np.array([340.0, 0.0, 0.0]))
    r = m.simulate(straight_target(np.array([0.0, 12000.0, 6000.0]),
                                   np.array([0.0, 340.0, 0.0])), dt=0.05, t_max=60.0)
    assert r.hit, f"侧向发射未命中（脱靶量 {r.miss_distance:.1f} m）"
    assert r.miss_distance <= KILL
    assert r.max_lat_g <= G_LIM + 1e-3


# ---------------------------------------------------------------------------
# 脱靶（超射程 / 大离轴角）与过载限幅
# ---------------------------------------------------------------------------
def test_miss_beyond_max_range():
    """超射程发射（160 km 迎头，远超本模型 R_max≈50~60 km）：必须脱靶。"""
    r0 = 160000.0
    m = Missile(np.array([0.0, 0.0, 6000.0]), np.array([340.0, 0.0, 0.0]))
    r = m.simulate(straight_target(np.array([r0, 0.0, 6000.0]),
                                   np.array([-340.0, 0.0, 0.0])), dt=0.2, t_max=120.0)
    assert not r.hit
    assert r.miss_distance > 1000.0


def test_miss_large_off_bore():
    """大离轴角态势：目标位于正横方 90°（大离轴角）、高速横穿（M2.0），
    视线角速度超过 30g 过载补偿能力，必须脱靶。"""
    r0, vt = 8000.0, 680.0
    pos0 = np.array([0.0, r0, 6000.0])
    vel0 = np.array([0.0, vt, 0.0])
    m = Missile(np.array([0.0, 0.0, 6000.0]), np.array([340.0, 0.0, 0.0]))
    r = m.simulate(straight_target(pos0, vel0), dt=0.05, t_max=60.0)
    assert not r.hit, f"大离轴角发射意外命中（脱靶量 {r.miss_distance:.1f} m）"
    assert r.miss_distance > 10.0 * KILL


def test_lateral_overload_limit_saturated():
    """过载限幅：高速正横穿越目标使指令饱和，实际施加的侧向过载不超过 30g。"""
    m = Missile(np.array([0.0, 0.0, 6000.0]), np.array([340.0, 0.0, 0.0]))
    r = m.simulate(straight_target(np.array([0.0, 5000.0, 6000.0]),
                                   np.array([0.0, 400.0, 0.0])), dt=0.05, t_max=60.0)
    assert r.max_lat_cmd_g >= G_LIM - 0.5, "该态势应触发过载饱和（指令 ≥ 30g）"
    assert r.max_lat_g <= G_LIM + 1e-3, "实际侧向过载超过 30g 上限"


# ---------------------------------------------------------------------------
# 攻击区（LAE/NEZ）解算
# ---------------------------------------------------------------------------
def _env(bearing, h=4572.0, va=280.0, vt=280.0):
    return launch_envelope(launcher_h=h, launcher_speed=va,
                           target_h=h, target_speed=vt, bearing_deg=bearing)


def test_envelope_sanity():
    """迎头攻击区：R_max 在公开量级内，NEZ ⊂ LAE，R_min 受引信保险距离限制。"""
    env = _env(0.0)
    assert isinstance(env, LaunchEnvelope)
    assert 20e3 < env.r_max < 160e3, f"R_max={env.r_max} 超出中距弹合理量级"
    assert env.r_min >= MISSILE_PARAMS["arm_dist"]
    assert env.r_nez is not None and env.r_nez > 0
    assert env.r_min < env.r_nez < env.r_max, "不满足 R_min < R_nez < R_max（NEZ ⊂ LAE）"


def test_envelope_bearing_trend():
    """R_max 随 |相对方位角| 增大而减小（迎头最大、侧向明显减小、正后方最小），且左右对称。"""
    r0 = _env(0.0).r_max
    r45 = _env(45.0).r_max
    r90 = _env(90.0).r_max
    r135 = _env(135.0).r_max
    r180 = _env(180.0).r_max
    for r in (r0, r45, r90, r135, r180):
        assert r is not None and r > 0
    assert r0 > r45 > r90            # 前半球随离轴角增大而减小
    assert r90 > r180                # 正后方（尾追几何）最小
    assert r135 == pytest.approx(r90, rel=0.15)  # 侧后方与正侧方量级相当
    # 对称性（正负方位角）
    assert _env(-90.0).r_max == pytest.approx(r90, rel=0.05)
    assert _env(-180.0).r_max == pytest.approx(r180, rel=0.10)


def test_envelope_altitude_trend():
    """R_max 随发射高度升高而增大（高空空气稀薄、阻力小）。"""
    r10 = _env(0.0, h=3048.0).r_max
    r20 = _env(0.0, h=6096.0).r_max
    r30 = _env(0.0, h=9144.0).r_max
    assert r10 < r20 < r30, f"攻击区未随高度增大：{r10/1e3:.1f}/{r20/1e3:.1f}/{r30/1e3:.1f} km"


def test_envelope_consistency_with_scalar_sim():
    """批量解算（launch_envelope）与逐帧仿真（Missile.simulate）交叉验证：
    在 R_max 内侧 3 km 发射应命中，R_max 外侧 8 km 应脱靶。"""
    env = launch_envelope(launcher_h=6096.0, launcher_speed=280.0,
                          target_h=6096.0, target_speed=280.0, bearing_deg=0.0)
    r_max = env.r_max
    r_in, r_out = r_max - 3000.0, r_max + 8000.0
    m_in = Missile(np.array([0.0, 0.0, 6096.0]), np.array([280.0, 0.0, 0.0]))
    res_in = m_in.simulate(straight_target(np.array([r_in, 0.0, 6096.0]),
                                           np.array([-280.0, 0.0, 0.0])),
                           dt=0.1, t_max=120.0)
    assert res_in.hit, f"R_max 内侧发射应命中（脱靶量 {res_in.miss_distance:.1f} m）"
    m_out = Missile(np.array([0.0, 0.0, 6096.0]), np.array([280.0, 0.0, 0.0]))
    res_out = m_out.simulate(straight_target(np.array([r_out, 0.0, 6096.0]),
                                             np.array([-280.0, 0.0, 0.0])),
                             dt=0.1, t_max=120.0)
    assert not res_out.hit, "R_max 外侧发射不应命中"


def test_envelope_nez_evasive_target():
    """NEZ 解算依据：对 9g 规避目标的命中距离应明显小于对直飞目标的 R_max。"""
    env = launch_envelope(launcher_h=6096.0, launcher_speed=280.0,
                          target_h=6096.0, target_speed=280.0, bearing_deg=0.0,
                          evasive_g=9.0)
    assert env.r_nez < 0.8 * env.r_max
