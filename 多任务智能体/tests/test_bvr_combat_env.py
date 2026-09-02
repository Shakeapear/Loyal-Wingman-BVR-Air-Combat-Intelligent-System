# -*- coding: utf-8 -*-
"""
tests/test_bvr_combat_env.py（交付物 D2.3-3 单元测试：BVRCombatEnv 与 JSBSimGymBridge）
==================================================================================
覆盖：NED→ENU 坐标转换、动作/观测空间、动作映射、随机 seed 复现、
奖励与终止全分支（命中/被命中/坠毁/燃油/逃逸/超时/安全违规/攻击区）、
ECM 对敌方雷达的影响、导弹发射→飞行→命中全链路（迎头 40 km）、
批量导弹步进与 Missile.simulate 的物理等价性。
运行（DC 环境）：python -m pytest 多任务智能体/tests -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from common.jsbsim_bridge import ned_ft_to_enu_m, wrap_pi
from common.bvr_combat_env import (
    BVRCombatEnv, make_bvr_env, OBS_LOW, OBS_HIGH, OBS_FIELDS_BVR, _missile_batch_step,
)
from common.missile_model import Missile

NOOP = {"flight": np.array([0.55, 0.0, 0.0, 0.0], dtype=np.float32),
        "weapon": np.array([0, 0, 0])}


def make_env(**cfg):
    cfg.setdefault("compute_envelope", False)   # 单测跳过攻击区解算（~1.2 s/次）
    return BVRCombatEnv(config=cfg)


# ---------------------------------------------------------------------------
# 坐标转换
# ---------------------------------------------------------------------------
def test_ned_ft_to_enu_m():
    """JSBSim NED+英制 → ENU+SI：东/北/上 顺序与 0.3048 换算、下降速度为负。"""
    pos, vel = ned_ft_to_enu_m(1000.0, -500.0, 15000.0, 300.0, -200.0, 100.0)
    assert np.allclose(pos, [1000.0, -500.0, 4572.0])
    assert np.allclose(vel, [91.44, -60.96, -30.48])


def test_wrap_pi():
    assert wrap_pi(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_pi(-np.pi - 0.2) == pytest.approx(np.pi - 0.2)
    assert wrap_pi(0.0) == 0.0


# ---------------------------------------------------------------------------
# 空间与动作映射
# ---------------------------------------------------------------------------
def test_action_observation_spaces():
    env = make_env()
    assert env.action_space["flight"].shape == (4,)
    assert np.all(env.action_space["flight"].low == [0, -1, -1, -1])
    assert np.all(env.action_space["flight"].high == [1, 1, 1, 1])
    assert list(env.action_space["weapon"].nvec) == [2, 2, 2]
    assert env.observation_space.shape == (len(OBS_FIELDS_BVR),)
    assert np.all(np.isfinite(env.observation_space.low))
    assert np.all(np.isfinite(env.observation_space.high))
    assert env.observation_space.dtype == np.float32
    env.own.close(); env.enemy.close()


def test_action_maps_to_fcs_channels():
    """连续动作直接映射到 F-104 人工通道（AP 主开关关闭）。"""
    env = make_env()
    env.reset(seed=0)
    env.step({"flight": np.array([0.3, 0.5, -0.2, 0.1], dtype=np.float32),
              "weapon": np.array([0, 0, 0])})
    assert env.own._get("fcs/ap-master-on") == 0
    assert env.own._get("fcs/throttle-cmd-pilot") == pytest.approx(0.3)
    assert env.own._get("fcs/aileron-cmd-norm") == pytest.approx(0.5)
    assert env.own._get("fcs/elevator-cmd-norm") == pytest.approx(-0.2)
    assert env.own._get("fcs/rudder-cmd-norm") == pytest.approx(0.1)
    env.own.close(); env.enemy.close()


# ---------------------------------------------------------------------------
# reset / 复现 / 随机 rollout
# ---------------------------------------------------------------------------
def test_reset_seed_reproducibility():
    """相同 seed 的两次 reset 给出完全相同的初始观测。"""
    e1, e2 = make_env(), make_env()
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    assert np.allclose(o1, o2)
    o3, _ = e1.reset(seed=8)
    assert not np.allclose(o1, o3)
    e1.own.close(); e1.enemy.close(); e2.own.close(); e2.enemy.close()


def test_reset_random_geometry_ranges():
    """随机初始态势范围：距离 30~80 km、相对方位 ±60°、高度 8000~20000 ft。"""
    env = make_env()
    for seed in range(5):
        _, info = env.reset(seed=seed)
        assert 30e3 <= info["dist_m"] <= 80e3 + 1.0
        assert 8000 * 0.3048 - 5 <= info["h_sl_m"] <= 20000 * 0.3048 + 5
    env.own.close(); env.enemy.close()


def test_random_action_rollout_no_error():
    """随机动作 40 步无报错、观测始终在有限空间内（100 步长跑见验证脚本）。"""
    env = make_env(max_steps=200)
    env.reset(seed=3)
    for _ in range(40):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert np.all(obs >= env.observation_space.low - 1e-6)
        assert np.all(obs <= env.observation_space.high + 1e-6)
        if term or trunc:
            env.reset()
    env.own.close(); env.enemy.close()


# ---------------------------------------------------------------------------
# 终止与奖励分支
# ---------------------------------------------------------------------------
def test_timeout_termination():
    env = make_env(max_steps=3)
    env.reset(seed=0)
    for _ in range(2):
        _, _, term, trunc, _ = env.step(NOOP)
        assert not term and not trunc
    _, _, term, trunc, info = env.step(NOOP)
    assert trunc and not term
    assert info["terminated_reason"] == "timeout"
    env.own.close(); env.enemy.close()


def test_escape_termination():
    env = make_env(escape_radius_m=200.0)
    env.reset(seed=0)
    _, _, term, trunc, info = env.step(NOOP)
    assert term and not trunc
    assert info["terminated_reason"] == "escape"
    env.own.close(); env.enemy.close()


def test_fuel_out_termination_and_reward():
    env = make_env()
    env.reset(seed=0)
    env.own._set("propulsion/tank[0]/contents-lbs", 0)
    env.own._set("propulsion/tank[1]/contents-lbs", 0)
    env.own._set("propulsion/tank[2]/contents-lbs", 0)
    _, r, term, trunc, info = env.step(NOOP)
    assert term and not trunc
    assert info["terminated_reason"] == "fuel_out"
    assert r == pytest.approx(-50.0 + r - r) or r <= -50.0 + 6.0   # -50 叠加攻击区/塑造项
    env.own.close(); env.enemy.close()


def test_own_crash_termination():
    env = make_env(crash_agl_m=40000.0)   # 阈值高于一切高度 → 立即触发坠毁判定
    env.reset(seed=0)
    _, _, term, trunc, info = env.step(NOOP)
    assert term and not trunc
    assert info["terminated_reason"] == "own_crash"
    env.own.close(); env.enemy.close()


def test_enemy_hit_reward():
    """本机导弹命中敌机：+200 并终止（注入位于敌机航向前方、以 500 m/s 逼近的导弹）。"""
    env = make_env()
    env.reset(seed=0)
    s_enemy = env.enemy.state_dict()
    u = s_enemy["v_enu_mps"] / np.linalg.norm(s_enemy["v_enu_mps"])
    env.own_missiles.append(Missile(pos=env._enemy_abs_pos(s_enemy) + 500.0 * u,
                                    vel=s_enemy["v_enu_mps"] - 500.0 * u))
    _, r, term, trunc, info = env.step(NOOP)
    assert term and not trunc
    assert info["terminated_reason"] == "enemy_hit"
    assert r >= 200.0 - 0.5
    env.own.close(); env.enemy.close()


def test_own_hit_reward():
    """被敌方导弹命中：-200 并终止（注入位于本机航向前方、以 500 m/s 逼近的敌导弹）。"""
    env = make_env()
    env.reset(seed=0)
    s_own = env.own.state_dict()
    u = s_own["v_enu_mps"] / np.linalg.norm(s_own["v_enu_mps"])
    env.enemy_missiles.append(Missile(pos=s_own["pos_enu_m"] + 500.0 * u,
                                      vel=s_own["v_enu_mps"] - 500.0 * u))
    env.r_max_own = None          # 排除攻击区 +5/步 奖励干扰
    _, r, term, trunc, info = env.step(NOOP)
    assert term and not trunc
    assert info["terminated_reason"] == "own_hit"
    assert r <= -200.0 + 0.5
    env.own.close(); env.enemy.close()


def test_in_zone_reward():
    """进入攻击区 +5/步（人为放宽包线阈值保证成立）。"""
    env = make_env()
    env.reset(seed=0)
    env.r_max_own, env.r_min_own = 300e3, 100.0
    _, r, _, _, info = env.step(NOOP)
    assert info["in_zone"] is True
    assert r == pytest.approx(5.0, abs=1.0)   # 5 + 接近率塑造（|塑造|≤0.5）
    env.own.close(); env.enemy.close()


def test_safety_violation_reward():
    """安全违规 -10/步（人为把升限约束压到 0 强制触发；排除攻击区奖励干扰）。"""
    env = make_env(safety_alt_max_m=0.0)
    env.reset(seed=0)
    env.r_max_own = None          # 排除 +5/步 攻击区奖励
    _, r, _, _, _ = env.step(NOOP)
    assert r <= -10.0 + 0.5
    env.own.close(); env.enemy.close()


def test_ecm_blocks_enemy_radar():
    """ECM 开启后敌方雷达有效距离减半：60 km 处失跟踪（80→40 km）。"""
    env = make_env()
    env.reset(seed=0)
    # 固定几何：本机 (30,120) 航向 0°，敌机正北 60 km 航向 180°（迎头）
    env.own.reset(seed=1, h_ft=15000.0, mach=0.85, heading_deg=0.0, autopilot=False)
    env.enemy.reset(seed=1, h_ft=15000.0, mach=0.85, heading_deg=180.0,
                    lat0_deg=30.0 + 60000.0 / 111320.0, lon0_deg=120.0, autopilot=True)
    env._enemy_offset = np.array([0.0, 60000.0])
    env.enemy_n_left = 0
    env.step({"flight": NOOP["flight"], "weapon": np.array([0, 0, 0])})
    assert env.enemy_radar.state("own") == "TRACK"       # 60 < 80 km：跟踪
    env.step({"flight": NOOP["flight"], "weapon": np.array([0, 0, 1])})  # 开启 ECM
    assert env.ecm_on
    assert env.enemy_radar.state("own") != "TRACK"       # 60 > 40 km：失跟踪
    env.own.close(); env.enemy.close()


def test_weapon_switch_toggle():
    env = make_env()
    env.reset(seed=0)
    env.step({"flight": NOOP["flight"], "weapon": np.array([0, 1, 0])})
    assert env.own_selected == 1
    env.step({"flight": NOOP["flight"], "weapon": np.array([0, 1, 0])})
    assert env.own_selected == 0
    env.own.close(); env.enemy.close()


# ---------------------------------------------------------------------------
# 导弹全链路与物理等价性
# ---------------------------------------------------------------------------
def test_missile_full_chain_headon():
    """全链路：迎头 40 km 发射 → 飞行 → 命中敌机（确定性态势，敌方不还击）。"""
    env = make_env()
    env.reset(seed=0)
    # 重建确定性迎头态势：本机 (30,120) 航向 0°，敌机正北 40 km 航向 180°
    env.own.reset(seed=1, h_ft=15000.0, mach=0.85, heading_deg=0.0, autopilot=False)
    env.enemy.reset(seed=1, h_ft=15000.0, mach=0.85, heading_deg=180.0,
                    lat0_deg=30.0 + 40000.0 / 111320.0, lon0_deg=120.0, autopilot=True)
    env._enemy_offset = np.array([0.0, 40000.0])
    env.r_max_own = env.r_max_enemy = 60e3
    env.r_min_own = env.r_min_enemy = 1e3
    env.enemy_n_left = 0                       # 敌方不还击，保证确定性命中敌机
    fired = False
    reason, info = None, {}
    for i in range(180):
        a = {"flight": np.array([0.55, 0.0, -0.08, 0.0], dtype=np.float32),
             "weapon": np.array([1, 0, 0]) if not fired else np.array([0, 0, 0])}
        _, r, term, trunc, info = env.step(a)
        if info["own_fired"] >= 1:
            fired = True
        if term or trunc:
            reason = info["terminated_reason"]
            break
    assert fired, "本机未成功发射导弹（雷达跟踪/攻击区判定失败）"
    assert reason == "enemy_hit", f"迎头 40 km 发射未命中（结果: {reason}）"
    assert len(env.enemy_missiles) == 0, "敌方不应发射任何导弹（enemy_n_left=0）"
    env.own.close(); env.enemy.close()


def test_missile_batch_step_equivalence():
    """批量步进（_missile_batch_step, dt=0.05）与 Missile.simulate(dt=0.05) 物理等价。"""
    def straight_target(pos0, vel0):
        pos0 = np.asarray(pos0, float); vel0 = np.asarray(vel0, float)
        def motion(t):
            return pos0 + vel0 * t, vel0, None
        return motion

    pos0 = np.array([0.0, 0.0, 6000.0])
    vel0 = np.array([340.0, 0.0, 0.0])
    tgt0 = np.array([20000.0, 0.0, 6000.0])
    tgt_v = np.array([-340.0, 0.0, 0.0])
    # 标量参考
    ref = Missile(pos0.copy(), vel0.copy()).simulate(straight_target(tgt0, tgt_v),
                                                     dt=0.05, t_max=60.0)
    # 批量步进
    m = Missile(pos0.copy(), vel0.copy())
    for i in range(int(60.0 / 0.05) + 1):
        tp, tv, _ = straight_target(tgt0, tgt_v)(i * 0.05)
        _missile_batch_step([m], tp, tv, 0.05)
        if not m.alive:
            break
    assert m.hit == ref.hit
    assert abs(m.miss_distance - ref.miss_distance) < 0.5
    # 侧向（含纯追踪/过载限幅路径）也验证一次
    tgt0b = np.array([0.0, 12000.0, 6000.0])
    tgt_vb = np.array([0.0, 340.0, 0.0])
    ref2 = Missile(pos0.copy(), vel0.copy()).simulate(straight_target(tgt0b, tgt_vb),
                                                      dt=0.05, t_max=60.0)
    m2 = Missile(pos0.copy(), vel0.copy())
    for i in range(int(60.0 / 0.05) + 1):
        tp, tv, _ = straight_target(tgt0b, tgt_vb)(i * 0.05)
        _missile_batch_step([m2], tp, tv, 0.05)
        if not m2.alive:
            break
    assert m2.hit == ref2.hit
    assert abs(m2.miss_distance - ref2.miss_distance) < 0.5


def test_make_bvr_env_factory():
    env = make_bvr_env(seed=1, config={"compute_envelope": False})
    assert env.steps == 0
    obs, _ = env.reset(seed=1)
    assert obs.shape == (len(OBS_FIELDS_BVR),)
    env.own.close(); env.enemy.close()
