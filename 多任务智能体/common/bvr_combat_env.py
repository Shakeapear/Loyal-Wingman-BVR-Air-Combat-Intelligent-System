# -*- coding: utf-8 -*-
"""
common/bvr_combat_env.py（交付物 D2.3-1）
=========================================
BVRCombatEnv：1v1 超视距空战标准化 Gym 环境（gymnasium.Env）。

依据《大创项目实施计划书》步骤 2.3 设计：
- 动作空间：连续 [油门 0~1, 滚转 -1~1, 俯仰 -1~1, 偏航 -1~1]（映射到 F-104
  人工通道舵面：fcs/throttle-cmd-pilot / aileron / elevator / rudder-cmd-norm，
  AP 主开关关闭）+ 离散武器控制 [发射, 切换武器, ECM]（spaces.Dict + MultiDiscrete，
  兼容 stable-baselines3 MultiInputPolicy）；
- 状态空间：本机状态（位姿+速度+油量+武器）+ 目标信息（相对位置/速度/方位，
  由 FireControlRadar 提供，未探测到置零+标志位）+ 威胁信息（RWR/MAWS 告警）+
  任务标签 one-hot；有限 Box（维度与范围见 OBS_FIELDS_BVR）；
- 奖励：命中 +200、被命中 -200、进入攻击区 +5/步、安全违规 -10/步、
  燃油耗尽 -50（终止时）；另加小幅稠密塑造项（接近率，±0.5/步，缓解稀疏奖励，
  计划书允许的补充塑造项）；
- 终止：被命中、坠毁（h_agl<30 m）、燃油耗尽（<50 lb）、逃逸区域（距原点>150 km）、
  episode 超时（默认 240 步）；敌方坠毁同样终止（奖励 0）。

与步骤 2.2 模块的集成（不改 2.2 接口，仅 import 使用）：
- 发射判定：reset 时用 common.missile_model.launch_envelope() 解算双方攻击区
  （R_max/R_min/R_nez），episode 内作为静态包线阈值使用（计算耗时约 1 s，
  仅在 reset 执行；近似处理，见 README_环境.md §4.2）；
- 导弹飞行：common.missile_model.Missile 对象按导弹亚步（默认 dt=0.2 s × 5，
  与 2.2 攻击区解算器同 dt；向量化批处理 + 目标轨迹连续插值）推进，
  脱靶量 < 10 m 判定命中；
- 探测/跟踪：common.radar_model.FireControlRadar 状态机（探测→丢失→重捕获）；
- 告警：common.radar_model.RWR（锁定告警+15° 量化方位）、MAWS（TTA）。

坐标系：全部 ENU + SI（东/北/上，米、米/秒），JSBSim NED/英制经
common.jsbsim_bridge.ned_ft_to_enu_m 转换。

可视化（步骤 2.4 交付物 D2.4-1）：可视化本体是项目级工具，独立在 `可视化工具/`
（定位：训练自我调整 + 成果展示；不含任何智能体逻辑），本环境只提供数据接口：
get_viz_frame() 返回全量态势快照；render() 支持 "human"（实时战术显示）与
"rgb_array"（离屏帧），经 config["render_mode"] 启用，默认 None（训练零开销），
首次调用时惰性接入该工具（见 可视化工具/README.md）。

典型用法：
    from common.bvr_combat_env import make_bvr_env
    env = make_bvr_env()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

并行（SubprocVecEnv，工厂函数必须模块顶层可导入）：
    from stable_baselines3.common.vec_env import SubprocVecEnv
    vec = SubprocVecEnv([make_bvr_env for _ in range(4)])
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from common.jsbsim_bridge import JSBSimGymBridge, wrap_pi
from common.missile_model import Missile, launch_envelope, air_density, G0, MISSILE_PARAMS
from common.radar_model import FireControlRadar, RWR, MAWS

DECISION_DT = 1.0                     # 决策周期 s（1 Hz）
MISSILE_DT_DEFAULT = 0.2              # 导弹亚步积分步长 s（与 2.2 launch_envelope 解算的
                                      # dt=0.2 一致；命中判定由连续碰撞检测保证）
INNER_DT_DEFAULT = 1.0 / 60.0         # 空战环境 JSBSim 内环步长 s（60 Hz；
                                      # 5 任务技能环境仍为 120 Hz，见 jsbsim_env.py）

_GV = np.array([0.0, 0.0, -G0])
_GZ = np.array([0.0, 0.0, G0])
_ZUP = np.array([0.0, 0.0, 1.0])

_VIZ_TOOL_DIR = Path(__file__).resolve().parents[2] / "可视化工具"


def _ensure_viz_tool_on_path():
    """把项目级 可视化工具/ 目录加入 sys.path（render() 惰性导入 visualization 包用）。

    可视化工具与本库分离（见 可视化工具/README.md）：本库只提供 get_viz_frame()
    数据接口，渲染实现由该工具提供；训练路径（render_mode=None）不触发此调用。
    """
    path = str(_VIZ_TOOL_DIR)
    if path not in sys.path:
        sys.path.append(path)


def _missile_batch_step(missiles, target_pos, target_vel, dt):
    """对一组 Missile 做一次向量化积分（与 Missile.step 完全等价的 3DOF 物理）。

    用于 BVRCombatEnv 每决策步内的导弹亚步推进（纯 numpy 批处理，避免逐个
    Missile.step 的 Python/numpy 小数组开销，保证 4 实例并行吞吐 ≥1000 steps/s；
    热路径用逐分量算术替代 np.cross/np.einsum，进一步压低小数组开销）。
    物理公式与 common.missile_model.Missile.step 逐项一致（TPN+后半球纯追踪+
    重力补偿+30g 限幅+双推力+阻力+连续碰撞检测），由 test_bvr_combat_env.py
    的等价性用例与 Missile.simulate 对拍验证。只写回 Missile 公开属性，
    不改 2.2 模块接口。
    """
    if not missiles:
        return
    p = missiles[0].p
    pos = np.stack([m.pos for m in missiles])
    vel = np.stack([m.vel for m in missiles])
    t = np.array([m.t for m in missiles])
    miss = np.array([m.miss_distance for m in missiles])
    max_g = np.array([m.max_lat_g for m in missiles])
    max_v = np.array([m.max_speed_mps for m in missiles])
    alive = np.array([m.alive for m in missiles], dtype=bool)

    # ---- 几何/连续碰撞检测（逐分量算术）----
    rv = target_pos - pos
    rn = np.sqrt((rv * rv).sum(axis=1))
    np.maximum(rn, 1e-9, out=rn)
    p_rel = pos - target_pos
    v_rel = vel - target_vel
    dot_pv = (p_rel * v_rel).sum(axis=1)
    v2 = (v_rel * v_rel).sum(axis=1)
    t_ca = np.where((dot_pv < 0.0) & (v2 > 1e-12),
                    np.clip(-dot_pv / np.maximum(v2, 1e-12), 0.0, dt), 0.0)
    d_min = np.sqrt(((p_rel + v_rel * t_ca[:, None]) ** 2).sum(axis=1))
    miss = np.minimum(miss, d_min)
    new_hit = alive & (d_min <= p["kill_radius"])
    alive &= ~new_hit
    speed = np.sqrt((vel * vel).sum(axis=1))
    max_v = np.maximum(max_v, speed)
    dead = alive & ((speed < p["v_min"]) | (pos[:, 2] <= 0.0))
    alive &= ~dead
    if not alive.any():
        _missile_batch_writeback(missiles, pos, vel, t, miss, max_g, max_v, alive, new_hit)
        return

    # ---- 制导（与 Missile.guidance 一致：TPN + 后半球纯追踪 + 重力补偿 + 限幅）----
    e = rv / rn[:, None]
    v_rel = target_vel - vel          # 注意符号：目标减导弹（与 Missile.guidance 一致）
    # omega = e × v_rel / rn ; a_pn = N * (omega × vel)
    ox = (e[:, 1] * v_rel[:, 2] - e[:, 2] * v_rel[:, 1]) / rn
    oy = (e[:, 2] * v_rel[:, 0] - e[:, 0] * v_rel[:, 2]) / rn
    oz = (e[:, 0] * v_rel[:, 1] - e[:, 1] * v_rel[:, 0]) / rn
    a_pn = np.empty_like(pos)
    a_pn[:, 0] = p["n_gain"] * (oy * vel[:, 2] - oz * vel[:, 1])
    a_pn[:, 1] = p["n_gain"] * (oz * vel[:, 0] - ox * vel[:, 2])
    a_pn[:, 2] = p["n_gain"] * (ox * vel[:, 1] - oy * vel[:, 0])
    u = vel / np.maximum(speed, 1e-6)[:, None]
    cos_ue = (u * e).sum(axis=1)
    use_pp = cos_ue < 0.0
    e_perp = e - cos_ue[:, None] * u
    e_norm = np.sqrt((e_perp * e_perp).sum(axis=1))
    degn = e_norm < 1e-6
    # 正后方退化：e_perp = ẑ × u
    zp = np.empty_like(e_perp)
    zp[:, 0] = -u[:, 1]
    zp[:, 1] = u[:, 0]
    zp[:, 2] = 0.0
    e_perp = np.where(degn[:, None], zp, e_perp)
    a_pp = p["pp_gain"] * speed[:, None] * e_perp
    a_pn = np.where(use_pp[:, None], a_pp, a_pn) + _GZ
    a_lim = p["n_max_g"] * G0
    a_norm = np.sqrt((a_pn * a_pn).sum(axis=1))
    a_lat = a_pn * np.minimum(1.0, a_lim / np.maximum(a_norm, 1e-12))[:, None]
    max_g = np.maximum(max_g, np.sqrt((a_lat * a_lat).sum(axis=1)) / G0)

    # ---- 轴向（双推力 - 阻力）----
    tb, ts = p["t_boost"], p["t_sustain"]
    md_b = p["thr_boost"] / (p["isp"] * G0)
    md_s = p["thr_sustain"] / (p["isp"] * G0)
    m_t = p["mass0"] - md_b * np.minimum(t, tb) - md_s * np.maximum(0.0, np.minimum(t, ts) - tb)
    thr = np.where(t < tb, p["thr_boost"], np.where(t < ts, p["thr_sustain"], 0.0))
    rho = air_density(pos[:, 2])
    drag = 0.5 * rho * p["cd"] * p["s_ref"] * speed ** 2
    a_ax = (thr - drag) / m_t
    new_vel = vel + (a_ax[:, None] * u + a_lat + _GV) * dt
    new_pos = pos + new_vel * dt
    vel = np.where(alive[:, None], new_vel, vel)
    pos = np.where(alive[:, None], new_pos, pos)
    t = t + np.where(alive, dt, 0.0)
    _missile_batch_writeback(missiles, pos, vel, t, miss, max_g, max_v, alive, new_hit)


def _missile_batch_writeback(missiles, pos, vel, t, miss, max_g, max_v, alive, new_hit):
    for i, m in enumerate(missiles):
        m.pos = pos[i]
        m.vel = vel[i]
        m.t = t[i]
        m.miss_distance = float(miss[i])
        m.max_lat_g = float(max_g[i])
        m.max_speed_mps = float(max_v[i])
        m.alive = bool(alive[i])
        if bool(new_hit[i]):
            m.hit = True
            m.terminal_reason = "hit"
        elif not m.alive and m.terminal_reason == "":
            m.terminal_reason = "energy" if np.linalg.norm(vel[i]) < m.p["v_min"] else "ground"

# ---- 交战配置默认值（可经 config 覆盖）----
DEFAULT_CONFIG = {
    "max_steps": 240,            # episode 最大决策步（超时截断）
    "inner_dt": INNER_DT_DEFAULT,   # JSBSim 内环步长 s（60 Hz，吞吐权衡，见 README §4.1）
    "missile_dt": MISSILE_DT_DEFAULT,  # 导弹亚步步长 s（0.1 s，与 2.2 攻击区解算 dt=0.2 同量级）
    "escape_radius_m": 150e3,    # 逃逸区域半径（距初始点）
    "fuel_min_lbs": 50.0,        # 燃油耗尽阈值 lb
    "crash_agl_m": 30.0,         # 坠毁判定离地高度 m
    "rcs_own": 5.0,              # 双方 RCS m²（战斗机典型 3~5）
    "rcs_enemy": 5.0,
    "n_missiles": 2,             # 双方挂载中距弹数（D2.1 约束 2×中距弹）
    "enemy_speed_fps": 760.0,    # 敌方脚本策略巡航速度 fps
    "enemy_fire_interval_s": 8.0,  # 敌方连续发射最小间隔 s
    "compute_envelope": True,   # reset 时是否解算攻击区（约 1.2 s；单测/吞吐测试可关）
    "ecm_radar_range_factor": 0.5,  # 本机 ECM 开启时敌方雷达有效距离倍率
    "safety_nz_max_g": 5.5,      # 安全包线：正过载上限 g（D2.1 约束 +5G，留 10% 裕度）
    "safety_nz_min_g": -2.5,     # 负过载下限 g（D2.1 约束 -2G）
    "safety_mach_max": 1.65,     # 最大马赫（D2.1 约束 M1.6）
    "safety_alt_max_m": 14500.0, # 最大升限 m（D2.1 约束 14000 m + 500 m 裕度）
    "safety_vmin_mps": 90.0,     # 失速近似速度 m/s
    "reward_hit": 200.0,
    "reward_hit_by": -200.0,
    "reward_in_zone": 5.0,
    "reward_violation": -10.0,
    "reward_fuel_out": -50.0,
    "shaping_scale": 0.02,       # 接近率塑造项系数（奖励/米/千米）
    "render_mode": None,         # 可视化：None / "human"（实时战术显示）/ "rgb_array"（离屏帧）
}

# ---- 观测字段（28 维，SI 单位；与 OBS_LOW/OBS_HIGH 一一对应）----
# 0  h_sl_m            海拔 m
# 1  vtrue_mps         真空速 m/s
# 2  mach              马赫数
# 3  phi_rad           滚转角 rad
# 4  theta_rad         俯仰角 rad
# 5  psi_rad           航向角 rad（0=北，顺时针为正）
# 6  alpha_rad         迎角 rad
# 7  beta_rad          侧滑角 rad
# 8  nz_g              法向过载 g
# 9  throttle          油门 0~1
# 10 fuel_frac         剩余燃油比例
# 11 vd_mps            升降速度 m/s（上为正）
# 12 n_missiles        剩余导弹数
# 13 ecm_on            ECM 状态 0/1
# 14 tgt_e_m           目标相对东向位置 m（未探测到置 0）
# 15 tgt_n_m           目标相对北向位置 m（未探测到置 0）
# 16 tgt_alt_m         目标相对高度 m（未探测到置 0）
# 17 tgt_dist_m        目标距离 m（未探测到置 0）
# 18 tgt_bearing_rad   目标方位角 rad（未探测到置 0）
# 19 tgt_speed_mps     目标真空速 m/s（未探测到置 0）
# 20 tgt_heading_rad   目标航向 rad（未探测到置 0）
# 21 tgt_detected      本机雷达探测标志 0/1
# 22 rwr_alarm         RWR 锁定告警 0/1
# 23 rwr_bearing_rad   RWR 粗测方位 rad（无告警置 0）
# 24 maws_alarm        MAWS 导弹逼近告警 0/1
# 25 maws_tta_s        导弹到达时间估计 s（无告警置 0）
# 26 mission_onehot_0  任务标签 one-hot（BVR 截击，恒 1）
# 27 mission_onehot_1  任务标签 one-hot（预留，恒 0）
OBS_FIELDS_BVR = [
    "h_sl_m", "vtrue_mps", "mach", "phi_rad", "theta_rad", "psi_rad",
    "alpha_rad", "beta_rad", "nz_g", "throttle", "fuel_frac", "vd_mps",
    "n_missiles", "ecm_on",
    "tgt_e_m", "tgt_n_m", "tgt_alt_m", "tgt_dist_m", "tgt_bearing_rad",
    "tgt_speed_mps", "tgt_heading_rad", "tgt_detected",
    "rwr_alarm", "rwr_bearing_rad", "maws_alarm", "maws_tta_s",
    "mission_onehot_0", "mission_onehot_1",
]

OBS_LOW = np.array([
    0.0, 0.0, 0.0, -np.pi, -np.pi, -np.pi, -1.5, -1.5, -12.0, 0.0, 0.0, -400.0,
    0.0, 0.0,
    -200e3, -200e3, -30000.0, 0.0, -np.pi, 0.0, -np.pi, 0.0,
    0.0, -np.pi, 0.0, 0.0,
    0.0, 0.0,
], dtype=np.float32)
OBS_HIGH = np.array([
    25000.0, 700.0, 3.0, np.pi, np.pi, np.pi, 1.5, 1.5, 16.0, 1.0, 1.0, 400.0,
    2.0, 1.0,
    200e3, 200e3, 30000.0, 300e3, np.pi, 700.0, np.pi, 1.0,
    1.0, np.pi, 1.0, 200.0,
    1.0, 1.0,
], dtype=np.float32)


class BVRCombatEnv(gym.Env):
    """1v1 超视距空战环境（本机 = DRL 智能体，敌机 = JSBSim + 脚本策略）。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.cfg = dict(DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)
        self.render_mode = self.cfg.get("render_mode")
        self._display = None                # 实时战术显示（render_mode="human" 时惰性创建）
        self.inner_dt = float(self.cfg["inner_dt"])
        self.frames_per_step = int(round(DECISION_DT / self.inner_dt))
        self.missile_dt = float(self.cfg["missile_dt"])
        self.missile_steps = int(round(DECISION_DT / self.missile_dt))

        self.action_space = spaces.Dict({
            "flight": spaces.Box(low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
                                 high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                                 shape=(4,), dtype=np.float32),
            "weapon": spaces.MultiDiscrete([2, 2, 2]),   # [发射, 切换武器, ECM]
        })
        self.observation_space = spaces.Box(low=OBS_LOW.copy(), high=OBS_HIGH.copy(),
                                            shape=(len(OBS_FIELDS_BVR),), dtype=np.float32)

        # JSBSim：本机/敌机各一个实例（敌方用脚本策略控制）
        self.own = JSBSimGymBridge(aircraft="f104", sim_dt=self.inner_dt)
        self.enemy = JSBSimGymBridge(aircraft="f104", sim_dt=self.inner_dt)
        # 2.2 模块：雷达/RWR/MAWS + 导弹
        self.radar = FireControlRadar()
        self.enemy_radar = FireControlRadar()
        self.rwr = RWR()
        self.maws = MAWS()
        self.enemy_maws = MAWS()

        self.rng = np.random.default_rng()
        self._reset_state()

    # ------------------------------------------------------------------
    # 内部状态初始化
    # ------------------------------------------------------------------
    def _reset_state(self):
        self.steps = 0
        # 传感器每个 episode 全新实例（避免跨 reset 的跟踪/告警记忆残留，
        # 保证 reset(seed) → step 的确定性，env_checker 的确定性检查依赖此点）
        self.radar = FireControlRadar()
        self.enemy_radar = FireControlRadar()
        self.rwr = RWR()
        self.maws = MAWS()
        self.enemy_maws = MAWS()
        self._enemy_offset = np.zeros(2)      # 敌机相对原点（东/北）偏移 m
        self.own_missiles = []                # 本机在飞导弹列表（Missile）
        self.enemy_missiles = []              # 敌方在飞导弹列表
        self.own_n_left = self.cfg["n_missiles"]
        self.enemy_n_left = self.cfg["n_missiles"]
        self.own_selected = 0                 # 本机当前选中武器槽（0/1，接口预留）
        self.ecm_on = False
        self.r_max_own = self.r_min_own = self.r_nez_own = None
        self.r_max_enemy = self.r_min_enemy = None
        self._enemy_next_fire_t = 0.0
        self._prev_dist = None
        self._prev_violation = False
        self.last_info = {}
        self._enemy_heading_cmd = 0.0
        self._enemy_alt_cmd_ft = 15000.0
        self._enemy_fired = False           # 本决策步敌方是否发射（可视化事件标志）
        self.last_events = {"fired_own": False, "fired_enemy": False,
                            "hit_enemy": False, "hit_own": False}

    # ------------------------------------------------------------------
    # Gym 接口
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self._reset_state()

        # ---- 随机初始态势（全部随机量由 self.rng 顺序抽取，seed 可复现）----
        own_h_ft = float(self.rng.uniform(8000.0, 20000.0))
        own_mach = float(self.rng.uniform(0.7, 0.9))
        own_heading_deg = float(self.rng.uniform(0.0, 360.0))
        dist = float(self.rng.uniform(30e3, 80e3))          # 初始距离 30~80 km
        bearing_off_deg = float(self.rng.uniform(-60.0, 60.0))  # 目标相对方位 ±60°
        enemy_h_ft = float(self.rng.uniform(8000.0, 20000.0))
        enemy_mach = float(self.rng.uniform(0.7, 0.9))
        enemy_heading_jitter = float(self.rng.uniform(-30.0, 30.0))
        lat0, lon0 = 30.0, 120.0

        self.own.reset(seed=None, h_ft=own_h_ft, mach=own_mach,
                       heading_deg=own_heading_deg, lat0_deg=lat0, lon0_deg=lon0,
                       autopilot=False)
        # 敌机地理位置：以本机为原点，按绝对方位放置（相对方位 ±60°）
        bearing_abs_rad = wrap_pi(np.deg2rad(own_heading_deg + bearing_off_deg))
        east_m = dist * np.sin(bearing_abs_rad)
        north_m = dist * np.cos(bearing_abs_rad)
        lat_e = lat0 + north_m / 111320.0
        lon_e = lon0 + east_m / (111320.0 * np.cos(np.deg2rad(lat0)))
        enemy_heading_deg = wrap_pi(np.deg2rad(own_heading_deg + bearing_off_deg) + np.pi
                                    + np.deg2rad(enemy_heading_jitter)) * 180.0 / np.pi
        self.enemy.reset(seed=None, h_ft=enemy_h_ft, mach=enemy_mach,
                         heading_deg=enemy_heading_deg % 360.0, lat0_deg=lat_e,
                         lon0_deg=lon_e, autopilot=True)
        self._enemy_offset = np.array([east_m, north_m])
        self._enemy_heading_cmd = float(np.deg2rad(own_heading_deg + bearing_off_deg) + np.pi)
        self._enemy_alt_cmd_ft = own_h_ft

        # ---- 攻击区解算（reset 时一次，episode 内作为静态包线阈值）----
        s_own, s_enemy = self.own.state_dict(), self.enemy.state_dict()
        enemy_abs = self._enemy_abs_pos(s_enemy)
        rel = enemy_abs - s_own["pos_enu_m"]
        dist0 = float(np.hypot(rel[0], rel[1]))
        bearing_own_deg = float(np.rad2deg(wrap_pi(np.arctan2(rel[0], rel[1]) - s_own["psi_rad"])))
        bearing_enemy_deg = float(np.rad2deg(wrap_pi(np.arctan2(-rel[0], -rel[1]) - s_enemy["psi_rad"])))
        if not self.cfg["compute_envelope"]:
            self.r_max_own = self.r_max_enemy = 45e3
            self.r_min_own = self.r_min_enemy = 1e3
            self.r_nez_own = 20e3
        else:
            try:
                env_own = launch_envelope(launcher_h=s_own["h_sl_m"], launcher_speed=s_own["vtrue_mps"],
                                          target_h=s_enemy["h_sl_m"], target_speed=s_enemy["vtrue_mps"],
                                          bearing_deg=bearing_own_deg)
                self.r_max_own, self.r_min_own, self.r_nez_own = env_own.r_max, env_own.r_min, env_own.r_nez
                env_enemy = launch_envelope(launcher_h=s_enemy["h_sl_m"], launcher_speed=s_enemy["vtrue_mps"],
                                            target_h=s_own["h_sl_m"], target_speed=s_own["vtrue_mps"],
                                            bearing_deg=bearing_enemy_deg)
                self.r_max_enemy, self.r_min_enemy = env_enemy.r_max, env_enemy.r_min
            except Exception:
                # 极端随机态势解算失败时退化为保守阈值（不影响环境运行）
                self.r_max_own = self.r_max_enemy = 45e3
                self.r_min_own = self.r_min_enemy = 1e3
                self.r_nez_own = 20e3

        self._prev_dist = dist0
        self._prev_violation = False
        obs = self._build_obs()
        self.last_info = {"steps": 0, "dist_m": dist0, "reward": 0.0,
                          "terminated_reason": None,
                          "h_sl_m": s_own["h_sl_m"], "vtrue_mps": s_own["vtrue_mps"],
                          "mach": s_own["mach"], "fuel_lbs": s_own["fuel_lbs"],
                          "own_missiles": self.own_n_left, "enemy_missiles": self.enemy_n_left,
                          "own_fired": 0, "enemy_fired": 0,
                          "radar_state": None, "enemy_radar_state": None,
                          "rwr_alarm": False, "rwr_bearing": 0.0,
                          "maws_alarm": False, "maws_tta": 0.0, "ecm_on": False,
                          "in_zone": False,
                          "r_max_own": self.r_max_own, "r_min_own": self.r_min_own,
                          "r_nez_own": self.r_nez_own}
        # 实时显示跨 episode 复用：重置轨迹与累计奖励（避免上一局残留）
        if self._display is not None:
            self._display.reset()
        return obs, self.last_info

    def step(self, action):
        flight = np.asarray(action["flight"], dtype=float)
        flight[0] = np.clip(flight[0], 0.0, 1.0)
        flight[1:] = np.clip(flight[1:], -1.0, 1.0)
        weapon = np.asarray(action["weapon"], dtype=int)

        # ---- 离散武器动作（边沿触发）----
        if weapon[2] == 1:
            self.ecm_on = not self.ecm_on
        if weapon[1] == 1:
            self.own_selected = 1 - self.own_selected   # 切换选中武器槽（接口预留）
        fired_own = False
        if weapon[0] == 1:
            fired_own = self._try_fire_own()

        # ---- 敌方脚本策略：追击 + 发射 ----
        self._enemy_fired = False
        self._enemy_ai()

        # ---- 飞行器推进（JSBSim 内环 dt=inner_dt × frames_per_step 帧）----
        # 推进前快照（用于导弹目标轨迹在本决策步内的连续插值）
        s_own_pre = self.own.state_dict()
        s_enemy_pre = self.enemy.state_dict()
        throttle, roll, pitch, yaw = flight
        self.own.step_manual(throttle, roll, pitch, yaw, self.frames_per_step)
        self.enemy.step_ap(self._enemy_roll_cmd, self._enemy_alt_cmd_ft,
                           self.cfg["enemy_speed_fps"], self.frames_per_step)

        # ---- 导弹推进（missile_dt × missile_steps 亚步）与命中判定 ----
        hit_enemy = self._advance_own_missiles(s_enemy_pre)
        hit_own = self._advance_enemy_missiles(s_own_pre)
        self.last_events = {"fired_own": bool(fired_own),
                            "fired_enemy": bool(self._enemy_fired),
                            "hit_enemy": bool(hit_enemy),
                            "hit_own": bool(hit_own)}

        # ---- 雷达/RWR/MAWS 更新 ----
        s_own = self.own.state_dict()
        s_enemy = self.enemy.state_dict()
        enemy_abs = self._enemy_abs_pos(s_enemy)
        own_abs = s_own["pos_enu_m"]
        self.enemy_radar.r_max_ref = self.enemy_radar.R_MAX_REF * (
            self.cfg["ecm_radar_range_factor"] if self.ecm_on else 1.0)
        # 航向约定转换：JSBSim ψ=0 为正北、顺时针为正；2.2 雷达模块的航向
        # 约定 0=正东、逆时针为正（其内部旋转矩阵），故传入 90°-ψ。
        own_hdg_radar = 90.0 - np.rad2deg(s_own["psi_rad"])
        enemy_hdg_radar = 90.0 - np.rad2deg(s_enemy["psi_rad"])
        self.radar.update([{"id": "enemy", "pos": enemy_abs, "rcs": self.cfg["rcs_enemy"]}],
                          own_abs, own_hdg_radar)
        self.enemy_radar.update([{"id": "own", "pos": own_abs, "rcs": self.cfg["rcs_own"]}],
                                enemy_abs, enemy_hdg_radar)
        enemy_locking = self.enemy_radar.state("own") == FireControlRadar.TRACK
        rwr_info = self.rwr.update(
            [{"id": "enemy_radar", "pos": enemy_abs,
              "mode": "lock" if enemy_locking else "search",
              "lock_target": "self" if enemy_locking else None}],
            own_abs, own_hdg_radar)
        maws_info = self.maws.update(
            [{"id": f"em{i}", "pos": m.pos.copy(), "vel": m.vel.copy()}
             for i, m in enumerate(self.enemy_missiles)],
            own_abs, s_own["v_enu_mps"])

        self.steps += 1
        dist = float(np.linalg.norm(enemy_abs - own_abs))

        # ---- 奖励 ----
        reward = 0.0
        if hit_enemy:
            reward += self.cfg["reward_hit"]
        if hit_own:
            reward += self.cfg["reward_hit_by"]
        if self._in_own_zone(dist):
            reward += self.cfg["reward_in_zone"]
        violation = self._safety_violation(s_own)
        if violation:
            reward += self.cfg["reward_violation"]
        if self._prev_dist is not None:
            reward += np.clip(self.cfg["shaping_scale"] * (self._prev_dist - dist) / 1000.0,
                              -0.5, 0.5)
        self._prev_dist = dist
        self._prev_violation = violation

        # ---- 终止判定 ----
        terminated, reason = False, None
        if hit_enemy:
            terminated, reason = True, "enemy_hit"
        elif hit_own:
            terminated, reason = True, "own_hit"
        elif s_own["h_agl_m"] < self.cfg["crash_agl_m"]:
            terminated, reason = True, "own_crash"
        elif s_enemy["h_agl_m"] < self.cfg["crash_agl_m"]:
            terminated, reason = True, "enemy_crash"
        elif s_own["fuel_lbs"] < self.cfg["fuel_min_lbs"]:
            terminated, reason = True, "fuel_out"
            reward += self.cfg["reward_fuel_out"]
        elif float(np.hypot(own_abs[0], own_abs[1])) > self.cfg["escape_radius_m"]:
            terminated, reason = True, "escape"
        truncated = self.steps >= self.cfg["max_steps"]
        if truncated and not terminated:
            reason = "timeout"

        obs = self._build_obs()
        self.last_info = {
            "steps": self.steps, "dist_m": dist, "reward": float(reward),
            "terminated_reason": reason,
            "h_sl_m": s_own["h_sl_m"], "vtrue_mps": s_own["vtrue_mps"],
            "mach": s_own["mach"], "fuel_lbs": s_own["fuel_lbs"],
            "own_missiles": self.own_n_left, "enemy_missiles": self.enemy_n_left,
            "own_fired": self.cfg["n_missiles"] - self.own_n_left,
            "enemy_fired": self.cfg["n_missiles"] - self.enemy_n_left,
            "radar_state": self.radar.state("enemy"),
            "enemy_radar_state": self.enemy_radar.state("own"),
            "rwr_alarm": rwr_info["alarm"],
            "rwr_bearing": (np.deg2rad(rwr_info["threats"][0]["bearing_deg"])
                            if rwr_info["threats"] else 0.0),
            "maws_alarm": maws_info["alarm"],
            "maws_tta": maws_info["warnings"][0]["tta"] if maws_info["warnings"] else 0.0,
            "ecm_on": self.ecm_on,
            "in_zone": self._in_own_zone(dist),
            "r_max_own": self.r_max_own, "r_min_own": self.r_min_own,
            "r_nez_own": self.r_nez_own,
        }
        return obs, float(reward), terminated, truncated, self.last_info

    # ------------------------------------------------------------------
    # 导弹
    # ------------------------------------------------------------------
    def _try_fire_own(self):
        """本机发射判定：弹药余量 + 雷达跟踪 + 攻击区（reset 时解算的静态包线）。"""
        if self.own_n_left <= 0:
            return False
        s_own = self.own.state_dict()
        s_enemy = self.enemy.state_dict()
        enemy_abs = self._enemy_abs_pos(s_enemy)
        dist = float(np.linalg.norm(enemy_abs - s_own["pos_enu_m"]))
        if self.radar.state("enemy") != FireControlRadar.TRACK:
            return False
        if self.r_min_own is None or self.r_max_own is None:
            return False
        if not (self.r_min_own <= dist <= self.r_max_own):
            return False
        self._launch_missile(self.own_missiles, s_own["pos_enu_m"], s_own["v_enu_mps"],
                             enemy_abs, s_enemy["v_enu_mps"])
        self.own_n_left -= 1
        return True

    def _enemy_fire(self, dist):
        """敌方发射判定（脚本策略）：雷达跟踪 + 攻击区内 + 发射间隔约束。"""
        if self.enemy_n_left <= 0:
            return
        t = self.steps * DECISION_DT
        if t < self._enemy_next_fire_t:
            return
        if self.enemy_radar.state("own") != FireControlRadar.TRACK:
            return
        if not (self.r_min_enemy is not None and self.r_max_enemy is not None
                and self.r_min_enemy <= dist <= 0.95 * self.r_max_enemy):
            return
        s_enemy = self.enemy.state_dict()
        s_own = self.own.state_dict()
        self._launch_missile(self.enemy_missiles, self._enemy_abs_pos(s_enemy),
                             s_enemy["v_enu_mps"], s_own["pos_enu_m"],
                             s_own["v_enu_mps"])
        self.enemy_n_left -= 1
        self._enemy_fired = True
        self._enemy_next_fire_t = t + self.cfg["enemy_fire_interval_s"] \
            + float(self.rng.uniform(0.0, 4.0))

    @staticmethod
    def _launch_missile(pool, shooter_pos, shooter_vel, target_pos, target_vel):
        """以发射机状态为初始条件创建导弹（初速 = 发射机速度 + 30 m/s 沿航向）。

        注意：shooter_pos 与 target_pos 必须同为绝对 ENU 坐标
        （敌机位置需含初始地理偏移 _enemy_offset）。
        """
        v = np.asarray(shooter_vel, dtype=float)
        vn = float(np.linalg.norm(v))
        u = v / max(vn, 1e-6)
        pos0 = np.asarray(shooter_pos, dtype=float) + 20.0 * u
        pool.append(Missile(pos=pos0, vel=v + 30.0 * u))

    def _advance_own_missiles(self, s_enemy_pre):
        """推进本机导弹（向量化亚步 + 目标轨迹连续插值），返回本步是否命中敌机。

        目标位置/速度在本决策步内由 P(t)→P(t+1) 线性插值，避免"目标每秒跳变"
        导致的末端脱靶/绕圈（与 2.2 攻击区解算器的连续目标运动一致）。
        """
        s_enemy_post = self.enemy.state_dict()
        p0 = self._enemy_abs_pos(s_enemy_pre)
        p1 = self._enemy_abs_pos(s_enemy_post)
        v0 = s_enemy_pre["v_enu_mps"]
        v1 = s_enemy_post["v_enu_mps"]
        for k in range(self.missile_steps):
            alpha = (k + 1) / self.missile_steps
            _missile_batch_step(self.own_missiles, p0 * (1.0 - alpha) + p1 * alpha,
                                v0 * (1.0 - alpha) + v1 * alpha, self.missile_dt)
        hit = any(m.hit for m in self.own_missiles)
        self.own_missiles = [m for m in self.own_missiles if m.alive]
        return hit

    def _advance_enemy_missiles(self, s_own_pre):
        """推进敌方导弹（向量化亚步 + 目标轨迹连续插值），返回本步是否命中本机。"""
        s_own_post = self.own.state_dict()
        p0 = s_own_pre["pos_enu_m"]
        p1 = s_own_post["pos_enu_m"]
        v0 = s_own_pre["v_enu_mps"]
        v1 = s_own_post["v_enu_mps"]
        for k in range(self.missile_steps):
            alpha = (k + 1) / self.missile_steps
            _missile_batch_step(self.enemy_missiles, p0 * (1.0 - alpha) + p1 * alpha,
                                v0 * (1.0 - alpha) + v1 * alpha, self.missile_dt)
        hit = any(m.hit for m in self.enemy_missiles)
        self.enemy_missiles = [m for m in self.enemy_missiles if m.alive]
        return hit

    # ------------------------------------------------------------------
    # 敌方脚本策略
    # ------------------------------------------------------------------
    def _enemy_ai(self):
        """敌方脚本策略：以 AP 保持姿态追击本机（航向对准 + 高度跟随），并择机发射。"""
        s_own = self.own.state_dict()
        s_enemy = self.enemy.state_dict()
        enemy_abs = self._enemy_abs_pos(s_enemy)
        rel = s_own["pos_enu_m"] - enemy_abs
        dist = float(np.hypot(rel[0], rel[1]))
        target_psi = float(np.arctan2(rel[0], rel[1]))
        self._enemy_roll_cmd = float(np.clip(2.0 * wrap_pi(target_psi - s_enemy["psi_rad"]),
                                             -0.9, 0.9))
        self._enemy_alt_cmd_ft = float(np.clip(s_own["h_sl_m"] / 0.3048, 8000.0, 20000.0))
        self._enemy_fire(dist)

    # ------------------------------------------------------------------
    # 观测/奖励辅助
    # ------------------------------------------------------------------
    def _enemy_abs_pos(self, s_enemy):
        """敌机绝对 ENU 位置（叠加初始地理偏移）。"""
        return s_enemy["pos_enu_m"] + np.array([self._enemy_offset[0],
                                                self._enemy_offset[1], 0.0])

    def _in_own_zone(self, dist):
        """本机当前是否位于攻击区内（静态包线阈值 + 雷达跟踪）。"""
        if self.r_min_own is None or self.r_max_own is None:
            return False
        if self.radar.state("enemy") != FireControlRadar.TRACK:
            return False
        return self.r_min_own <= dist <= self.r_max_own

    def _safety_violation(self, s_own):
        """安全包线违规判定（D2.1 平台约束 + 失速近似）。"""
        return (s_own["nz_g"] > self.cfg["safety_nz_max_g"]
                or s_own["nz_g"] < self.cfg["safety_nz_min_g"]
                or s_own["mach"] > self.cfg["safety_mach_max"]
                or s_own["h_sl_m"] > self.cfg["safety_alt_max_m"]
                or s_own["vtrue_mps"] < self.cfg["safety_vmin_mps"])

    def _build_obs(self):
        s_own = self.own.state_dict()
        s_enemy = self.enemy.state_dict()
        enemy_abs = self._enemy_abs_pos(s_enemy)
        rel = enemy_abs - s_own["pos_enu_m"]
        dist = float(np.linalg.norm(rel))
        detected = self.radar.state("enemy") == FireControlRadar.TRACK
        rwr_alarm = self.last_info.get("rwr_alarm", False) if self.steps > 0 else False
        rwr_bearing = self.last_info.get("rwr_bearing", 0.0) if self.steps > 0 else 0.0
        maws_alarm = self.last_info.get("maws_alarm", False) if self.steps > 0 else False
        maws_tta = self.last_info.get("maws_tta", 0.0) if self.steps > 0 else 0.0
        tgt = (rel[0], rel[1], rel[2], dist,
               wrap_pi(np.arctan2(rel[0], rel[1]) - s_own["psi_rad"]),
               s_enemy["vtrue_mps"], s_enemy["psi_rad"]) if detected else (0.0,) * 7
        obs = np.array([
            s_own["h_sl_m"], s_own["vtrue_mps"], s_own["mach"],
            s_own["phi_rad"], s_own["theta_rad"], s_own["psi_rad"],
            s_own["alpha_rad"], s_own["beta_rad"], s_own["nz_g"],
            s_own["throttle"], s_own["fuel_frac"], s_own["vd_mps"],
            self.own_n_left, 1.0 if self.ecm_on else 0.0,
            *tgt,
            1.0 if detected else 0.0,
            1.0 if rwr_alarm else 0.0, rwr_bearing,
            1.0 if maws_alarm else 0.0, maws_tta,
            1.0, 0.0,
        ], dtype=np.float32)
        return np.clip(obs, OBS_LOW, OBS_HIGH)

    # ------------------------------------------------------------------
    # 可视化（步骤 2.4 接口统一：get_viz_frame + render）
    # ------------------------------------------------------------------
    def get_viz_frame(self):
        """返回当前态势的可视化快照（纯 Python 标量/list，ENU + SI，可 JSON 序列化）。

        键集稳定，可在 reset 后 / step 后任意时刻调用，供可视化工具（可视化工具/）、
        CSV 记录器与 render() 复用。单位：位置 m、速度 m/s、角度 rad、油量 lb。
        reset 前调用会抛出 RuntimeError（此时 FDM 未 run_ic，快照无意义）。

        返回键：
            steps, t, own, enemy, own_missiles, enemy_missiles, dist_m,
            radar_state, enemy_radar_state, rwr_alarm, rwr_bearing,
            maws_alarm, maws_tta, in_zone,
            r_max_own, r_min_own, r_nez_own, r_max_enemy, r_min_enemy,
            radar_az_limit_deg,
            reward, terminated_reason, events
        own/enemy: pos(3), psi_rad, phi_rad, theta_rad, vtrue_mps, mach, h_sl_m,
                   n_left；own 另含 nz_g, fuel_lbs, throttle, ecm_on。
        own_missiles/enemy_missiles: 在飞导弹 [{pos(3), vel(3), t}]。
        radar_az_limit_deg: 火控雷达方位扫描半角（攻击区扇形绘制用；取自
                   FireControlRadar.AZ_LIMIT，随帧自包含，工具侧无需 import 本库）。
        events: {fired_own, fired_enemy, hit_enemy, hit_own}（本决策步事件）。
        """
        if not self.last_info:
            raise RuntimeError("get_viz_frame() 需先调用 reset()：此时 FDM 未 run_ic，"
                               "快照无意义（距离≈0、无攻击区）")
        s_own = self.own.state_dict()
        s_enemy = self.enemy.state_dict()
        own_abs = s_own["pos_enu_m"]
        enemy_abs = self._enemy_abs_pos(s_enemy)
        dist = float(np.linalg.norm(enemy_abs - own_abs))

        def _ac(s, pos):
            return {"pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "psi_rad": float(s["psi_rad"]),
                    "phi_rad": float(s["phi_rad"]),
                    "theta_rad": float(s["theta_rad"]),
                    "vtrue_mps": float(s["vtrue_mps"]),
                    "mach": float(s["mach"]),
                    "h_sl_m": float(s["h_sl_m"])}

        def _msl(pool):
            return [{"pos": [float(v) for v in m.pos],
                     "vel": [float(v) for v in m.vel],
                     "t": float(m.t)} for m in pool if m.alive]

        def _f(x):
            return float(x) if x is not None else None

        info = self.last_info
        own = _ac(s_own, own_abs)
        own.update({"nz_g": float(s_own["nz_g"]),
                    "fuel_lbs": float(s_own["fuel_lbs"]),
                    "throttle": float(s_own["throttle"]),
                    "n_left": int(self.own_n_left),
                    "ecm_on": bool(self.ecm_on)})
        enemy = _ac(s_enemy, enemy_abs)
        enemy.update({"n_left": int(self.enemy_n_left)})
        return {
            "steps": int(self.steps),
            "t": float(self.steps * DECISION_DT),
            "own": own,
            "enemy": enemy,
            "own_missiles": _msl(self.own_missiles),
            "enemy_missiles": _msl(self.enemy_missiles),
            "dist_m": dist,
            "radar_state": self.radar.state("enemy"),
            "enemy_radar_state": self.enemy_radar.state("own"),
            "rwr_alarm": bool(info.get("rwr_alarm", False)),
            "rwr_bearing": float(info.get("rwr_bearing", 0.0)),
            "maws_alarm": bool(info.get("maws_alarm", False)),
            "maws_tta": float(info.get("maws_tta", 0.0)),
            "in_zone": bool(self._in_own_zone(dist)),
            "r_max_own": _f(self.r_max_own), "r_min_own": _f(self.r_min_own),
            "r_nez_own": _f(self.r_nez_own),
            "r_max_enemy": _f(self.r_max_enemy), "r_min_enemy": _f(self.r_min_enemy),
            "radar_az_limit_deg": float(FireControlRadar.AZ_LIMIT),
            "reward": float(info.get("reward", 0.0)),
            "terminated_reason": info.get("terminated_reason"),
            "events": dict(self.last_events),
        }

    def render(self):
        """Gym 渲染接口（render_mode 经 config["render_mode"] 指定）。

        - "human"：实时 2D 战术显示（TacticalDisplay 窗口，1 Hz 决策节奏刷新，
          惰性创建；窗口历史轨迹由 TacticalDisplay 内部维护）；
        - "rgb_array"：离屏渲染当前帧（Agg，无需显示设备），返回 HxWx3 uint8；
        - None：返回 None（训练/吞吐路径零开销）。

        绘制实现属于项目级工具 可视化工具/（与本库分离，见其 README.md）；
        首次调用时把该工具加入 sys.path 后惰性导入。
        """
        if self.render_mode is None:
            return None
        _ensure_viz_tool_on_path()
        frame = self.get_viz_frame()
        if self.render_mode == "rgb_array":
            from visualization.offscreen import frame_to_rgb
            return frame_to_rgb(frame)
        if self.render_mode == "human":
            from visualization.tactical_display import TacticalDisplay
            if self._display is None:
                self._display = TacticalDisplay()
            self._display.update(frame)
            return None
        raise ValueError(f"未知 render_mode: {self.render_mode!r}，"
                         f"可选 None/'human'/'rgb_array'")

    def close(self):
        """释放可视化资源（JSBSim 实例无显式释放接口，交由 GC）。"""
        if self._display is not None:
            self._display.close()
            self._display = None


def make_bvr_env(config=None, seed=None):
    """模块顶层环境工厂（SubprocVecEnv 的 spawn 启动需要可 pickle 的顶层函数）。

    返回已 reset 的环境（可立即 step，亦符合 SubprocVecEnv 首次调用约定）。
    """
    env = BVRCombatEnv(config=config)
    if seed is not None:
        env.reset(seed=seed)
    return env


class FlatActionBVRCombatEnv(BVRCombatEnv):
    """SB3 兼容动作包装：动作空间展平为 Box(7)。

    稳定基线（stable-baselines3 2.x）仅支持 Box/Discrete/MultiDiscrete/MultiBinary
    动作空间，不支持 Dict/Tuple 动作空间，故提供本包装用于训练接入：
    [油门 0~1, 滚转/俯仰/偏航 -1~1, 发射/切换/ECM 阈值维 -1~1]，
    后 3 维按 >0 阈值转为离散 0/1。不改变环境物理/奖励/终止与观测。
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(7,), dtype=np.float32)

    def step(self, action):
        a = np.asarray(action, dtype=float)
        return super().step({"flight": a[:4],
                             "weapon": (a[4:7] > 0.0).astype(np.int64)})


def make_flat_bvr_env(config=None, seed=None):
    """FlatActionBVRCombatEnv 的模块顶层工厂（SB3 训练/SubprocVecEnv 用）。"""
    env = FlatActionBVRCombatEnv(config=config)
    if seed is not None:
        env.reset(seed=seed)
    return env
