# -*- coding: utf-8 -*-
"""
common/jsbsim_bridge.py（交付物 D2.3-2）
======================================
JSBSimGymBridge：单机 JSBSim 生命周期管理（init/load/reset/step/close），
供 JSBSimFlightEnv（5 个技能任务）与 BVRCombatEnv（1v1 超视距交战）复用。

已知坑位（jsbsim 1.3.1 Python API，务必遵守）：
- `jsbsim.FGFDMExec(str(root), None)`；
- `fdm.set_debug_level(0)`（否则每帧刷屏）；
- `fdm.load_ic("reset_cruise", True)` 必须两个参数；
- 坐标系：JSBSim 输出 NED+英制，本模块统一转换为 ENU+SI（见 ned_ft_to_enu_m）；
  ENU = [东, 北, 上]。位置由经纬度差值换算（lat/lon → 本地米），
  **不要用 position/distance-from-start-lat/lon-mt**：该属性是无符号距离，
  向南/向西飞行时符号丢失（实测验证）；速度 [东, 北, 上] =
  [v-east-fps, v-north-fps, -v-down-fps]·0.3048（带符号，正确）；
- 属性读取用 `fdm["name"]`（约 0.3 μs/次，比 get_property_value 快 3~4 倍），
  写入用 `fdm.set_property_value(name, value)`（约 1 μs/次）。

两种控制模式（互斥）：
- 人工通道（BVRCombatEnv 玩家动作）：AP 主开关关闭，
  直接驱动 fcs/throttle-cmd-pilot、fcs/aileron-cmd-norm、fcs/elevator-cmd-norm、
  fcs/rudder-cmd-norm（F-104 FCS 通道，见 aircraft/f104/Systems/FCS-*.xml）；
- 自动驾驶仪（BVRCombatEnv 敌方脚本策略 / JSBSimFlightEnv 5 任务）：
  fcs/ap-master-on=1 + ap-roll/ap-alt/ap-speed 设定值
  （aircraft/f104/Systems/autopilot-hold.xml，本项目已调参验证）。

标准采集配置（reset 后调用，与《04 数据采集规范》§3 一致）：
燃油 4700 lb（3500+600+600）、发动机置运行、收轮、减速板/襟翼收起。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import jsbsim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSB_ROOT = PROJECT_ROOT / "JSBSim"

SIM_DT = 1.0 / 120.0          # JSBSim 内环积分步长（秒）
G0 = 9.80665


def ned_ft_to_enu_m(pos_e_m, pos_n_m, h_ft, v_e_fps, v_n_fps, v_d_fps):
    """JSBSim NED/英制 → ENU/SI 坐标与速度转换。

    输入：东向距离 m、北向距离 m、海拔 ft、速度三分量 [东, 北, 下] fps。
    输出：(pos, vel)，pos=[东, 北, 上] m，vel=[东, 北, 上] m/s。
    """
    pos = np.array([pos_e_m, pos_n_m, h_ft * 0.3048], dtype=float)
    vel = np.array([v_e_fps, v_n_fps, -v_d_fps], dtype=float) * 0.3048
    return pos, vel


def wrap_pi(a):
    """角度归一化到 [-π, π)。"""
    return float((np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi)


def _sound_speed_fps(h_ft):
    """ISA 声速 ft/s（用于马赫数 → 真空速换算）。"""
    h_m = h_ft * 0.3048
    t_k = 288.15 - 0.0065 * h_m
    return np.sqrt(1.4 * 287.05 * max(t_k, 150.0)) * 3.28084


class JSBSimGymBridge:
    """单机 JSBSim 桥接层（init/load/reset/step/close + SI 状态字典）。"""

    def __init__(self, root=None, aircraft="f104", sim_dt=SIM_DT):
        self.root = str(root or JSB_ROOT)
        self.aircraft = aircraft
        self.sim_dt = float(sim_dt)
        self.fdm = jsbsim.FGFDMExec(self.root, None)
        try:
            self.fdm.set_debug_level(0)
        except Exception:
            pass
        self.fdm.set_dt(self.sim_dt)
        self.fdm.load_model(aircraft)
        self.load()
        self.rng = np.random.default_rng()
        self._origin_lat = 30.0   # reset_cruise.xml 初始纬度
        self._origin_lon = 120.0

    # ---------- 生命周期 ----------

    def load(self):
        """（重新）加载初始条件文件（reset 内部调用，亦可手动）。"""
        self.fdm.load_ic("reset_cruise", True)

    def reset(self, seed=None, h_ft=None, mach=None, heading_deg=None,
              lat0_deg=30.0, lon0_deg=120.0, autopilot=False,
              setup_standard=True, warmup_frames=60):
        """重置到巡航初始条件（可随机初始高度/速度/航向），返回 rng。

        h_ft/mach/heading_deg 为 None 时随机：高度 8000~20000 ft、M0.7~0.9、
        航向 0~360°。随机量全部由 seed 决定（可复现）。
        autopilot=True 时接通 AP（ap-master-on=1，设定值待调用方设定）；
        warmup_frames：重置后预热帧数（发动机/气动状态建立）。
        """
        self.rng = np.random.default_rng(seed)
        if h_ft is None:
            h_ft = float(self.rng.uniform(8000.0, 20000.0))
        if mach is None:
            mach = float(self.rng.uniform(0.7, 0.9))
        if heading_deg is None:
            heading_deg = float(self.rng.uniform(0.0, 360.0))
        h_ft, mach, heading_deg = float(h_ft), float(mach), float(heading_deg)
        # 记住本次重置的地理原点（用于 state_dict 中 lat/lon → 本地 ENU 换算）
        self._origin_lat = float(lat0_deg)
        self._origin_lon = float(lon0_deg)
        # 每次 reset 重建全新 FDM 实例（load_model 约 20 ms）：
        # jsbsim 1.3.1 在已飞行过的 FDM 上二次 run_ic 时，ic/vt-fps、ic/mach
        # 的空速换算会复用上一次飞行高度的大气状态，导致重置后马赫数错误；
        # 全新实例上任意高度/马赫覆盖均验证正确。
        self.fdm = jsbsim.FGFDMExec(self.root, None)
        try:
            self.fdm.set_debug_level(0)
        except Exception:
            pass
        self.fdm.set_dt(self.sim_dt)
        self.fdm.load_model(self.aircraft)
        self.load()
        self._set("ic/h-sl-ft", h_ft)
        self._set("ic/terrain-elevation-ft", 0.0)
        self._set("ic/lat-gc-deg", float(lat0_deg))
        self._set("ic/long-gc-deg", float(lon0_deg))
        self._set("ic/psi-true-deg", heading_deg)
        self._set("ic/vt-fps", mach * _sound_speed_fps(h_ft))
        self.fdm.run_ic()
        if setup_standard:
            self.setup_standard(autopilot=autopilot)
        if autopilot:
            # AP 接通时预热期间保持当前高度（避免设定值为 0 时的俯仰指令）
            self._set("fcs/ap-alt-setpoint-ft", h_ft)
            self._set("fcs/ap-speed-setpoint-fps", mach * _sound_speed_fps(h_ft))
            self._set("fcs/ap-roll-setpoint-rad", 0.0)
        for _ in range(int(warmup_frames)):
            self.fdm.run()
        return self.rng

    def close(self):
        """释放资源（pyjsbsim 无显式释放接口，实例交由 GC，此处保持接口完整）。"""
        pass

    # ---------- 标准采集配置 ----------

    def setup_standard(self, autopilot=False):
        """标准配置：燃油 4700 lb、发动机运行、收轮、减速板/襟翼收起。

        autopilot=True 时接通 AP 主开关与高度保持通道。
        """
        self._set("propulsion/tank[0]/contents-lbs", 3500)
        self._set("propulsion/tank[1]/contents-lbs", 600)
        self._set("propulsion/tank[2]/contents-lbs", 600)
        self._set("propulsion/engine[0]/set-running", 1)
        self._set("propulsion/cutoff_cmd", 0)
        self._set("propulsion/engine[0]/n2", 100)
        self._set("propulsion/engine[0]/n1", 100)
        self._set("gear/gear-cmd-norm", 0)
        self._set("fcs/speedbrake-cmd-norm", 0)
        self._set("fcs/flap-cmd-norm", 0)
        self._set("fcs/ap-master-on", 1 if autopilot else 0)
        self._set("fcs/ap-alt-hold-on", 1 if autopilot else 0)

    # ---------- 控制推进 ----------

    def step_manual(self, throttle, roll, pitch, yaw, n_frames):
        """人工通道控制（AP 关闭时）：设定舵面后推进 n_frames 帧。

        throttle∈[0,1]→fcs/throttle-cmd-pilot；roll/pitch/yaw∈[-1,1] 分别对应
        fcs/aileron-cmd-norm、fcs/elevator-cmd-norm、fcs/rudder-cmd-norm。
        """
        self._set("fcs/throttle-cmd-pilot", float(np.clip(throttle, 0.0, 1.0)))
        self._set("fcs/aileron-cmd-norm", float(np.clip(roll, -1.0, 1.0)))
        self._set("fcs/elevator-cmd-norm", float(np.clip(pitch, -1.0, 1.0)))
        self._set("fcs/rudder-cmd-norm", float(np.clip(yaw, -1.0, 1.0)))
        for _ in range(int(n_frames)):
            self.fdm.run()

    def step_ap(self, roll_rad, alt_ft, speed_fps, n_frames):
        """自动驾驶仪模式推进：滚转/高度/速度保持设定值 + 积分 n_frames 帧。"""
        self._set("fcs/ap-master-on", 1)
        self._set("fcs/ap-alt-hold-on", 1)
        self._set("fcs/ap-roll-setpoint-rad", float(np.clip(roll_rad, -0.9, 0.9)))
        self._set("fcs/ap-alt-setpoint-ft", float(np.clip(alt_ft, 3000.0, 40000.0)))
        self._set("fcs/ap-speed-setpoint-fps", float(np.clip(speed_fps, 300.0, 1200.0)))
        for _ in range(int(n_frames)):
            self.fdm.run()

    # ---------- 状态读取 ----------

    def state_dict(self):
        """返回 SI 状态字典（ENU 坐标、SI 单位）。

        键：pos_e_m, pos_n_m, h_sl_m, h_agl_m, pos_enu_m(3),
            v_enu_mps(3), vtrue_mps, mach, vd_mps,
            phi_rad, theta_rad, psi_rad, alpha_rad, beta_rad,
            nz_g, throttle, fuel_lbs, fuel_frac
        """
        f = self.fdm
        # 位置：由 lat/lon 差值换算本地 ENU（注意：jsbsim 的
        # position/distance-from-start-lat/lon-mt 是无符号距离，飞行方向会丢失，
        # 不能用；经纬度与速度分量才是带符号的）
        lat = f["position/lat-gc-deg"]
        lon = f["position/long-gc-deg"]
        north_m = (lat - self._origin_lat) * 111320.0
        east_m = (lon - self._origin_lon) * 111320.0 * np.cos(np.deg2rad(self._origin_lat))
        pos_e_m, pos_n_m = float(east_m), float(north_m)
        h_sl_m = f["position/h-sl-ft"] * 0.3048
        v_e = f["velocities/v-east-fps"] * 0.3048
        v_n = f["velocities/v-north-fps"] * 0.3048
        v_d = -f["velocities/v-down-fps"] * 0.3048
        return {
            "pos_e_m": float(pos_e_m),
            "pos_n_m": float(pos_n_m),
            "h_sl_m": float(h_sl_m),
            "h_agl_m": float(f["position/h-agl-ft"] * 0.3048),
            "pos_enu_m": np.array([pos_e_m, pos_n_m, h_sl_m]),
            "v_enu_mps": np.array([v_e, v_n, v_d]),
            "vtrue_mps": float(f["velocities/vtrue-fps"] * 0.3048),
            "mach": float(f["velocities/mach"]),
            "vd_mps": float(v_d),
            "phi_rad": float(f["attitude/phi-rad"]),
            "theta_rad": float(f["attitude/theta-rad"]),
            "psi_rad": float(f["attitude/psi-rad"]),
            "alpha_rad": float(f["aero/alpha-rad"]),
            "beta_rad": float(f["aero/beta-rad"]),
            "nz_g": float(f["accelerations/Nz"]),
            "throttle": float(f["fcs/throttle-cmd-norm"]),
            "fuel_lbs": float(f["propulsion/total-fuel-lbs"]),
            "fuel_frac": float(f["propulsion/total-fuel-lbs"]) / 4700.0,
        }

    # ---------- 底层属性访问 ----------

    def _set(self, name, value):
        self.fdm.set_property_value(name, value)

    def _get(self, name):
        return self.fdm.get_property_value(name)
