# -*- coding: utf-8 -*-
"""
common/control_mode.py
======================
ActionV1 连续 4 维到本机控制通道的统一映射（阶段三稳定飞行基线）。

两种模式：
- raw_fcs：保留 BVRCombatEnv 原行为，ActionV1 前 4 维直接驱动 F-104 人工
  舵面通道（fcs/throttle-cmd-pilot、aileron/elevator/rudder-cmd-norm）。
- assisted：本机接通 JSBSim AP（fcs/ap-master-on），前 4 维解释为设定值：
    throttle → 速度设定 180~280 m/s（线性，0→180，0.55→235，1→280）
    roll     → 滚转角设定 ±0.9 rad
    pitch    → 高度设定 = clip(episode 参考高度 + 500·pitch, 3000, 8000) m
    yaw      → 忽略（由 AP 滚转/协调转弯承担）
  通过 JSBSimGymBridge.step_ap 下发，ActionV1 形状/范围/边沿位不变。

训练（FlatTaskMissionEnv）、部署（TaskMissionEnv）与行为树（to_env_action）
共用本模块/action_adapter 的同一映射，避免训练/部署漂移。
"""
from __future__ import annotations

import numpy as np

from common.errors import CFG_INVALID, ConfigError

CONTROL_MODES = ("raw_fcs", "assisted")
DEFAULT_CONTROL_MODE = "assisted"      # 任务级 MissionMixin 默认

ASSISTED_SPEED_MIN_MPS = 180.0
ASSISTED_SPEED_MAX_MPS = 280.0
ASSISTED_ROLL_LIMIT_RAD = 0.90
ASSISTED_ALT_PER_PITCH_M = 500.0
ASSISTED_ALT_MIN_M = 3000.0
ASSISTED_ALT_MAX_M = 8000.0

# 兼容别名（供 action_spec/外部只读引用）
ASSISTED_MAPPING = {
    "throttle": "v_cmd_mps = 180 + 100*throttle",
    "roll": "roll_cmd_rad = 0.90*roll",
    "pitch": "alt_cmd_m = clip(alt_ref_m + 500*pitch, 3000, 8000)",
    "yaw": "ignored (AP coordinated turn)",
}


def validate_control_mode(mode: str) -> str:
    if mode not in CONTROL_MODES:
        raise ConfigError(CFG_INVALID, "非法 control_mode",
                          detail={"control_mode": mode, "valid": list(CONTROL_MODES)})
    return mode


def assisted_setpoints(action, current_alt_m: float, cfg: dict | None = None,
                       alt_ref_m: float | None = None) -> dict:
    """ActionV1/前 4 维 → AP 设定值（SI 与 JSBSim 英制双返回）。

    action 可以是 7 维 ActionV1，也可以是 flight 前 4 维；只读取前 3 维。
    alt_ref_m 为 episode 参考高度（通常 reset 时初始高度）；缺省用当前高度。
    返回 {roll_rad, alt_m, alt_ft, speed_mps, speed_fps}。
    """
    cfg = cfg or {}
    arr = np.asarray(action, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ConfigError(CFG_INVALID, "assisted 动作维度不足",
                          detail={"shape": list(arr.shape)})
    throttle = float(np.clip(arr[0], 0.0, 1.0))
    roll = float(np.clip(arr[1], -1.0, 1.0))
    pitch = float(np.clip(arr[2], -1.0, 1.0))

    vmin = float(cfg.get("assisted_speed_min_mps", ASSISTED_SPEED_MIN_MPS))
    vmax = float(cfg.get("assisted_speed_max_mps", ASSISTED_SPEED_MAX_MPS))
    roll_limit = float(cfg.get("assisted_roll_limit_rad", ASSISTED_ROLL_LIMIT_RAD))
    alt_per_pitch = float(cfg.get("assisted_alt_per_pitch_m", ASSISTED_ALT_PER_PITCH_M))
    alt_min = float(cfg.get("assisted_alt_min_m", ASSISTED_ALT_MIN_M))
    alt_max = float(cfg.get("assisted_alt_max_m", ASSISTED_ALT_MAX_M))

    speed_mps = float(np.clip(vmin + throttle * (vmax - vmin), vmin, vmax))
    roll_rad = float(np.clip(roll * roll_limit, -roll_limit, roll_limit))
    base_alt = float(current_alt_m if alt_ref_m is None else alt_ref_m)
    alt_m = float(np.clip(base_alt + alt_per_pitch * pitch, alt_min, alt_max))
    return {
        "roll_rad": roll_rad,
        "alt_m": alt_m,
        "alt_ft": alt_m / 0.3048,
        "speed_mps": speed_mps,
        "speed_fps": speed_mps / 0.3048,
    }


__all__ = ["CONTROL_MODES", "DEFAULT_CONTROL_MODE", "ASSISTED_MAPPING",
           "ASSISTED_SPEED_MIN_MPS", "ASSISTED_SPEED_MAX_MPS",
           "ASSISTED_ROLL_LIMIT_RAD", "ASSISTED_ALT_PER_PITCH_M",
           "ASSISTED_ALT_MIN_M", "ASSISTED_ALT_MAX_M",
           "validate_control_mode", "assisted_setpoints"]
