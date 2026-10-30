# -*- coding: utf-8 -*-
"""
agents/action_adapter.py
========================
ActionV1 ↔ 现有环境的纯逻辑适配层（设计文档 §3.5）。

- to_env_action：ActionV1 → BVRCombatEnv(Dict) 动作，用于行为树/部署主通道；
- to_flat_action：ActionV1 → FlatActionBVRCombatEnv(Box(7))，用于训练通道；
- to_ap_command：ActionV1 → JSBSimFlightEnv(Box(3)) / 本机 AP 设定点；
- to_assisted_setpoints：assisted 模式连续 4 维 → AP 设定值（与 BVRCombatEnv 同源）；
- coerce_action_v1：PolicyOutput / Dict / 序列统一收敛为 ActionV1。

本模块不创建环境、不依赖 SB3，便于单元测试与行为树复用。
"""
from __future__ import annotations

import numpy as np

from agents.policy_interface import (ACTION_V1_DIM, PolicyOutput, clip_action_v1,
                                     validate_action_v1)
from common.control_mode import assisted_setpoints
from common.errors import ARG_INVALID, ValidationError

COMMAND_V1_VERSION = "command_v1"
COMMAND_V1_LOW = np.array([180.0, -0.90, 3000.0], dtype=np.float32)
COMMAND_V1_HIGH = np.array([280.0, 0.90, 8000.0], dtype=np.float32)


def coerce_action_v1(action) -> np.ndarray:
    """把 PolicyOutput / 环境 Dict / 序列统一转成合法 ActionV1（裁剪+二值化）。"""
    if isinstance(action, PolicyOutput):
        action = action.action
    if isinstance(action, dict):
        flight = np.asarray(action.get("flight", []), dtype=np.float32).reshape(-1)
        weapon = np.asarray(action.get("weapon", []), dtype=np.float32).reshape(-1)
        if flight.size != 4 or weapon.size != 3:
            raise ValidationError(ARG_INVALID, "环境 Dict 动作形状非法",
                                  detail={"flight": flight.tolist(), "weapon": weapon.tolist()})
        action = np.concatenate([flight, weapon])
    try:
        arr = np.asarray(action, dtype=np.float32).reshape(ACTION_V1_DIM)
    except (TypeError, ValueError) as exc:
        raise ValidationError(ARG_INVALID, "动作无法转换为 ActionV1",
                              detail={"type": type(action).__name__}) from exc
    return clip_action_v1(arr)


def to_env_action(action) -> dict:
    """ActionV1 → Dict{"flight": (4,) float32, "weapon": (3,) int64}。"""
    a7 = coerce_action_v1(action)
    return {
        "flight": a7[:4].astype(np.float32, copy=True),
        "weapon": (a7[4:7] > 0.5).astype(np.int64),
    }


def to_flat_action(action) -> np.ndarray:
    """ActionV1 → FlatActionBVRCombatEnv 的 Box(7) 动作（恒等，裁剪后）。"""
    return coerce_action_v1(action)


def from_env_action(action) -> np.ndarray:
    """Dict/扁平环境动作 → ActionV1（行为树 → JSRL 引导动作）。"""
    return coerce_action_v1(action)


def to_assisted_setpoints(action, current_alt_m: float, cfg: dict | None = None) -> dict:
    """assisted 模式连续 4 维 → AP 设定值（与 BVRCombatEnv 使用同一函数）。"""
    return assisted_setpoints(coerce_action_v1(action), current_alt_m, cfg)


def to_ap_command(action, obs=None) -> np.ndarray:
    """ActionV1 → 3 维 AP 设定点 [v_cmd_mps, roll_cmd_rad, alt_cmd_m]。

    这是 command_v1/assisted 共用映射，固定速度映射（不再使用当前速度反馈）：
    - throttle → 180 + 100*throttle m/s；
    - roll → 0.90*roll rad；
    - pitch → clip(当前高度 + 50*pitch, 3000, 8000) m；
    - yaw 忽略（F-104 AP 无偏航设定点，由滚转协调转弯）。
    """
    a7 = coerce_action_v1(action)
    alt_m = 4572.0
    if obs is not None:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        if obs_arr.size >= 1:
            alt_m = float(obs_arr[0])
    sp = assisted_setpoints(a7, alt_m)
    cmd = np.array([sp["speed_mps"], sp["roll_rad"], sp["alt_m"]],
                   dtype=np.float32)
    return np.clip(cmd, COMMAND_V1_LOW, COMMAND_V1_HIGH).astype(np.float32)


# 语义化别名：技能级兼容通道的显式入口
def command_v1(action, obs=None) -> np.ndarray:
    return to_ap_command(action, obs)


def validate_adapter_action(action, *, name="action") -> np.ndarray:
    """测试/部署辅助：严格校验 ActionV1。"""
    return validate_action_v1(action, name=name)
