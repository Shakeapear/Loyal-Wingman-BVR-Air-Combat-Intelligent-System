# -*- coding: utf-8 -*-
"""
agents/policy_interface.py
==========================
阶段三 JSRL 底座：ActionV1 / ActionMaskV1 / PolicyProtocol / PolicyOutput。

设计依据：`多任务智能体/统一输出接口与DRL网络设计.md` §3（唯一设计依据），
并兼容根目录 `接口规范.md` v1.0（版本常量、异常类型、数据校验）。

约定
----
- ActionV1：np.ndarray, shape=(7,), dtype=float32
  [throttle, roll, pitch, yaw, fire, switch, ecm]
  throttle∈[0,1]，roll/pitch/yaw∈[-1,1]，fire/switch/ecm∈{0,1}（边沿触发）。
- ActionMaskV1：np.ndarray, shape=(7,), dtype=bool，True=允许。
- 所有任务级专家智能体（strike/patrol/defense/ew）统一通过
  PolicyProtocol.get_action 推理；技能级智能体走 action_adapter.command_v1 兼容通道，
  不进入本协议主通道。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np

from common.errors import ARG_INVALID, ValidationError

ACTION_V1_VERSION = "action_v1"
ACTION_V1_DIM = 7
ACTION_V1_FIELDS = ("throttle", "roll", "pitch", "yaw", "fire", "switch", "ecm")
ACTION_V1_EDGE_DIMS = (4, 5, 6)

ACTION_V1_LOW = np.array([0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_V1_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# 无模型/异常时的安全兜底动作：巡航油门 + 平飞 + 不发射
SAFE_ACTION_V1 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# 动作类型别名（设计文档中的 ActionV1 即 7 维 float32 向量）
ActionV1: TypeAlias = np.ndarray
ActionMaskV1: TypeAlias = np.ndarray


@dataclass
class PolicyOutput:
    """一次推理的完整结果（动作 + 学习/追溯元数据）。"""

    action: np.ndarray                 # (7,) float32，ActionV1
    source: str = "drl"                # "drl" | "guide" | "fallback" | "safety"
    log_prob: float | None = None
    value: float | None = None
    hidden_state: Any = None
    masked: np.ndarray | None = None   # (7,) bool，本步 ActionMaskV1
    action_version: str = ACTION_V1_VERSION
    task_id: str | None = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.action = np.asarray(self.action, dtype=np.float32).reshape(ACTION_V1_DIM)
        if self.masked is not None:
            self.masked = np.asarray(self.masked, dtype=bool).reshape(ACTION_V1_DIM)

    def to_dict(self) -> dict:
        """JSON 可序列化（行为树黑板/CSV 追溯用）。"""
        return {
            "action": self.action.tolist(),
            "source": self.source,
            "log_prob": self.log_prob,
            "value": self.value,
            "masked": self.masked.tolist() if self.masked is not None else None,
            "action_version": self.action_version,
            "task_id": self.task_id,
            "info": self.info,
        }


@runtime_checkable
class PolicyProtocol(Protocol):
    """所有任务级智能体（含 JSRL 探索策略）的统一推理入口。"""

    task_id: str

    def action_spec(self) -> dict: ...

    def reset(self) -> None: ...

    def get_action(self, obs, *, info=None, deterministic=True, mask=None,
                   guide_action=None, guide_steps=0) -> PolicyOutput: ...


def action_spec(task_id: str) -> dict:
    """ActionV1 的机器可读声明（随模型归档，接口规范 §4）。"""
    return {
        "version": ACTION_V1_VERSION,
        "task_id": str(task_id),
        "dim": ACTION_V1_DIM,
        "dtype": "float32",
        "fields": list(ACTION_V1_FIELDS),
        "low": ACTION_V1_LOW.tolist(),
        "high": ACTION_V1_HIGH.tolist(),
        "edge_dims": list(ACTION_V1_EDGE_DIMS),
        "frame": "ENU/SI",
        "decision_hz": 1.0,
        "mask": {
            "version": "action_mask_v1",
            "dim": ACTION_V1_DIM,
            "fire": "n_missiles>0 and radar_state==TRACK and r_min<=dist<=r_max",
            "switch": "reserved (always False)",
            "ecm": "always True",
        },
    }


def command_v1_spec() -> dict:
    """技能级兼容通道声明（3 维自驾仪设定点，非阶段三训练主线）。"""
    return {
        "version": "command_v1",
        "dim": 3,
        "fields": ["v_cmd_mps", "roll_cmd_rad", "alt_cmd_m"],
        "low": [180.0, -0.90, 3000.0],
        "high": [280.0, 0.90, 8000.0],
        "decision_hz": 1.0,
    }


def validate_action_v1(action, *, name: str = "action") -> np.ndarray:
    """校验并返回 float32 (7,)；不合法抛 ValidationError(code=ARG_INVALID)。"""
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape != (ACTION_V1_DIM,):
        raise ValidationError(ARG_INVALID, f"{name} 形状非法", detail={"shape": list(arr.shape)})
    if not np.all(np.isfinite(arr)):
        raise ValidationError(ARG_INVALID, f"{name} 含非有限值", detail={"values": arr.tolist()})
    if np.any(arr < ACTION_V1_LOW - 1e-6) or np.any(arr > ACTION_V1_HIGH + 1e-6):
        raise ValidationError(ARG_INVALID, f"{name} 超出 ActionV1 范围",
                              detail={"values": arr.tolist()})
    return arr


def clip_action_v1(action) -> np.ndarray:
    """裁剪到 ActionV1 合法域；武器维按 >0.5 二值化（边沿语义）。"""
    arr = np.asarray(action, dtype=np.float32).reshape(ACTION_V1_DIM)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, ACTION_V1_LOW, ACTION_V1_HIGH)
    arr[4:7] = (arr[4:7] > 0.5).astype(np.float32)
    return arr.astype(np.float32, copy=False)


def apply_action_mask(action, mask) -> np.ndarray:
    """把掩码为 False 的维清零（连续维/武器维通用，主要用于武器边沿位）。"""
    arr = clip_action_v1(action)
    m = np.asarray(mask, dtype=bool).reshape(ACTION_V1_DIM)
    arr[~m] = 0.0
    return arr


def _as_float(info: dict, key: str, default=None):
    value = info.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_action_mask(obs, info: dict | None = None, env_cfg: dict | None = None) -> np.ndarray:
    """ActionMaskV1：与 BVRCombatEnv._try_fire_own 的发射条件严格一致。

    0~3 恒 True；4 fire 需 弹量>0 ∧ 雷达 TRACK ∧ r_min≤dist≤r_max；
    5 switch 当前恒 False（单弹种，接口预留）；6 ecm 恒 True。
    信息缺失时保守禁用 fire（返回 False），避免训练/评估口径漂移。
    """
    info = dict(info or {})
    obs_arr = None
    try:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        obs_arr = None

    mask = np.ones(ACTION_V1_DIM, dtype=bool)
    mask[5] = False
    mask[6] = True

    n_left = _as_float(info, "own_missiles_left")
    if n_left is None and obs_arr is not None and obs_arr.size > 12:
        n_left = float(obs_arr[12])
    radar_state = info.get("radar_state")
    dist = _as_float(info, "dist_m")
    if dist is None and obs_arr is not None and obs_arr.size > 21 and obs_arr[21] > 0.5:
        dist = float(obs_arr[17])
    r_min = _as_float(info, "r_min_own_m")
    r_max = _as_float(info, "r_max_own_m")
    if env_cfg:
        r_min = r_min if r_min is not None else _as_float(env_cfg, "r_min_own_m")
        r_max = r_max if r_max is not None else _as_float(env_cfg, "r_max_own_m")

    tracked = isinstance(radar_state, str) and radar_state.upper() == "TRACK"
    mask[4] = bool((n_left or 0.0) > 0.0 and tracked and dist is not None
                   and r_min is not None and r_max is not None
                   and r_min <= dist <= r_max)
    return mask
