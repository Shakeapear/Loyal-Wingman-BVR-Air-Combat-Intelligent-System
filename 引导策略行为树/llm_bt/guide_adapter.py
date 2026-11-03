# -*- coding: utf-8 -*-
"""行为树 Dict 动作 → ActionV1 的统一适配层。

- RawFCSAdapter：原始 F-104 舵面/油门通道（行为树原生模式）
- AssistedAdapter：assisted 设定值语义（速度/坡度/高度），要求行为树
  `Blackboard.control_mode == "assisted"`，避免 raw_fcs 动作误入 assisted 环境。

适配层只做格式/范围转换，不改变行为树战术逻辑；控制模式语义由
`executor` 的指令回路根据 `Blackboard.control_mode` 生成。
"""
from __future__ import annotations

import numpy as np

from .executor import Blackboard

ACTION_LOW = np.array([0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                      dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       dtype=np.float32)


def action_dict_to_v1(action: dict, *, control_mode: str) -> np.ndarray:
    """把执行器 Dict 动作转成 ActionV1 (7,)。"""
    flight = np.asarray(action.get("flight", [0.55, 0.0, 0.0, 0.0]),
                        dtype=np.float32).reshape(4)
    weapon = np.asarray(action.get("weapon", [0, 0, 0]),
                        dtype=np.float32).reshape(3)
    out = np.concatenate([flight, weapon], axis=0)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    out = np.clip(out, ACTION_LOW, ACTION_HIGH)
    out[4:7] = (out[4:7] > 0.5).astype(np.float32)
    if control_mode not in ("assisted", "raw_fcs"):
        raise ValueError(f"非法 control_mode: {control_mode}")
    return out.astype(np.float32, copy=False)


class RawFCSAdapter:
    control_mode = "raw_fcs"

    def to_action_v1(self, action: dict, bb: Blackboard | None = None) -> np.ndarray:
        if bb is not None and getattr(bb, "control_mode", self.control_mode) != self.control_mode:
            raise ValueError(
                "RawFCSAdapter 不能用于 assisted 环境；请使用 AssistedAdapter")
        return action_dict_to_v1(action, control_mode=self.control_mode)


class AssistedAdapter:
    control_mode = "assisted"

    def to_action_v1(self, action: dict, bb: Blackboard | None = None) -> np.ndarray:
        if bb is not None and getattr(bb, "control_mode", self.control_mode) != self.control_mode:
            raise ValueError(
                "AssistedAdapter 需要 Blackboard.control_mode='assisted'")
        return action_dict_to_v1(action, control_mode=self.control_mode)


def adapter_for(control_mode: str):
    if control_mode == "raw_fcs":
        return RawFCSAdapter()
    if control_mode == "assisted":
        return AssistedAdapter()
    raise ValueError(f"非法 control_mode: {control_mode}")


__all__ = ["ACTION_LOW", "ACTION_HIGH", "action_dict_to_v1",
           "RawFCSAdapter", "AssistedAdapter", "adapter_for"]
