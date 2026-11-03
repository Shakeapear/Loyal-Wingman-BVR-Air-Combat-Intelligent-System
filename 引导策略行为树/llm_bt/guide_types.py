# -*- coding: utf-8 -*-
"""统一引导步契约（JSRL teacher/student 交接点）。

`GuideStep` 是行为树/人类招式/DRL 之间唯一需要共享的引导返回结构：

    action       ActionV1 (7,) 或 None（未激活时）
    active       本步是否由教师接管
    source       "bt" | "human" | "drl" | "fallback"
    guide_step   本 episode 已引导步数
    control_mode "assisted" | "raw_fcs"（必须与目标环境一致）
    mask         ActionMaskV1 (7,) 或 None
    meta         树 id、边界触发、错误等可追溯信息

统一该结构后，训练循环不再需要猜测“这一步是谁的动作、是否有效、
是否与策略同一控制模式”。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ACTION_DIM = 7
CONTROL_MODES = ("assisted", "raw_fcs")
SOURCES = ("bt", "human", "drl", "fallback")


def _vector(value, *, name: str, dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.shape != (ACTION_DIM,):
        raise ValueError(f"{name} 形状必须为 ({ACTION_DIM},)，实际 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限值")
    return arr


@dataclass
class GuideStep:
    action: np.ndarray | None
    active: bool
    source: str = "bt"
    guide_step: int = 0
    control_mode: str = "assisted"
    mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.active = bool(self.active)
        self.source = str(self.source)
        self.control_mode = str(self.control_mode)
        self.guide_step = int(self.guide_step)
        if self.source not in SOURCES:
            raise ValueError(f"source 必须为 {SOURCES}，实际 {self.source}")
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(
                f"control_mode 必须为 {CONTROL_MODES}，实际 {self.control_mode}")
        if self.action is not None:
            self.action = _vector(self.action, name="GuideStep.action",
                                  dtype=np.float32)
            low = np.array([0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                           dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            dtype=np.float32)
            if np.any(self.action < low) or np.any(self.action > high):
                raise ValueError("GuideStep.action 超出 ActionV1 范围")
        elif self.active:
            raise ValueError("active=True 时 action 不能为 None")
        if self.mask is not None:
            self.mask = _vector(self.mask, name="GuideStep.mask",
                                dtype=bool)
        if not isinstance(self.meta, dict):
            raise TypeError("GuideStep.meta 必须为 dict")

    def to_dict(self) -> dict:
        return {
            "action": None if self.action is None else self.action.tolist(),
            "active": self.active,
            "source": self.source,
            "guide_step": self.guide_step,
            "control_mode": self.control_mode,
            "mask": None if self.mask is None else self.mask.tolist(),
            "meta": dict(self.meta),
        }


__all__ = ["ACTION_DIM", "CONTROL_MODES", "SOURCES", "GuideStep"]
