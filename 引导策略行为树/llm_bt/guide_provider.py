# -*- coding: utf-8 -*-
"""BTGuideProvider：行为树 → 统一 GuideStep 的引导提供器。

它是行为树与 JSRL/DRL 之间唯一的桥：

    obs/info → BTGuideProvider.step() → GuideStep(action, active, source,
                                                   guide_step, control_mode,
                                                   mask, meta)

设计约束：
- 控制模式必须与目标环境一致（assisted/raw_fcs），不允许隐式混用；
- 边界约束在行为树动作之后、返回 GuideStep 之前执行；
- 引导步数由 GuideInterface 管理，耗尽后 active=False；
- 不在此层启动训练；JSRL trainer 只消费 GuideStep。
"""
from __future__ import annotations

import copy

import numpy as np

from common.control_mode import validate_control_mode

from .boundary_constraints import BoundaryConstraints
from .executor import Blackboard, build_tree, tick_once
from .guide_adapter import adapter_for
from .guide_interface import GuideInterface
from .guide_types import GuideStep
from .tactical_execution_tree import tactical_tree_json


def _validate_obs(obs):
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size != 28:
        raise ValueError(
            f"BTGuideProvider 需要 ObsV1 28 维，实际 {arr.size} 维；"
            "P1 的 28+7 维请先拆出前 28 维。")
    return arr


class BTGuideProvider:
    """把一个行为树包装成逐帧 GuideStep 提供器。"""

    def __init__(self, tree: dict | None = None, *,
                 control_mode: str = "assisted",
                 max_guide_steps: int = 600,
                 guide_active: bool = True,
                 env=None,
                 boundary_config: dict | None = None):
        self.tree = copy.deepcopy(tree) if tree is not None else tactical_tree_json()
        self.control_mode = validate_control_mode(control_mode)
        self.env = env
        self.env_control_mode = getattr(getattr(env, "unwrapped", env),
                                        "control_mode", None)
        if env is not None and self.env_control_mode is not None \
                and self.env_control_mode != self.control_mode:
            raise ValueError(
                f"引导控制模式与环境不一致：provider={self.control_mode}, "
                f"env={self.env_control_mode}。禁止 raw_fcs BT 直接驱动 assisted 环境。")
        self.guide = GuideInterface(max_guide_steps=max_guide_steps,
                                    guide_active=guide_active)
        self.boundary_config = dict(boundary_config or {})
        self.adapter = adapter_for(self.control_mode)
        self.bb: Blackboard | None = None
        self.root = None
        self.boundary: BoundaryConstraints | None = None
        self._initialized = False
        self._current_step = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self, obs=None, info=None, alt_ref_m: float | None = None):
        self.guide.reset()
        if obs is not None:
            obs = _validate_obs(obs)
        if alt_ref_m is None and obs is not None:
            alt_ref_m = float(obs[0])
        self.bb = Blackboard(self.env, control_mode=self.control_mode,
                             alt_ref_m=alt_ref_m)
        self.root = build_tree(self.tree, self.bb)
        self.boundary = BoundaryConstraints(
            config=self.boundary_config,
            env_cfg=getattr(getattr(self.env, "unwrapped", self.env), "cfg", None))
        self._initialized = True
        self._current_step = 0
        return self

    def _ensure_ready(self, obs, info=None):
        if not self._initialized or self.bb is None or self.root is None:
            self.reset(obs, info)
        if self.boundary is None:
            self.boundary = BoundaryConstraints(
                config=self.boundary_config,
                env_cfg=getattr(getattr(self.env, "unwrapped", self.env),
                                "cfg", None))

    # ------------------------------------------------------------------
    # 引导
    # ------------------------------------------------------------------
    def step(self, obs, info: dict | None = None, step: int | None = None,
             mask=None) -> GuideStep:
        obs = _validate_obs(obs)
        info = dict(info or {})
        self._ensure_ready(obs, info)
        if step is None:
            step = self._current_step

        def _policy(_obs):
            action_dict = tick_once(self.root, self.bb, obs, info, step,
                                    boundary=self.boundary)
            return self.adapter.to_action_v1(action_dict, self.bb)

        action, active = self.guide.get_action(obs, _policy)
        self._current_step = int(step) + 1
        meta = {
            "tree_id": str(self.tree.get("bt_id", "unknown")),
            "tree_mission": str(self.tree.get("mission", "")),
            "boundary_triggers": int(
                sum(self.boundary.counts.values())
                if self.boundary is not None else 0),
        }
        if not active:
            return GuideStep(action=None, active=False, source="bt",
                             guide_step=self.guide.current_guide_step,
                             control_mode=self.control_mode, mask=mask,
                             meta=meta)
        return GuideStep(action=action, active=True, source="bt",
                         guide_step=self.guide.current_guide_step,
                         control_mode=self.control_mode, mask=mask, meta=meta)

    # ------------------------------------------------------------------
    # 课程/开关代理
    # ------------------------------------------------------------------
    @property
    def current_guide_step(self) -> int:
        return int(self.guide.current_guide_step)

    @property
    def exhausted(self) -> bool:
        return bool(self.guide.exhausted)

    def update_guide_steps(self, new_max_steps: int):
        self.guide.update_guide_steps(new_max_steps)
        return self

    def set_active(self, active: bool):
        self.guide.set_active(active)
        return self


__all__ = ["BTGuideProvider"]
