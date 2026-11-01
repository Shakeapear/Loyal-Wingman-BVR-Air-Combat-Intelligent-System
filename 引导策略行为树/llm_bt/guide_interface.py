# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/guide_interface.py（步骤 2.6 D2.6-3）
================================================================
JSRL（Jump-Start RL）引导接口：在 episode 前/中期由行为树提供动作，引导
步数耗尽后交还 DRL 策略。

语义：
- `guide_active=True` 且 `current_guide_step < max_guide_steps` → 调用
  `guide_policy(obs)` 返回行为树动作，`is_guided=True`；
- 引导关闭或步数耗尽 → 返回 `(None, False)`，由 DRL 接管；
- `reset()` 每个 episode 调用：步数清零并恢复初始激活状态；
- `update_guide_steps(new_max_steps)` 支持课程式衰减/扩展；
- `guide_policy` 返回 None 视为本步无法引导，返回 `(None, False)` 且不计数。
"""
from __future__ import annotations


class GuideInterface:
    """行为树引导接口（不依赖 LLM，纯计数/分发）。"""

    def __init__(self, max_guide_steps: int = 240, guide_active: bool = True):
        self._initial_active = bool(guide_active)
        self.guide_active = bool(guide_active)
        self.max_guide_steps = int(max_guide_steps)
        self.current_guide_step = 0

    # ------------------------------------------------------------------
    # 查询/控制
    # ------------------------------------------------------------------
    @property
    def exhausted(self) -> bool:
        return self.current_guide_step >= self.max_guide_steps

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_guide_steps - self.current_guide_step)

    def update_guide_steps(self, new_max_steps: int):
        """更新引导最大步数（课程学习：随训练推进逐步减少/调整）。"""
        steps = int(new_max_steps)
        if steps < 0:
            raise ValueError("max_guide_steps 必须 >= 0")
        self.max_guide_steps = steps
        if steps == 0:
            self.guide_active = False

    def set_active(self, active: bool):
        self.guide_active = bool(active)

    def reset(self):
        """每个 episode 重置：步数清零，恢复初始激活状态。"""
        self.current_guide_step = 0
        self.guide_active = self._initial_active

    # ------------------------------------------------------------------
    # 动作分发
    # ------------------------------------------------------------------
    def get_action(self, obs, guide_policy):
        """返回 `(action, is_guided)`；不引导时 action=None。

        `guide_policy` 为可调用对象，签名 `policy(obs) -> action | None`。
        """
        if not self.guide_active or self.exhausted:
            return None, False
        action = guide_policy(obs)
        if action is None:
            return None, False
        self.current_guide_step += 1
        if self.exhausted:
            # 达到上限后本步仍是引导动作，下一步自然交还 DRL
            pass
        return action, True
