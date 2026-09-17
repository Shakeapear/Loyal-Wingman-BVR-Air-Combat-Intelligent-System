# -*- coding: utf-8 -*-
"""
可视化工具/visualization/tactical_display.py（实时战术显示）
================================================================
TacticalDisplay：实时 2D 战术显示窗口（TkAgg，plt.ion 交互模式），
左俯视态势图 + 右态势面板，1 Hz 决策节奏刷新；历史轨迹由内部
TrailHistory 维护（reset() 清空）。

典型用法：
    from visualization.tactical_display import TacticalDisplay
    disp = TacticalDisplay()
    obs, info = env.reset(seed=0)
    disp.reset()
    for _ in range(240):
        obs, r, term, trunc, info = env.step(policy(obs, info))
        disp.update(env.get_viz_frame())
        if term or trunc:
            break
    disp.close()

也可由环境自动创建：BVRCombatEnv(config={"render_mode": "human"}) 后
每步调用 env.render()（内部惰性创建本类实例）。
"""
from __future__ import annotations

import numpy as np

from .tactical_core import TrailHistory, apply_chinese_font, draw_tactical_frame


class TacticalDisplay:
    """实时 2D 战术显示（interactive=True 打开窗口；False 仅离屏重绘，供测试）。"""

    def __init__(self, figsize=(12.6, 6.2), interactive=True, pause_s=0.05):
        apply_chinese_font()
        import matplotlib.pyplot as plt   # 延迟导入：离屏路径不触碰 pyplot 后端
        self.plt = plt
        self.interactive = bool(interactive)
        self.pause_s = float(pause_s)
        self.fig, (self.ax_map, self.ax_panel) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1.0]})
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.09, wspace=0.10)
        self.fig.canvas.manager.set_window_title("BVR 空战战术显示（步骤 2.4）")
        self.hist = TrailHistory()
        self.cum_reward = 0.0
        if self.interactive:
            plt.ion()
            self.fig.show()

    def reset(self):
        """清空历史轨迹与累计奖励（episode 开始时调用）。"""
        self.hist.clear()
        self.cum_reward = 0.0

    def update(self, frame, pause_s=None):
        """刷新一帧（自动累积轨迹与奖励）。"""
        self.cum_reward += frame.get("reward", 0.0)
        self.hist.push(frame)
        draw_tactical_frame(self.ax_map, self.ax_panel, frame, self.hist, self.cum_reward)
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.plt.pause(self.pause_s if pause_s is None else pause_s)
        else:
            self.fig.canvas.draw()

    def hold(self, seconds=3.0):
        """保持窗口一段时间（episode 结束后展示最终态势）。"""
        if self.interactive:
            self.plt.pause(seconds)

    def close(self):
        if self.interactive:
            self.plt.ioff()
        self.plt.close(self.fig)
