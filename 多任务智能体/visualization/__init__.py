# -*- coding: utf-8 -*-
"""
visualization/（步骤 2.4 交付物 D2.4-1：可视化模块）
================================================================
BVR 空战 2D 战术显示与回放：

- tactical_core      绘图内核（后端无关，俯视态势 + 态势面板）
- tactical_display   TacticalDisplay：实时战术显示窗口（TkAgg）
- offscreen          离屏渲染：frame_to_rgb / render_episode（GIF/MP4，Agg）
- recorder           EpisodeRecorder：逐帧 CSV 记录/加载
- acmi               TacView ACMI 2.2 回放导出（可选增强）
- replay             回放 CLI：python -m visualization.replay xxx.csv
- demo               演示 CLI：python -m visualization.demo

输入统一为 common.bvr_combat_env.BVRCombatEnv.get_viz_frame() 帧字典。
环境侧经 config["render_mode"] = "human" / "rgb_array" 启用 env.render()。
"""
from .recorder import EpisodeRecorder
from .tactical_core import TrailHistory, apply_chinese_font, draw_tactical_frame

__all__ = [
    "EpisodeRecorder",
    "TrailHistory",
    "apply_chinese_font",
    "draw_tactical_frame",
]
