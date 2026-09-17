# -*- coding: utf-8 -*-
"""
可视化工具/visualization（步骤 2.4 交付物 D2.4-1：项目级可视化工具）
================================================================
定位：训练自我调整 + 成果展示。独立于 多任务智能体/ 智能体库：
只消费 get_viz_frame() 帧字典，不含任何智能体逻辑；训练路径零开销。

- tactical_core      绘图内核（后端无关，俯视态势 + 态势面板）
- tactical_display   TacticalDisplay：实时战术显示窗口（TkAgg）
- offscreen          离屏渲染：frame_to_rgb / render_episode（GIF/MP4，Agg）
- recorder           EpisodeRecorder：逐帧 CSV 记录/加载
- acmi               TacView ACMI 2.2 回放导出（可选增强）
- replay             回放 CLI：python -m visualization.replay xxx.csv
- csv2acmi           批量 CSV → ACMI CLI
- demo               演示 CLI：python -m visualization.demo

帧来源：多任务智能体/common/bvr_combat_env.py 的 BVRCombatEnv.get_viz_frame()；
环境侧经 config["render_mode"] = "human" / "rgb_array" 启用 env.render()
（惰性接入本包）。命令均在 可视化工具/ 目录下运行，详见 可视化工具/README.md。
"""
from .recorder import EpisodeRecorder
from .tactical_core import TrailHistory, apply_chinese_font, draw_tactical_frame

__all__ = [
    "EpisodeRecorder",
    "TrailHistory",
    "apply_chinese_font",
    "draw_tactical_frame",
]
