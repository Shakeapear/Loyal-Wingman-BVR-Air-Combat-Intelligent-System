# -*- coding: utf-8 -*-
"""
visualization/offscreen.py（步骤 2.4 离屏渲染）
================================================================
离屏（Agg canvas，无需显示设备）渲染路径：
- frame_to_rgb(frame)      : 单帧 → HxWx3 uint8（env.render("rgb_array") 用）
- render_episode(frames)   : 帧序列 → GIF（PillowWriter，零额外依赖）
                             或 MP4（需 imageio-ffmpeg，自动定位其自带 ffmpeg）

刻意不 import pyplot、不切换全局 backend：直接构造 Figure + FigureCanvasAgg，
与实时 TkAgg 窗口路径互不干扰，可安全用于训练机/无头环境。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .tactical_core import TrailHistory, apply_chinese_font, draw_tactical_frame

DEFAULT_FIGSIZE = (12.4, 6.0)


def _new_figure(figsize=DEFAULT_FIGSIZE, dpi=100):
    """新建离屏画布（左：俯视态势图；右：态势面板）。"""
    apply_chinese_font()
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax_map, ax_panel = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1.0]})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.09, wspace=0.10)
    return fig, ax_map, ax_panel


_fig_cache = {}   # (figsize, dpi) -> (fig, ax_map, ax_panel)，frame_to_rgb 复用画布


def frame_to_rgb(frame, figsize=DEFAULT_FIGSIZE, dpi=100):
    """渲染单帧为 RGB 数组（HxWx3 uint8）。hist 为空（仅当前帧，无历史轨迹）。

    画布按 (figsize, dpi) 缓存复用：env.render("rgb_array") 逐帧调用时
    不再每次新建 Figure（避免未释放画布随帧数累积，依赖循环 GC 的问题）。
    """
    key = (tuple(figsize), int(dpi))
    if key not in _fig_cache:
        _fig_cache[key] = _new_figure(figsize, dpi)
    fig, ax_map, ax_panel = _fig_cache[key]
    hist = TrailHistory()
    hist.push(frame)
    draw_tactical_frame(ax_map, ax_panel, frame, hist)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def _make_writer(out_path, fps):
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        from matplotlib.animation import PillowWriter
        return PillowWriter(fps=fps)
    if suffix == ".mp4":
        try:
            import imageio_ffmpeg
        except ImportError as e:
            raise RuntimeError(
                "MP4 渲染需要 imageio-ffmpeg（pip install imageio-ffmpeg，"
                "自带 ffmpeg 可执行文件，无需系统安装）；GIF 无任何额外依赖") from e
        from matplotlib.animation import FFMpegWriter
        rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return FFMpegWriter(fps=fps, metadata={"title": "BVR 1v1 episode"})
    raise ValueError(f"不支持的输出格式 {suffix!r}，请用 .gif 或 .mp4")


def render_episode(frames, out_path, fps=10, figsize=DEFAULT_FIGSIZE, dpi=90,
                   progress=False):
    """把帧序列批量渲染为 GIF/MP4（回放主入口）。

    frames   : get_viz_frame() 帧列表（或 EpisodeRecorder.load_csv 的返回）
    out_path : 输出路径（.gif / .mp4）
    fps      : 播放帧率（决策 1 Hz，fps=10 即 10 倍速回放）
    """
    frames = list(frames)
    if not frames:
        raise ValueError("frames 为空，无法渲染")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_map, ax_panel = _new_figure(figsize, dpi)
    hist = TrailHistory()
    cum = 0.0
    writer = _make_writer(out_path, fps)
    with writer.saving(fig, str(out_path), dpi=dpi):
        for i, frm in enumerate(frames):
            cum += frm.get("reward", 0.0)
            hist.push(frm)
            draw_tactical_frame(ax_map, ax_panel, frm, hist, cum)
            writer.grab_frame()
            if progress and (i % 20 == 0 or i == len(frames) - 1):
                print(f"  [渲染] {i + 1}/{len(frames)} 帧", flush=True)
    fig.clear()   # 显式释放 artist/画布引用（fig↔canvas 引用环仅靠循环 GC 延迟回收）
    return out_path
