# -*- coding: utf-8 -*-
"""
visualization/replay.py（步骤 2.4 交付物 D2.4-1：回放脚本）
================================================================
把 EpisodeRecorder 保存的 CSV 回放渲染为 GIF/MP4，可选同时导出 ACMI。

用法（DC 环境，在 多任务智能体/ 目录下）：
    python -m visualization.replay visualization/results/demo_ep.csv
    python -m visualization.replay xxx.csv --out out.mp4 --fps 12
    python -m visualization.replay xxx.csv --acmi xxx.acmi
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.acmi import export_acmi
from visualization.offscreen import render_episode
from visualization.recorder import EpisodeRecorder


def main(argv=None):
    p = argparse.ArgumentParser(description="BVR 空战 episode CSV 回放渲染（GIF/MP4/ACMI）")
    p.add_argument("csv", help="EpisodeRecorder 保存的 CSV 路径")
    p.add_argument("--out", default=None, help="输出 .gif/.mp4（默认与 CSV 同名 .gif）")
    p.add_argument("--fps", type=int, default=10, help="播放帧率（默认 10，即 10 倍速）")
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--acmi", default=None, help="可选：同时导出 TacView ACMI 文件路径")
    args = p.parse_args(argv)

    frames = EpisodeRecorder.load_csv(args.csv)
    out = args.out or str(Path(args.csv).with_suffix(".gif"))
    t0 = time.time()
    render_episode(frames, out, fps=args.fps, dpi=args.dpi, progress=True)
    print(f"[回放完成] {len(frames)} 帧 → {out}（{time.time() - t0:.0f} s）")
    if args.acmi:
        export_acmi(frames, args.acmi)
        print(f"[ACMI 导出] → {args.acmi}")


if __name__ == "__main__":
    main()
