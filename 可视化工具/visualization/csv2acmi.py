# -*- coding: utf-8 -*-
"""
可视化工具/visualization/csv2acmi.py（任意录制文件 → TacView ACMI 转换器）
================================================================
把 EpisodeRecorder 保存的 CSV 录制文件转换为 ACMI 2.2 文本，
输出 .acmi 到原 CSV 同目录，双击即可用 TacView 3D 回放。
离线可用：无需环境与智能体库在场（帧格式自包含）。

适用于训练/评测中积累的任意多份录制文件的批量可视化：
转换后每个 .acmi 都可独立双击打开（可多开 TacView 实例对比）。

用法（DC 环境，在 可视化工具/ 目录下）：
    python -m visualization.csv2acmi results/demo_ep_seed8.csv
    python -m visualization.csv2acmi a.csv b.csv c.csv          # 多个文件
    python -m visualization.csv2acmi x.csv -o out/x.acmi        # 指定输出
    python -m visualization.csv2acmi x.csv --lat0 30 --lon0 120 # 地理原点
"""
from __future__ import annotations

import argparse
from pathlib import Path

from visualization.acmi import export_acmi
from visualization.recorder import EpisodeRecorder


def convert(csv_path, out_path=None, lat0_deg=30.0, lon0_deg=120.0, title=None):
    """单个 CSV → ACMI，返回输出路径。"""
    csv_path = Path(csv_path)
    frames = EpisodeRecorder.load_csv(csv_path)
    if not frames:
        raise ValueError(f"{csv_path} 中没有帧数据")
    out_path = Path(out_path) if out_path else csv_path.with_suffix(".acmi")
    return export_acmi(frames, out_path, lat0_deg=lat0_deg, lon0_deg=lon0_deg,
                       title=title or csv_path.stem)


def main(argv=None):
    p = argparse.ArgumentParser(description="CSV 录制文件 → TacView ACMI 转换")
    p.add_argument("csv", nargs="+", help="一个或多个 EpisodeRecorder CSV 文件")
    p.add_argument("-o", "--out", default=None,
                   help="输出路径（仅单文件时可用；默认同名 .acmi）")
    p.add_argument("--lat0", type=float, default=30.0, help="地理原点纬度（默认 30）")
    p.add_argument("--lon0", type=float, default=120.0, help="地理原点经度（默认 120）")
    p.add_argument("--title", default=None, help="ACMI 标题（默认取 CSV 文件名）")
    args = p.parse_args(argv)
    if args.out and len(args.csv) > 1:
        p.error("-o/--out 仅支持单文件转换")
    for csv_file in args.csv:
        out = convert(csv_file, args.out, args.lat0, args.lon0, args.title)
        print(f"[ACMI] {csv_file} → {out}", flush=True)


if __name__ == "__main__":
    main()
