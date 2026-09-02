# -*- coding: utf-8 -*-
"""
make_trajectory.py
==================
把 JSBSim 原始 CSV（含自驾仪指令）转换为**智能体训练用轨迹文件**（标准格式 trajectory_v1）。

输出文件结构：
    - 前若干行以 # 开头的元数据（键值对，pandas 用 comment='#' 读取可自动跳过）；
    - 一行表头（列名）；
    - 数据行：1 Hz（可改），每行 = 一个决策步 (state, action, label, done)。

用法:
    py make_trajectory.py <原始CSV> <输出CSV> [选项]

选项:
    --rate <Hz>            输出采样率（默认 1）
    --episode <ID>         回合编号（默认由输出文件名推断）
    --scenario <编号>      场景编号（如 S3）
    --task <标签>          任务标签（如 flight_skill_demo / cruise / turn）
    --collector <姓名>     采集人
    --script <路径>        生成该数据的 JSBSim 脚本名

示例:
    py make_trajectory.py JSBSim\f104_training_demo_raw.csv 样例数据\f104_demo_ep0001.csv --episode f104_demo_ep0001 --scenario S3 --task flight_skill_demo --collector 唐煜皓

阶段标签（phase_label）由自驾仪指令自动推导：
    |roll指令| > 0.02 → turn_right / turn_left
    高度指令升高      → climb
    高度指令降低      → descend
    其余              → cruise
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

from convert_jsbsim_csv import load_raw, build_si_dataframe, resample

FT2M = 0.3048
FPS2MPS = 0.3048
FORMAT_VERSION = "trajectory_v1"


def derive_phase(df):
    """根据自驾仪指令与高度误差推导每步的阶段标签（150 m 死区）。"""
    roll = df["fcs/ap-roll-setpoint-rad"].to_numpy(dtype=float)
    h = df["h_sl_m"].to_numpy(dtype=float)
    alt_set_m = df["fcs/ap-alt-setpoint-ft"].to_numpy(dtype=float) * FT2M

    labels = []
    for r, hh, a in zip(roll, h, alt_set_m):
        if r > 0.02:
            labels.append("turn_right")
        elif r < -0.02:
            labels.append("turn_left")
        elif a < 100.0:          # t=0 时高度指令尚未设置
            labels.append("cruise")
        elif hh < a - 150.0:
            labels.append("climb")
        elif hh > a + 150.0:
            labels.append("descend")
        else:
            labels.append("cruise")
    return labels


SETPOINT_COLS = ["fcs/ap-roll-setpoint-rad", "fcs/ap-alt-setpoint-ft",
                 "fcs/ap-speed-setpoint-fps"]


def build_trajectory(raw_df, rate, meta):
    si = build_si_dataframe(raw_df)
    missing = [c for c in SETPOINT_COLS if c not in raw_df.columns]
    if missing:
        print("[error] 原始 CSV 缺少自驾仪指令列，请在脚本 <output> 中加入以下属性后重跑:")
        for c in missing:
            print(f"        <property> {c} </property>")
        return None
    for c in SETPOINT_COLS:
        si[c] = raw_df[c].to_numpy(dtype=float)

    df = resample(si, rate).copy()

    v = df["vtrue_mps"].to_numpy(dtype=float)
    vd = df["vd_mps"].to_numpy(dtype=float)
    gamma = np.degrees(-np.arcsin(np.clip(vd / np.maximum(v, 1.0), -1.0, 1.0)))
    df["gamma_deg"] = gamma

    df["episode_id"] = meta["episode_id"]
    df["step"] = np.arange(len(df))
    df["task_label"] = meta["task_label"]
    df["phase_label"] = derive_phase(df)
    df["action_roll_cmd_rad"] = df["fcs/ap-roll-setpoint-rad"].to_numpy(dtype=float)
    df["action_alt_cmd_m"] = df["fcs/ap-alt-setpoint-ft"].to_numpy(dtype=float) * FT2M
    df["action_speed_cmd_mps"] = df["fcs/ap-speed-setpoint-fps"].to_numpy(dtype=float) * FPS2MPS
    done = np.zeros(len(df), dtype=int)
    done[-1] = 1
    df["done"] = done

    lead = ["episode_id", "step", "t_s", "task_label", "phase_label",
            "action_roll_cmd_rad", "action_alt_cmd_m", "action_speed_cmd_mps",
            "gamma_deg", "done"]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


def main(argv):
    ap = argparse.ArgumentParser(description="生成训练轨迹文件（trajectory_v1）")
    ap.add_argument("raw_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--episode", default="")
    ap.add_argument("--scenario", default="")
    ap.add_argument("--task", default="flight_skill_demo")
    ap.add_argument("--collector", default="")
    ap.add_argument("--script", default="")
    args = ap.parse_args(argv[1:])

    episode = args.episode or os.path.splitext(os.path.basename(args.out_csv))[0]

    meta = {
        "format": FORMAT_VERSION,
        "episode_id": episode,
        "aircraft": "f104",
        "scenario": args.scenario,
        "task_label": args.task,
        "collector": args.collector,
        "script": args.script,
        "rate_hz": str(args.rate),
        "jsbsim_version": "1.2.4",
    }

    raw = load_raw(args.raw_csv)
    df = build_trajectory(raw, args.rate, meta)
    if df is None:
        return 1

    header = "".join(f"#{k}: {v}\n" for k, v in meta.items())
    body = df.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        f.write(body)

    counts = df.groupby("phase_label").size().to_dict()
    print(f"[ok] 已写出 {args.out_csv}  行数={len(df)}  列数={len(df.columns)}")
    print("     阶段分布:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
