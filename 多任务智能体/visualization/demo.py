# -*- coding: utf-8 -*-
"""
visualization/demo.py（步骤 2.4 演示脚本）
================================================================
BVR 空战可视化演示：脚本策略（接敌→攻击区边缘发射→RWR 开 ECM→MAWS 规避）
驱动 BVRCombatEnv，实时战术显示 + CSV 录制 + 离线 GIF 渲染 + 可选 ACMI。

用法（DC 环境，在 多任务智能体/ 目录下）：
    python -m visualization.demo                        # 实时窗口 + 录制 CSV/GIF
    python -m visualization.demo --seed 7 --steps 240
    python -m visualization.demo --no-live              # 无窗口（仅离线录制渲染）
    python -m visualization.demo --no-gif --acmi        # 只录制 CSV + 导出 ACMI

脚本策略仅为演示服务（非训练产物），动作空间与 DRL 智能体一致（Dict 动作）。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.bvr_combat_env import BVRCombatEnv
from visualization.acmi import export_acmi
from visualization.offscreen import render_episode
from visualization.recorder import EpisodeRecorder

RESULTS = Path(__file__).resolve().parent / "results"

# 地理原点（须与 common/bvr_combat_env.py reset() 的 lat0/lon0 一致；
# 经 EpisodeRecorder.meta 传给 ACMI 导出，消除多处重复硬编码）
LAT0_DEG, LON0_DEG = 30.0, 120.0


class ScriptedDemoPolicy:
    """演示脚本策略：对准目标巡航，进入攻击区边沿发射，RWR 告警开 ECM，
    MAWS 告警大坡度规避滚转。动作与 BVRCombatEnv Dict 动作空间一致。

    姿态闭环（1 Hz 人工通道，增益经实测整定）：
    - 滚转：目标相对方位 → 坡度指令（±40°）→ 副翼指令（含坡度反馈阻尼）。
      直接以方位角×增益驱动副翼会在 1 Hz 零阶保持下失稳（phi 在 ±180°
      振荡→倒飞拉杆加速俯冲，实测复现），必须经坡度内环；
    - 俯仰：高度误差 → 垂速指令 → 俯仰指令。F-104 人工通道升降舵符号与
      常规相反（实测：正指令→低头），爬升需负指令；零输入配平不能保持
      高度（8 s 掉约 200 m），必须闭环。

    战术逻辑：开局即开 ECM（敌雷达有效距离 ×0.5 → 敌首发推迟）；进入
    攻击区边沿双发齐射（间隔 4 s，提高命中概率）；首发后立即 F-pole
    偏转（crank 60° 背向敌方位+增速下降，拉长敌弹拦截链——实测同时
    发射时我弹命中总晚 1~2 s，crank 是胜负手）；MAWS 告警持续大坡度
    转弯，TTA<1.5 s 时 65° 急转（末端 jink，使 30g 导弹在最小转弯半径
    内无法修正）。"""

    def __init__(self, h_ref_m):
        self.h_ref_m = float(h_ref_m)
        self.prev_in_zone = False
        self.ecm_on = False
        self.n_fired = 0
        self.last_fire_step = -99
        self.crank_sign = 1.0       # F-pole 偏转方向（首发时按背向敌方位确定）

    def __call__(self, obs, info):
        detected = float(obs[21]) > 0.5
        bearing = float(obs[18])                    # 目标相对方位 rad（未探测到为 0）
        in_zone = bool(info.get("in_zone", False))
        maws = bool(info.get("maws_alarm", False))
        maws_tta = float(info.get("maws_tta", 99.0))
        step = int(info.get("steps", 0))
        # ---- 滚转：方位 → 坡度指令 → 副翼（phi 内环阻尼）----
        if maws and maws_tta < 1.5:
            phi_cmd_deg = 65.0 * self.crank_sign    # 末端 jink：最大急转
        elif maws:
            phi_cmd_deg = 55.0 * self.crank_sign    # 来袭导弹：持续大坡度+下降
        elif self.n_fired > 0:
            phi_cmd_deg = 60.0 * self.crank_sign    # F-pole 偏转（导弹已离架）
        elif detected:
            phi_cmd_deg = float(np.clip(np.rad2deg(bearing), -40.0, 40.0))
        else:
            phi_cmd_deg = 20.0                      # 未探测到：缓转搜索
        phi_deg = float(np.rad2deg(obs[3]))
        roll = float(np.clip(0.010 * (phi_cmd_deg - phi_deg), -0.3, 0.3))
        # ---- 俯仰：h 误差 → vd 指令 → 俯仰指令（正俯仰=低头）；规避时附加下降
        # （高度下限：低于 2500 m 不再下降，防止长局持续规避撞地）----
        vd_cmd = float(np.clip(0.5 * (self.h_ref_m - float(obs[0])), -20.0, 20.0))
        if (maws or self.n_fired > 0) and float(obs[0]) > 2500.0:
            vd_cmd -= 15.0
        pitch = float(np.clip(-0.01 * (vd_cmd - float(obs[11])), -0.3, 0.3))
        throttle = 1.0 if (maws or self.n_fired > 0) else 0.85
        flight = np.array([throttle, roll, pitch, 0.0], dtype=np.float32)
        # ---- 武器：攻击区双发齐射（首发边沿 + 4 s 后补射），ECM 开局开启 ----
        fire = 0
        if in_zone and self.n_fired < 2:
            if (not self.prev_in_zone) or (self.n_fired == 1 and step - self.last_fire_step >= 4):
                fire, self.n_fired, self.last_fire_step = 1, self.n_fired + 1, step
                self.crank_sign = -1.0 if bearing >= 0.0 else 1.0   # 背向目标偏转
        ecm = 1 if not self.ecm_on else 0
        self.ecm_on = True
        self.prev_in_zone = in_zone
        return {"flight": flight, "weapon": np.array([fire, 0, ecm])}


def main(argv=None):
    p = argparse.ArgumentParser(description="BVR 空战可视化演示（步骤 2.4）")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--no-live", action="store_true", help="不开实时窗口（离线模式）")
    p.add_argument("--no-gif", action="store_true", help="跳过 GIF 渲染")
    p.add_argument("--acmi", action="store_true", help="同时导出 TacView ACMI")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--tag", default=None, help="输出文件名后缀（默认 seed）")
    args = p.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    # --tag 清洗：仅字母数字下划线连字符（防止 ../ 目录遍历与路径注入）
    tag = re.sub(r"[^\w\-]", "_", args.tag) if args.tag else f"seed{args.seed}"
    live = not args.no_live

    env = BVRCombatEnv(config={"compute_envelope": True})   # 演示需要真实攻击区扇形
    if live:
        env.render_mode = "human"
    recorder = EpisodeRecorder(meta={"lat0_deg": LAT0_DEG, "lon0_deg": LON0_DEG})

    print(f"[演示] seed={args.seed}  最大步数={args.steps}  实时窗口={'开' if live else '关'}",
          flush=True)
    t0 = time.time()
    obs, info = env.reset(seed=args.seed)
    policy = ScriptedDemoPolicy(h_ref_m=info["h_sl_m"])   # 高度保持基准=初始高度
    if live:
        env.render()          # 首帧（含初始态势）
    recorder.capture_env(env)

    terminated = truncated = False
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(policy(obs, info))
        if live:
            env.render()      # 1 Hz 战术显示刷新
        recorder.capture_env(env)
        if terminated or truncated:
            break

    reason = info.get("terminated_reason") or "（未终止）"
    print(f"[演示结束] 步数={info['steps']}  终止原因={reason}  "
          f"仿真耗时 {time.time() - t0:.1f} s", flush=True)
    if live:
        env._display.hold(3.0)
    env.close()

    csv_path = recorder.save_csv(RESULTS / f"demo_ep_{tag}.csv")
    print(f"[CSV] {len(recorder)} 帧 → {csv_path}", flush=True)
    if not args.no_gif:
        t1 = time.time()
        gif = render_episode(recorder.frames, RESULTS / f"demo_ep_{tag}.gif",
                             fps=args.fps, progress=True)
        print(f"[GIF] → {gif}（渲染 {time.time() - t1:.0f} s）", flush=True)
    if args.acmi:
        acmi = export_acmi(recorder.frames, RESULTS / f"demo_ep_{tag}.acmi",
                           lat0_deg=recorder.meta["lat0_deg"],
                           lon0_deg=recorder.meta["lon0_deg"],
                           title=f"BVR 1v1 (seed {args.seed})")
        print(f"[ACMI] → {acmi}", flush=True)


if __name__ == "__main__":
    main()
