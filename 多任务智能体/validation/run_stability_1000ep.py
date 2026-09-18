# -*- coding: utf-8 -*-
"""
validation/run_stability_1000ep.py（环境稳定性验收补测：1000 episode 无崩溃）
================================================================
连续运行 N 个 episode（默认 1000），验收标准：无未捕获异常、观测始终在
观测空间界内、奖励有限、环境可连续 reset。

设计（对应计划书步骤 2.4 验证项「连续运行 1000 episode 无崩溃」）：
- 单环境复用 + 逐 episode 换 seed（seed = --base-seed + episode 序号，可复现）；
- 默认 `--policy level`：近水平巡航动作，能持续飞行（episode 自然超时截断），
  用于稳态长跑；`--policy random` 用随机动作（坠毁即终止，episode 很短，
  用于快速逐条遍历）;
- 默认 `--envelope false`：跳过 reset 时约 1.2 s 的攻击区解算（验收关注环境
  链路稳定性；如需含解算加 `--envelope`，耗时约增加 20 min/1000 ep）；
- 断点续跑：进度写入 results/stability_progress.json（每 25 episode 落盘），
  中断后重跑自动续跑，`--no-resume` 可强制重来；
- 异常不逃跑：记录 traceback 并重建环境继续跑完，最终判定 FAIL 并给证据。

用法（DC 环境，在 多任务智能体/ 目录下）：
    python validation/run_stability_1000ep.py                    # 完整 1000 episode
    python validation/run_stability_1000ep.py --episodes 5       # 冒烟（约 10 s）
    python validation/run_stability_1000ep.py --policy random --episodes 100
    python validation/run_stability_1000ep.py --envelope         # 含攻击区解算

产物：
    validation/results/stability_NNNep_report.txt     最终报告（含结论）
    validation/results/stability_progress.json        断点续跑进度
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.bvr_combat_env import BVRCombatEnv

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PROGRESS_FILE = RESULTS_DIR / "stability_progress.json"

# 近水平巡航动作（与 check_env_and_throughput.py 的吞吐工况一致）
LEVEL_ACTION = {"flight": np.array([0.6, 0.0, -0.15, 0.0], dtype=np.float32),
                "weapon": np.array([0, 0, 0])}


def _new_env(envelope: bool) -> BVRCombatEnv:
    return BVRCombatEnv(config={"compute_envelope": bool(envelope)})


def _run_episode(env, seed: int, policy: str) -> dict:
    """执行单个 episode，返回步数/终止原因/奖励与不变量检查结果。"""
    obs, info = env.reset(seed=seed)
    total_reward, steps, reason = 0.0, 0, None
    obs_bad = reward_bad = 0
    while True:
        action = env.action_space.sample() if policy == "random" else LEVEL_ACTION
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        if not np.all(np.isfinite(obs)):
            obs_bad += 1
        if not np.isfinite(float(reward)):
            reward_bad += 1
        if term or trunc:
            reason = info.get("terminated_reason")
            break
    return {"steps": steps, "reward": total_reward, "reason": reason,
            "obs_nonfinite": obs_bad, "reward_nonfinite": reward_bad}


def main(argv=None):
    ap = argparse.ArgumentParser(description="环境稳定性 1000 episode 验收")
    ap.add_argument("--episodes", type=int, default=1000, help="episode 总数（默认 1000）")
    ap.add_argument("--policy", choices=["level", "random"], default="level",
                    help="动作策略（level=巡航长跑，random=随机短局）")
    ap.add_argument("--envelope", action="store_true",
                    help="reset 时解算攻击区（更接近真实训练，但每次约 +1.2 s）")
    ap.add_argument("--base-seed", type=int, default=10000, help="episode i 的 seed=base+i")
    ap.add_argument("--no-resume", action="store_true", help="忽略进度文件从头跑")
    args = ap.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = {"done": 0, "errors": [], "reasons": {}, "steps_total": 0,
             "obs_nonfinite": 0, "reward_nonfinite": 0, "elapsed_s": 0.0}
    if PROGRESS_FILE.is_file() and not args.no_resume and args.episodes >= 1000:
        try:
            saved = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            if saved.get("policy") == args.policy and saved.get("done", 0) < args.episodes:
                state = saved
                print(f"[续跑] 已完成 {state['done']}/{args.episodes}，继续。", flush=True)
        except Exception as e:
            print(f"[提示] 进度文件不可用（{e}），从头开始。", flush=True)

    env = _new_env(args.envelope)
    t0 = time.time()
    done = int(state.get("done", 0))
    for i in range(done, args.episodes):
        seed = args.base_seed + i
        try:
            ep = _run_episode(env, seed, args.policy)
            state["steps_total"] += ep["steps"]
            state["obs_nonfinite"] += ep["obs_nonfinite"]
            state["reward_nonfinite"] += ep["reward_nonfinite"]
            reason = ep["reason"] or "unterminated"
            state["reasons"][reason] = state["reasons"].get(reason, 0) + 1
        except Exception:
            tb = traceback.format_exc()
            state["errors"].append({"episode": i, "seed": seed, "traceback": tb})
            print(f"[异常] episode {i}（seed={seed}）:\n{tb}", flush=True)
            env.close()
            env = _new_env(args.envelope)     # 重建环境继续，最终判定 FAIL
        state["done"] = i + 1
        state["elapsed_s"] = round(float(state.get("elapsed_s", 0.0)) +
                                   (time.time() - t0), 1)
        if (i + 1) % 25 == 0 or i + 1 == args.episodes:
            wall = time.time() - t0
            print(f"[进度] {i + 1}/{args.episodes}  episode，"
                  f"本段耗时 {str(timedelta(seconds=int(wall)))}，"
                  f"累计 {str(timedelta(seconds=int(state['elapsed_s'])))}，"
                  f"异常 {len(state['errors'])}", flush=True)
            if args.episodes >= 1000:   # 冒烟运行不写进度，避免污染正式长跑续跑点
                PROGRESS_FILE.write_text(json.dumps(
                    {**state, "policy": args.policy, "envelope": args.envelope},
                    ensure_ascii=False, indent=1), encoding="utf-8")
            t0 = time.time()   # 段落计时归零（累计值已并入 state）
    env.close()
    total_wall = float(state.get("elapsed_s", time.time() - t0))
    ok = (not state["errors"] and state["obs_nonfinite"] == 0
          and state["reward_nonfinite"] == 0)
    report = "\n".join([
        f"环境稳定性测试报告（{datetime.now():%Y-%m-%d %H:%M:%S}）",
        "=" * 60,
        f"episode 总数：{state['done']}（policy={args.policy}，"
        f"compute_envelope={bool(args.envelope)}，base_seed={args.base_seed}）",
        f"累计耗时：{str(timedelta(seconds=int(total_wall)))}   "
        f"总决策步：{state['steps_total']}   "
        f"平均步/秒：{state['steps_total'] / max(total_wall, 1e-9):.1f}",
        f"未捕获异常：{len(state['errors'])} 次",
        f"观测非有限值：{state['obs_nonfinite']} 次；奖励非有限值：{state['reward_nonfinite']} 次",
        "episode 终止原因分布：" + ", ".join(
            f"{k}×{v}" for k, v in sorted(state["reasons"].items())),
        f"结论：{'通过（无崩溃，观测/奖励始终有效）' if ok else '失败（见上方异常记录）'}",
    ]) + "\n"
    out = RESULTS_DIR / f"stability_{state['done']}ep_report.txt"
    out.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"报告已写入：{out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
