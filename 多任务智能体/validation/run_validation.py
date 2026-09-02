# -*- coding: utf-8 -*-
"""
validation/run_validation.py
============================
验证性运行脚本（非完整训练！）

对每个任务专精智能体执行三级验证：
    L1 环境冒烟：reset + 随机动作 rollout 60 步，确认 JSBSim 环境链路正常；
    L2 模型构建：按 config.py 节点数构建对应 DRL 模型，打印网络结构；
    L3 极小训练：仅训练 validation_timesteps 步（1024~2048，几分钟级），
       确认训练循环可运行——**不做完整训练**（完整训练步数见 config.py 的
       total_timesteps_full，默认 50 万~200 万步）。

用法（DC 环境）：
    python validation/run_validation.py [--skip-train] [--tasks cruise_hold,turn]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import AGENTS
from common.config import TASK_CONFIGS
from common.tasks import TASKS, init_task_state

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def check_env_smoke(task_id, n_steps=60):
    agent_cls = AGENTS[task_id]
    agent = agent_cls()
    env = agent.build_env()
    obs, _ = agent.reset_env()
    total_reward, crashed = 0.0, False
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated:
            crashed = info.get("crash", False)
            break
    print(f"  [L1 环境冒烟] {task_id}: {n_steps} 步随机动作, 累计奖励 {total_reward:.1f}, "
          f"终止={terminated}, 坠毁={crashed}, obs={env.observation_space}, act={env.action_space}")
    return True


def check_model_build(task_id):
    agent_cls = AGENTS[task_id]
    agent = agent_cls()
    model = agent.build_model()
    cfg = TASK_CONFIGS[task_id]
    print(f"  [L2 模型构建] {task_id}: algo={cfg['algo']}, net_arch={cfg['net_arch']}, "
          f"activation={cfg['activation']}, lr={cfg['learning_rate']}")
    return True


def check_tiny_train(task_id):
    agent_cls = AGENTS[task_id]
    agent = agent_cls()
    model = agent.build_model()
    ts = TASK_CONFIGS[task_id]["validation_timesteps"]
    t0 = time.time()
    model.learn(total_timesteps=ts, progress_bar=False)
    wall = time.time() - t0
    agent.model = model
    path = agent.save()
    print(f"  [L3 极小训练] {task_id}: {ts} 步耗时 {wall:.1f}s -> {path.name}")
    return True


def main():
    ap = argparse.ArgumentParser(description="多任务智能体验证性运行")
    ap.add_argument("--tasks", default="all", help="逗号分隔的任务列表，默认 all")
    ap.add_argument("--skip-train", action="store_true", help="跳过 L3 极小训练")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    task_ids = list(AGENTS.keys()) if args.tasks == "all" else args.tasks.split(",")

    report = []
    for task_id in task_ids:
        print(f"== 验证任务: {task_id} ({TASKS[task_id]['algo']}) ==")
        ok = check_env_smoke(task_id)
        ok = check_model_build(task_id) and ok
        if not args.skip_train:
            ok = check_tiny_train(task_id) and ok
        report.append(f"{task_id}: {'通过' if ok else '失败'}")

    print("\n===== 验证汇总 =====")
    for line in report:
        print(" " + line)

    with open(RESULTS_DIR / "validation_report.txt", "w", encoding="utf-8") as f:
        f.write("多任务智能体验证性运行报告（未执行完整训练）\n")
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
