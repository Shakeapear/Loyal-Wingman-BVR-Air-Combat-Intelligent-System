# -*- coding: utf-8 -*-
"""任务专精智能体：威胁规避（PPO，net_arch=[256,256]，奖励稀疏）。"""
from agents.base_agent import TaskAgent


class EvasionAgent(TaskAgent):
    task_id = "evasion"


if __name__ == "__main__":
    agent = EvasionAgent()
    agent.build_model()
    agent.train(timesteps=agent.train_cfg["validation_timesteps"])   # 验证性运行
    agent.save()
