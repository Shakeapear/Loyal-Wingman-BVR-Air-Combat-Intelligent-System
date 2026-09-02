# -*- coding: utf-8 -*-
"""任务专精智能体：爬升/下降（TD3，net_arch=[128,128]）。"""
from agents.base_agent import TaskAgent


class ClimbDescentAgent(TaskAgent):
    task_id = "climb_descent"


if __name__ == "__main__":
    agent = ClimbDescentAgent()
    agent.build_model()
    agent.train(timesteps=agent.train_cfg["validation_timesteps"])   # 验证性运行
    agent.save()
