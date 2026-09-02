# -*- coding: utf-8 -*-
"""任务专精智能体：接敌追击（DQN，离散 48 动作，net_arch=[128,128]）。"""
from agents.base_agent import TaskAgent


class PursuitAgent(TaskAgent):
    task_id = "pursuit"


if __name__ == "__main__":
    agent = PursuitAgent()
    agent.build_model()
    agent.train(timesteps=agent.train_cfg["validation_timesteps"])   # 验证性运行
    agent.save()
