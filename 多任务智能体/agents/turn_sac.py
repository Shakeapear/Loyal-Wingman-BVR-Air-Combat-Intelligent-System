# -*- coding: utf-8 -*-
"""任务专精智能体：盘旋机动（SAC，net_arch=[128,128]）。"""
from agents.base_agent import TaskAgent


class TurnAgent(TaskAgent):
    task_id = "turn"


if __name__ == "__main__":
    agent = TurnAgent()
    agent.build_model()
    agent.train(timesteps=agent.train_cfg["validation_timesteps"])   # 验证性运行
    agent.save()
