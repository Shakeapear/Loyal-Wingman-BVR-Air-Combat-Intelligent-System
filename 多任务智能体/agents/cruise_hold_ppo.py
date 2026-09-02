# -*- coding: utf-8 -*-
"""任务专精智能体：巡航保持（PPO，net_arch=[64,64]）。"""
from agents.base_agent import TaskAgent


class CruiseHoldAgent(TaskAgent):
    task_id = "cruise_hold"


if __name__ == "__main__":
    agent = CruiseHoldAgent()
    agent.build_model()
    agent.train(timesteps=agent.train_cfg["validation_timesteps"])   # 验证性运行
    agent.save()
