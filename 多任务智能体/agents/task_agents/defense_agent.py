# -*- coding: utf-8 -*-
"""任务级专家智能体：战术防御 defense（中期检查版：单机生存/ECM 近似）。"""
from agents.task_agents.base_task_agent import TaskDRLExpertAgent


class DefenseAgent(TaskDRLExpertAgent):
    task_id = "defense"
    mission = "defense"


if __name__ == "__main__":
    agent = DefenseAgent()
    agent.train(timesteps=agent.train_config["validation_timesteps"])
    agent.save()
