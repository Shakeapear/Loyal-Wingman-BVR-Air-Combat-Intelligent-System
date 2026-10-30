# -*- coding: utf-8 -*-
"""任务级专家智能体：电子压制 ew（中期检查版：单机 ECM 干扰近似）。"""
from agents.task_agents.base_task_agent import TaskDRLExpertAgent


class EWAgent(TaskDRLExpertAgent):
    task_id = "ew"
    mission = "ew"


if __name__ == "__main__":
    agent = EWAgent()
    agent.train(timesteps=agent.train_config["validation_timesteps"])
    agent.save()
