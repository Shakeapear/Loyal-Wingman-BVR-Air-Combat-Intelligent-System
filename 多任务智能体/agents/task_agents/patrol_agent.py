# -*- coding: utf-8 -*-
"""任务级专家智能体：空域巡逻 patrol（中期检查版：单机拦截近似）。"""
from agents.task_agents.base_task_agent import TaskDRLExpertAgent


class PatrolAgent(TaskDRLExpertAgent):
    task_id = "patrol"
    mission = "patrol"


if __name__ == "__main__":
    agent = PatrolAgent()
    agent.train(timesteps=agent.train_config["validation_timesteps"])
    agent.save()
