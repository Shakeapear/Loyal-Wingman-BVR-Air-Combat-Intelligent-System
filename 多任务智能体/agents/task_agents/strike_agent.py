# -*- coding: utf-8 -*-
"""任务级专家智能体：超视距打击 strike（阶段三基线，复用 BVR 环境 + PPO）。"""
from agents.task_agents.base_task_agent import TaskDRLExpertAgent


class StrikeAgent(TaskDRLExpertAgent):
    task_id = "strike"
    mission = "strike"


if __name__ == "__main__":
    agent = StrikeAgent()
    agent.train(timesteps=agent.train_config["validation_timesteps"])
    agent.save()
