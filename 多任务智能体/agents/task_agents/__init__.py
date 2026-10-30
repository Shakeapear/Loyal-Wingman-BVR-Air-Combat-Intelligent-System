# -*- coding: utf-8 -*-
"""阶段三任务级专家智能体包（strike/patrol/defense/ew）。"""
from agents.task_agents.base_task_agent import TaskDRLExpertAgent
from agents.task_agents.defense_agent import DefenseAgent
from agents.task_agents.ew_agent import EWAgent
from agents.task_agents.mission_env import (MISSION_ENV_CONFIG, MISSION_IDS,
                                           MISSION_LABELS, MISSION_NAMES_ZH,
                                           FlatTaskMissionEnv, TaskMissionEnv,
                                           make_task_env)
from agents.task_agents.patrol_agent import PatrolAgent
from agents.task_agents.strike_agent import StrikeAgent

TASK_AGENTS = {
    "strike": StrikeAgent,
    "patrol": PatrolAgent,
    "defense": DefenseAgent,
    "ew": EWAgent,
}

__all__ = ["TaskDRLExpertAgent", "StrikeAgent", "PatrolAgent", "DefenseAgent",
           "EWAgent", "TASK_AGENTS", "FlatTaskMissionEnv", "TaskMissionEnv",
           "make_task_env", "MISSION_IDS", "MISSION_LABELS", "MISSION_NAMES_ZH",
           "MISSION_ENV_CONFIG"]
