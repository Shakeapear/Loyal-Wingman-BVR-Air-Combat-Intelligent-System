# -*- coding: utf-8 -*-
"""任务专精智能体包：5 个技能级 + 4 个任务级专家智能体，统一注册表。"""
from agents.base_agent import TaskAgent
from agents.climb_descent_td3 import ClimbDescentAgent
from agents.cruise_hold_ppo import CruiseHoldAgent
from agents.evasion_ppo import EvasionAgent
from agents.pursuit_dqn import PursuitAgent
from agents.turn_sac import TurnAgent

AGENTS = {
    "cruise_hold": CruiseHoldAgent,
    "turn": TurnAgent,
    "climb_descent": ClimbDescentAgent,
    "pursuit": PursuitAgent,
    "evasion": EvasionAgent,
}

# 阶段三任务级专家智能体与注册表（延迟到 AGENTS 定义之后，避免循环导入）
from agents.expert_registry import (DEFAULT_REGISTRY, ExpertAgentRegistry,  # noqa: E402
                                    build_default_registry)
from agents.task_agents import (DefenseAgent, EWAgent, PatrolAgent,  # noqa: E402
                                StrikeAgent, TASK_AGENTS)
from agents.hybrid_policy import (HybridActorCritic, HybridActorCriticPolicy,  # noqa: E402
                                  HybridDistribution, MaskedObsWrapper)

__all__ = ["TaskAgent", "CruiseHoldAgent", "TurnAgent", "ClimbDescentAgent",
           "PursuitAgent", "EvasionAgent", "AGENTS",
           "StrikeAgent", "PatrolAgent", "DefenseAgent", "EWAgent", "TASK_AGENTS",
           "ExpertAgentRegistry", "build_default_registry", "DEFAULT_REGISTRY",
           "HybridActorCritic", "HybridActorCriticPolicy", "HybridDistribution",
           "MaskedObsWrapper"]
