# -*- coding: utf-8 -*-
"""任务专精智能体包：不同任务使用不同 DRL 模型。"""
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

__all__ = ["TaskAgent", "CruiseHoldAgent", "TurnAgent", "ClimbDescentAgent",
           "PursuitAgent", "EvasionAgent", "AGENTS"]
