# -*- coding: utf-8 -*-
"""
agents/expert_registry.py
=========================
专家智能体注册表：4 个任务级（strike/patrol/defense/ew）+ 5 个技能级
（cruise_hold/turn/climb_descent/pursuit/evasion）统一按任务名解析。

行为树主通道只选任务级（tier="task"），调用 get_action → ActionV1；
技能级（tier="skill"）保留 command_v1 兼容通道，不进入 ActionV1 训练主线。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agents import AGENTS as SKILL_AGENTS
from agents.task_agents import TASK_AGENTS

TIER_TASK = "task"
TIER_SKILL = "skill"

TASK_ALIASES = {
    "超视距打击": "strike", "打击": "strike", "bvr_strike": "strike",
    "空域巡逻": "patrol", "巡逻": "patrol",
    "战术防御": "defense", "防御": "defense",
    "电子压制": "ew", "电子战": "ew", "干扰": "ew",
}
SKILL_ALIASES = {
    "巡航保持": "cruise_hold", "巡航": "cruise_hold",
    "盘旋": "turn", "盘旋机动": "turn",
    "爬升下降": "climb_descent", "爬降": "climb_descent",
    "接敌追击": "pursuit", "追击": "pursuit",
    "威胁规避": "evasion", "规避": "evasion",
}


@dataclass(frozen=True)
class AgentRegistration:
    task_id: str
    tier: str
    agent_cls: type
    description: str = ""
    action_channel: str = "action_v1"      # action_v1 | command_v1
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "class": self.agent_cls.__name__,
            "description": self.description,
            "action_channel": self.action_channel,
            "aliases": list(self.aliases),
        }


class ExpertAgentRegistry:
    """任务名 → 智能体类 的注册/解析/实例化中心。"""

    def __init__(self):
        self._entries: dict[str, AgentRegistration] = {}
        self._aliases: dict[str, str] = {}

    # ---------- 注册 ----------
    def register(self, task_id: str, agent_cls: type, tier: str = TIER_TASK,
                 description: str = "", action_channel: str = "action_v1",
                 aliases=()) -> AgentRegistration:
        if tier not in (TIER_TASK, TIER_SKILL):
            raise ValueError(f"非法 tier: {tier}")
        task_id = str(task_id)
        if task_id in self._entries:
            raise ValueError(f"重复注册: {task_id}")
        entry = AgentRegistration(task_id=task_id, tier=tier, agent_cls=agent_cls,
                                  description=description, action_channel=action_channel,
                                  aliases=tuple(aliases))
        self._entries[task_id] = entry
        for alias in (task_id, *entry.aliases):
            if alias in self._aliases and self._aliases[alias] != task_id:
                raise ValueError(f"别名冲突: {alias}")
            self._aliases[str(alias)] = task_id
        return entry

    def register_alias(self, alias: str, task_id: str) -> None:
        target = self.resolve(task_id)
        if alias in self._aliases and self._aliases[alias] != target:
            raise ValueError(f"别名冲突: {alias}")
        self._aliases[str(alias)] = target

    # ---------- 解析/实例化 ----------
    def resolve(self, name: str) -> str:
        key = str(name)
        if key not in self._aliases:
            raise KeyError(f"未注册的专家智能体: {name}")
        return self._aliases[key]

    def get(self, name: str) -> AgentRegistration:
        return self._entries[self.resolve(name)]

    def create(self, name: str, **kwargs):
        return self.get(name).agent_cls(**kwargs)

    # ---------- 查询 ----------
    def names(self, tier: str | None = None) -> list[str]:
        return [e.task_id for e in self._entries.values()
                if tier is None or e.tier == tier]

    def task_ids(self) -> list[str]:
        return self.names(TIER_TASK)

    def skill_ids(self) -> list[str]:
        return self.names(TIER_SKILL)

    def list(self, tier: str | None = None) -> list[dict]:
        return [e.to_dict() for e in self._entries.values()
                if tier is None or e.tier == tier]

    def __contains__(self, name: str) -> bool:
        return str(name) in self._aliases

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())


def build_default_registry() -> ExpertAgentRegistry:
    """构建默认 9 项注册表（4 任务级 + 5 技能级）。"""
    reg = ExpertAgentRegistry()
    descriptions = {
        "strike": "超视距打击（BVR 1v1 基线，PPO）",
        "patrol": "空域巡逻（中期检查版：单机拦截近似，PPO）",
        "defense": "战术防御（中期检查版：单机生存/ECM 近似，PPO）",
        "ew": "电子压制（中期检查版：单机 ECM 干扰近似，PPO）",
    }
    for task_id in ("strike", "patrol", "defense", "ew"):
        reg.register(task_id, TASK_AGENTS[task_id], tier=TIER_TASK,
                     description=descriptions[task_id], action_channel="action_v1",
                     aliases=[a for a, t in TASK_ALIASES.items() if t == task_id])
    skill_desc = {
        "cruise_hold": "巡航保持（PPO，3 维 AP 指令）",
        "turn": "盘旋机动（SAC，3 维 AP 指令）",
        "climb_descent": "爬升/下降（TD3，3 维 AP 指令）",
        "pursuit": "接敌追击（DQN 48 离散 → 3 维 AP 指令）",
        "evasion": "威胁规避（PPO，3 维 AP 指令）",
    }
    for skill_id in ("cruise_hold", "turn", "climb_descent", "pursuit", "evasion"):
        reg.register(skill_id, SKILL_AGENTS[skill_id], tier=TIER_SKILL,
                     description=skill_desc[skill_id], action_channel="command_v1",
                     aliases=[a for a, t in SKILL_ALIASES.items() if t == skill_id])
    return reg


DEFAULT_REGISTRY = build_default_registry()

__all__ = ["ExpertAgentRegistry", "AgentRegistration", "build_default_registry",
           "DEFAULT_REGISTRY", "TIER_TASK", "TIER_SKILL", "TASK_ALIASES",
           "SKILL_ALIASES"]
