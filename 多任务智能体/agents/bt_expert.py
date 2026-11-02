# -*- coding: utf-8 -*-
"""
agents/bt_expert.py
===================
行为树 → 任务级专家智能体调度桥（阶段三中期检查版）。

结构（py_trees）：
    Selector(memory=False)
      ├── Sequence(MissionIs["strike"], ExpertAgentControl["strike"])
      ├── Sequence(MissionIs["patrol"], ExpertAgentControl["patrol"])
      ├── Sequence(MissionIs["defense"], ExpertAgentControl["defense"])
      └── Sequence(MissionIs["ew"], ExpertAgentControl["ew"])

行为树按黑板 `mission`（英文任务名或中文别名）选择专家智能体，叶节点调用
`get_action(obs)` 得到 ActionV1，再经 `to_env_action` 转为 BVRCombatEnv 的
Dict 动作下发。本轮不改造现有 LLM 行为树 schema，作为阶段三独立调度通道，
后续可把本叶节点并入 `引导策略行为树/llm_bt` 动作目录。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import py_trees

from agents.action_adapter import to_env_action
from agents.expert_registry import DEFAULT_REGISTRY, ExpertAgentRegistry
from agents.task_agents.base_task_agent import MODEL_DIR

STATUS = py_trees.common.Status


@dataclass
class MissionBlackboard:
    """行为树黑板（最小版）：当前任务名 + 每任务执行结果。"""

    mission: str | None = None
    results: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)


class MissionIs(py_trees.behaviour.Behaviour):
    """条件节点：黑板 mission 是否等于指定任务（支持中文别名）。"""

    def __init__(self, task_id: str, blackboard: MissionBlackboard,
                 registry: ExpertAgentRegistry | None = None, name: str | None = None):
        super().__init__(name or f"MissionIs[{task_id}]")
        self.registry = registry or DEFAULT_REGISTRY
        self.task_id = self.registry.resolve(task_id)
        self.blackboard = blackboard

    def update(self):
        if self.blackboard.mission is None:
            return STATUS.FAILURE
        try:
            current = self.registry.resolve(self.blackboard.mission)
        except KeyError:
            return STATUS.FAILURE
        return STATUS.SUCCESS if current == self.task_id else STATUS.FAILURE


class ExpertAgentControl(py_trees.behaviour.Behaviour):
    """叶节点：按任务名创建任务级专家智能体，逐步推理并下发 ActionV1。"""

    def __init__(self, task_id: str, blackboard: MissionBlackboard,
                 registry: ExpertAgentRegistry | None = None, max_steps: int = 24,
                 seed: int = 7, deterministic: bool = True,
                 model_path: str | Path | None = None, load_model: bool = False,
                 env=None, name: str | None = None):
        super().__init__(name or f"ExpertAgent[{task_id}]")
        self.registry = registry or DEFAULT_REGISTRY
        self.task_id = self.registry.resolve(task_id)
        self.blackboard = blackboard
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.model_path = Path(model_path) if model_path else None
        self.load_model = bool(load_model)
        self._external_env = env
        self.agent = None
        self.env = None
        self.obs = None
        self.info = {}
        self.records: list[dict] = []
        self.total_reward = 0.0
        self._closed = False
        self._finished = False

    def _resolved_model_path(self):
        if self.model_path is None:
            return None
        if self.model_path.exists():
            return self.model_path
        if self.model_path.suffix != ".zip":
            zipped = self.model_path.with_suffix(".zip")
            if zipped.exists():
                return zipped
        return self.model_path

    def initialise(self):
        if self.model_path is not None:
            self.agent = self.registry.create(self.task_id, model_path=self.model_path)
        else:
            self.agent = self.registry.create(self.task_id)
        model_path = self._resolved_model_path()
        if self.load_model and model_path is not None and model_path.exists():
            self.agent.load(model_path)
        self.env = self._external_env or self.agent.build_deploy_env()
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.agent.reset()
        self.records = []
        self.total_reward = 0.0
        self._closed = False
        self._finished = False
        self.blackboard.logs.append({
            "event": "mission_start", "task_id": self.task_id,
            "name": self.agent.action_spec().get("mission_name_zh", self.task_id),
            "seed": self.seed,
        })

    def update(self):
        out = self.agent.get_action(self.obs, info=self.info,
                                    deterministic=self.deterministic)
        env_action = to_env_action(out.action)
        self.obs, reward, terminated, truncated, self.info = self.env.step(env_action)
        self.total_reward += float(reward)
        self.records.append({
            "step": len(self.records),
            "task_id": self.task_id,
            "action_v1": out.action.tolist(),
            "source": out.source,
            "masked": out.masked.tolist() if out.masked is not None else None,
            "reward": float(reward),
            "dist_m": self.info.get("dist_m"),
            "radar_state": self.info.get("radar_state"),
            "in_zone": bool(self.info.get("in_zone", False)),
            "terminated_reason": self.info.get("terminated_reason"),
        })
        if terminated or truncated or len(self.records) >= self.max_steps:
            return STATUS.SUCCESS
        return STATUS.RUNNING

    def terminate(self, new_status):
        if not self.records or self._finished:
            return
        self._finished = True
        summary = {
            "task_id": self.task_id,
            "steps": len(self.records),
            "total_reward": float(self.total_reward),
            "terminated_reason": self.info.get("terminated_reason"),
            "records": list(self.records),
            "model_loaded": bool(self.load_model
                                 and self._resolved_model_path() is not None
                                 and self._resolved_model_path().exists()),
            "fallback_count": sum(1 for r in self.records
                                  if r.get("source") == "fallback"),
        }
        self.blackboard.results[self.task_id] = summary
        self.blackboard.logs.append({
            "event": "mission_end", "task_id": self.task_id,
            "steps": summary["steps"], "total_reward": summary["total_reward"],
            "terminated_reason": summary["terminated_reason"],
        })
        if not self._closed:
            if self._external_env is None and self.env is not None:
                self.env.close()
            if self.agent is not None:
                self.agent.close()
            self._closed = True


def build_expert_tree(registry: ExpertAgentRegistry | None = None,
                      blackboard: MissionBlackboard | None = None,
                      tasks=("strike", "patrol", "defense", "ew"),
                      max_steps: int = 24, seed: int = 7,
                      load_models: bool = False,
                      model_dir: str | Path | None = None,
                      env_factory=None):
    """构建任务选择行为树；返回 (root, blackboard)。

    env_factory(task_id) 可选，用于测试/自定义部署环境；缺省由叶节点创建
    TaskMissionEnv(Dict 动作，配合 to_env_action)。
    """
    registry = registry or DEFAULT_REGISTRY
    blackboard = blackboard or MissionBlackboard()
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    branches = []
    for index, task in enumerate(tasks):
        task_id = registry.resolve(task)
        env = env_factory(task_id) if callable(env_factory) else None
        branch = py_trees.composites.Sequence(
            name=f"Mission[{task_id}]",
            children=[
                MissionIs(task_id, blackboard, registry),
                ExpertAgentControl(
                    task_id, blackboard, registry, max_steps=max_steps,
                    seed=seed + index,
                    model_path=model_dir / f"task_{task_id}_ppo",
                    load_model=load_models,
                    env=env,
                ),
            ],
            memory=False,
        )
        branches.append(branch)
    root = py_trees.composites.Selector(name="ExpertMissionSelector",
                                        children=branches, memory=False)
    return root, blackboard


def run_missions(root, blackboard: MissionBlackboard, missions,
                 max_ticks_per_mission: int | None = None,
                 registry: ExpertAgentRegistry | None = None) -> dict:
    """按任务名顺序驱动行为树，执行到每局叶节点 SUCCESS/FAILURE。"""
    registry = registry or DEFAULT_REGISTRY
    results = {}
    for mission in missions:
        blackboard.mission = mission
        limit = max_ticks_per_mission or 1000
        for _ in range(limit):
            root.tick_once()          # py_trees 1.x：tick() 为生成器，用 tick_once + root.status
            if root.status != STATUS.RUNNING:
                break
        task_id = registry.resolve(mission)
        results[task_id] = {
            "status": root.status.name,
            "result": blackboard.results.get(task_id),
        }
    return results


def records_to_csv(records, path: str | Path) -> Path:
    """把行为树记录写成 UTF-8 CSV（无 pandas 依赖）。"""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "task_id", "action_v1", "source", "masked", "reward",
              "dist_m", "radar_state", "in_zone", "terminated_reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            row = dict(row)
            row["action_v1"] = ";".join(f"{v:.6f}" for v in row["action_v1"])
            row["masked"] = (";" .join("1" if v else "0" for v in row["masked"])
                             if row["masked"] is not None else "")
            writer.writerow(row)
    return path


__all__ = ["MissionBlackboard", "MissionIs", "ExpertAgentControl",
           "build_expert_tree", "run_missions", "records_to_csv", "STATUS"]
