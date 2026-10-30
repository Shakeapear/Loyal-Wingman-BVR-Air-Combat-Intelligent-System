# -*- coding: utf-8 -*-
"""
多任务智能体/tests/test_expert_agents.py（阶段三专家智能体底座测试）
==================================================================
覆盖：
- ActionV1/ActionMaskV1/PolicyOutput/action_spec；
- action_adapter（to_env_action/to_flat_action/to_ap_command/边界）；
- ExpertAgentRegistry（4 任务级 + 5 技能级、中文别名、tier）；
- 4 个任务级智能体实例化/get_action/predict/小规模训练/保存加载一致；
- 任务环境标签/奖励/动作空间；
- 行为树按任务名选择 4 个专家智能体（假环境，不启动 JSBSim）；
- 技能级智能体 command_v1 兼容通道可调用（抽查巡航保持）。

运行：python -m pytest tests/test_expert_agents.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.action_adapter import (command_v1, coerce_action_v1, from_env_action,
                                   to_ap_command, to_env_action, to_flat_action)
from agents.bt_expert import build_expert_tree, run_missions
from agents.expert_registry import (DEFAULT_REGISTRY, ExpertAgentRegistry,
                                    TASK_ALIASES, TIER_SKILL, TIER_TASK)
from agents.policy_interface import (ACTION_V1_DIM, ACTION_V1_HIGH, ACTION_V1_LOW,
                                     PolicyOutput, PolicyProtocol, action_spec,
                                     apply_action_mask,
                                     command_v1_spec, compute_action_mask)
from agents.task_agents import (TASK_AGENTS, FlatTaskMissionEnv, MISSION_IDS,
                                MISSION_LABELS, TaskMissionEnv)

TASKS = list(TASK_AGENTS)


# ---------------------------------------------------------------------------
# 接口/适配层
# ---------------------------------------------------------------------------
def test_action_spec_and_command_spec():
    for task in TASKS:
        spec = action_spec(task)
        assert spec["version"] == "action_v1"
        assert spec["dim"] == ACTION_V1_DIM == 7
        assert spec["dtype"] == "float32"
        assert spec["fields"] == ["throttle", "roll", "pitch", "yaw",
                                  "fire", "switch", "ecm"]
        assert spec["low"] == ACTION_V1_LOW.tolist()
        assert spec["high"] == ACTION_V1_HIGH.tolist()
        assert spec["edge_dims"] == [4, 5, 6]
    cmd = command_v1_spec()
    assert cmd["version"] == "command_v1" and cmd["dim"] == 3


def test_action_adapter_threshold_and_bounds():
    raw = np.array([1.4, -1.8, 0.3, 0.9, 0.51, 0.49, 1.0], dtype=np.float32)
    a7 = coerce_action_v1(raw)
    assert a7.shape == (7,) and a7.dtype == np.float32
    assert np.all(a7 >= ACTION_V1_LOW) and np.all(a7 <= ACTION_V1_HIGH)
    env_action = to_env_action(a7)
    assert env_action["flight"].shape == (4,)
    assert env_action["weapon"].tolist() == [1, 0, 1]
    assert env_action["weapon"].dtype == np.int64
    assert np.allclose(to_flat_action(a7), a7)
    assert np.allclose(from_env_action(env_action), a7)
    assert np.allclose(to_env_action({"flight": [0.2, 0, 0, 0],
                                      "weapon": [1, 0, 0]})["flight"],
                       [0.2, 0, 0, 0])


def test_to_ap_command_ranges_and_sign():
    obs = np.zeros(28, dtype=np.float32)
    obs[0], obs[1] = 5000.0, 230.0
    a7 = np.array([0.55, 0.5, 1.0, 0.0, 0, 0, 0], dtype=np.float32)
    cmd = to_ap_command(a7, obs)
    assert cmd.shape == (3,) and cmd.dtype == np.float32
    assert 180.0 <= cmd[0] <= 280.0
    assert -0.90 <= cmd[1] <= 0.90
    assert 3000.0 <= cmd[2] <= 8000.0
    assert np.isclose(cmd[0], 230.0)
    assert np.isclose(cmd[1], 0.45)
    assert np.isclose(cmd[2], 5050.0)
    assert np.allclose(command_v1(a7, obs), cmd)
    full = to_ap_command(np.array([1, 1, 1, 1, 1, 1, 1], np.float32), obs)
    idle = to_ap_command(np.array([0, -1, -1, -1, 0, 0, 0], np.float32), obs)
    assert np.isclose(full[0], 280.0) and np.isclose(idle[0], 180.0)


def test_compute_action_mask_matches_fire_conditions():
    obs = np.zeros(28, dtype=np.float32)
    obs[12] = 2.0
    obs[21] = 1.0
    obs[17] = 20000.0
    base = {"own_missiles_left": 2, "radar_state": "TRACK", "dist_m": 20000.0,
            "r_min_own_m": 1000.0, "r_max_own_m": 45000.0}
    mask = compute_action_mask(obs, base)
    assert mask.shape == (7,) and mask.dtype == bool
    assert mask.tolist() == [True, True, True, True, True, False, True]
    assert not compute_action_mask(obs, {**base, "own_missiles_left": 0})[4]
    assert not compute_action_mask(obs, {**base, "radar_state": "LOST"})[4]
    assert not compute_action_mask(obs, {**base, "dist_m": 50000.0})[4]
    assert not compute_action_mask(obs, {})[4]
    masked = apply_action_mask(np.ones(7, np.float32), mask)
    assert masked[5] == 0.0 and masked[4] == 1.0


def test_policy_output_roundtrip():
    out = PolicyOutput(np.ones(7, np.float32), source="guide",
                       masked=np.ones(7, bool), task_id="strike")
    d = out.to_dict()
    assert d["action_version"] == "action_v1"
    assert d["source"] == "guide" and d["task_id"] == "strike"
    assert len(d["action"]) == 7


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
def test_registry_default_nine_agents():
    assert len(DEFAULT_REGISTRY) == 9
    assert set(DEFAULT_REGISTRY.task_ids()) == set(TASKS)
    assert set(DEFAULT_REGISTRY.skill_ids()) == {
        "cruise_hold", "turn", "climb_descent", "pursuit", "evasion"}
    assert DEFAULT_REGISTRY.get("strike").tier == TIER_TASK
    assert DEFAULT_REGISTRY.get("cruise_hold").tier == TIER_SKILL
    assert DEFAULT_REGISTRY.get("strike").action_channel == "action_v1"
    assert DEFAULT_REGISTRY.get("cruise_hold").action_channel == "command_v1"
    assert DEFAULT_REGISTRY.resolve("超视距打击") == "strike"
    assert DEFAULT_REGISTRY.resolve("空域巡逻") == "patrol"
    assert DEFAULT_REGISTRY.resolve("战术防御") == "defense"
    assert DEFAULT_REGISTRY.resolve("电子压制") == "ew"
    assert DEFAULT_REGISTRY.resolve("威胁规避") == "evasion"
    assert isinstance(DEFAULT_REGISTRY.create("strike"), TASK_AGENTS["strike"])
    assert "不存在的任务" not in DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# 任务环境
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mission", MISSION_IDS)
def test_flat_mission_env_label_and_step(mission):
    env = FlatTaskMissionEnv(mission=mission,
                             config={"max_steps": 2, "compute_envelope": False})
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == (28,) and obs.dtype == np.float32
        assert (obs[26], obs[27]) == MISSION_LABELS[mission]
        assert np.allclose(env.action_space.low, ACTION_V1_LOW)
        assert np.allclose(env.action_space.high, ACTION_V1_HIGH)
        obs2, reward, terminated, truncated, info2 = env.step(
            np.array([0.55, 0, 0, 0, 0, 0, 0], np.float32))
        assert obs2.shape == (28,)
        assert np.isfinite(reward)
        assert info2["mission_id"] == mission
        assert np.isfinite(info2["mission_reward"])
    finally:
        env.close()


def test_dict_mission_env_accepts_to_env_action():
    env = TaskMissionEnv(mission="patrol",
                         config={"max_steps": 2, "compute_envelope": False})
    try:
        obs, info = env.reset(seed=1)
        out = PolicyOutput(np.array([0.55, 0, 0, 0, 0, 0, 0], np.float32))
        obs2, reward, terminated, truncated, info2 = env.step(to_env_action(out.action))
        assert obs2.shape == (28,) and np.isfinite(reward)
        assert (obs2[26], obs2[27]) == MISSION_LABELS["patrol"]
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 任务级智能体
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task_id", TASKS)
def test_task_agent_instantiate_and_action_interface(task_id):
    cls = TASK_AGENTS[task_id]
    agent = cls(env_config={"max_steps": 2, "compute_envelope": False},
                train_config={"n_steps": 16, "batch_size": 8, "n_epochs": 1},
                seed=42)
    try:
        assert isinstance(agent, PolicyProtocol)
        assert agent.action_spec()["task_id"] == task_id
        obs, info = agent.build_env().reset(seed=42)
        out = agent.get_action(obs, info=info, deterministic=True)
        assert out.action.shape == (7,) and out.action.dtype == np.float32
        assert np.all(out.action >= ACTION_V1_LOW)
        assert np.all(out.action <= ACTION_V1_HIGH)
        assert out.source == "drl" and out.task_id == task_id
        assert out.masked.shape == (7,)
        pred = agent.predict(obs, deterministic=True)
        assert pred.shape == (7,) and pred.dtype == np.float32
    finally:
        agent.close()


def test_guide_action_passthrough_and_mask():
    agent = TASK_AGENTS["strike"](env_config={"max_steps": 2,
                                              "compute_envelope": False})
    try:
        obs = np.zeros(28, np.float32)
        obs[12] = 2.0
        info = {"own_missiles_left": 2, "radar_state": "TRACK", "dist_m": 50000.0,
                "r_min_own_m": 1000.0, "r_max_own_m": 45000.0}
        guide = np.array([0.7, 0.2, -0.1, 0.0, 1, 1, 1], np.float32)
        out = agent.get_action(obs, info=info, guide_action=guide, guide_steps=5)
        assert out.source == "guide"
        assert np.allclose(out.action, [0.7, 0.2, -0.1, 0.0, 0.0, 0.0, 1.0])
        drl = agent.get_action(obs, info=info, guide_action=guide, guide_steps=0)
        assert drl.source == "drl"
    finally:
        agent.close()


@pytest.mark.parametrize("task_id", TASKS)
def test_task_agent_tiny_train_save_load(task_id, tmp_path):
    cls = TASK_AGENTS[task_id]
    kwargs = dict(env_config={"max_steps": 16, "compute_envelope": False},
                  train_config={"n_steps": 16, "batch_size": 8, "n_epochs": 1},
                  seed=42)
    agent = cls(**kwargs)
    try:
        train = agent.train(timesteps=32)
        assert train["timesteps"] == 32
        assert train["all_finite"] is True
        assert any("loss" in k for k in train["metrics"])
        stats = agent.evaluate(n_steps=4)
        assert stats["action_shape"] == [4, 7]
        assert stats["action_dtype"] == "float32"
        assert stats["reward_finite"] is True
        obs, info = agent.build_env().reset(seed=7)
        before = agent.get_action(obs, info=info, deterministic=True).action.copy()
        path = agent.save(tmp_path / f"{task_id}.zip")
        loaded = cls(**kwargs)
        loaded.load(path)
        after = loaded.get_action(obs, info=info, deterministic=True).action
        assert np.allclose(before, after, atol=1e-6)
        assert np.all(after >= ACTION_V1_LOW) and np.all(after <= ACTION_V1_HIGH)
        loaded.close()
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# 行为树调度（假环境，验证按任务名选择 4 个专家）
# ---------------------------------------------------------------------------
class _FakeEnv:
    def __init__(self):
        self.steps = 0
        self.obs = np.zeros(28, np.float32)
        self.info = {"steps": 0, "dist_m": 20000.0, "radar_state": "TRACK",
                     "r_min_own_m": 1000.0, "r_max_own_m": 45000.0,
                     "own_missiles_left": 2, "ecm_on": False, "in_zone": False,
                     "rwr_alarm": False, "maws_alarm": False,
                     "enemy_radar_state": "SEARCH", "terminated_reason": None}

    def reset(self, seed=None):
        self.steps = 0
        return self.obs.copy(), dict(self.info)

    def step(self, action):
        assert set(action) == {"flight", "weapon"}
        assert np.asarray(action["flight"]).shape == (4,)
        assert np.asarray(action["weapon"]).shape == (3,)
        self.steps += 1
        info = dict(self.info)
        info["steps"] = self.steps
        return self.obs.copy(), 1.0, False, False, info

    def close(self):
        pass


class _FakeAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def action_spec(self):
        return {"mission_name_zh": "fake", "version": "action_v1"}

    def reset(self):
        pass

    def get_action(self, obs, **kwargs):
        return PolicyOutput(np.array([0.55, 0, 0, 0, 0, 0, 0], np.float32),
                            source="drl", masked=np.ones(7, bool))

    def close(self):
        pass


def test_bt_selects_all_four_task_agents_by_name():
    reg = ExpertAgentRegistry()
    for task_id in TASKS:
        reg.register(task_id, _FakeAgent,
                     aliases=[a for a, t in TASK_ALIASES.items() if t == task_id])
    root, bb = build_expert_tree(registry=reg, tasks=tuple(TASKS), max_steps=2,
                                 env_factory=lambda _task: _FakeEnv())
    results = run_missions(root, bb, ["超视距打击", "空域巡逻", "战术防御", "电子压制"],
                           max_ticks_per_mission=8, registry=reg)
    assert set(bb.results) == set(TASKS)
    for task_id in TASKS:
        assert results[task_id]["status"] == "SUCCESS"
        row = bb.results[task_id]
        assert row["steps"] == 2
        assert len(row["records"]) == 2
        assert len(row["records"][0]["action_v1"]) == 7


# ---------------------------------------------------------------------------
# 技能级兼容通道
# ---------------------------------------------------------------------------
def test_skill_agent_command_v1_channel():
    agent = DEFAULT_REGISTRY.create("巡航保持")
    try:
        agent.build_model()
        obs, _ = agent.reset_env()
        pred = agent.predict(obs, deterministic=True)
        assert np.asarray(pred).shape == (3,)
        assert np.all(np.isfinite(np.asarray(pred)))
        cmd = to_ap_command(np.array([0.55, 0, 0, 0, 0, 0, 0], np.float32), obs)
        assert cmd.shape == (3,) and np.all(np.isfinite(cmd))
    finally:
        if agent.env is not None:
            agent.env.close()
