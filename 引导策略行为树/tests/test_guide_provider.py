# -*- coding: utf-8 -*-
"""统一 GuideStep / BTGuideProvider / 控制模式适配回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))

import numpy as np
import pytest

from common.bvr_combat_env import BVRCombatEnv
from llm_bt.executor import Blackboard
from llm_bt.guide_adapter import (AssistedAdapter, RawFCSAdapter,
                                  action_dict_to_v1)
from llm_bt.guide_provider import BTGuideProvider
from llm_bt.guide_types import GuideStep
from llm_bt.tactical_execution_tree import (tactical_tree_json,
                                            tactical_tree_v3_json)


def _obs(**kw):
    o = np.zeros(28, dtype=np.float64)
    o[0], o[1], o[2] = 6000.0, 240.0, 0.8
    o[9], o[10], o[11], o[12] = 0.55, 1.0, 0.0, 2
    for key, value in kw.items():
        o[int(key[1:])] = value
    return o


def test_guide_step_validation_and_serialization():
    step = GuideStep(action=np.array([0.5, 0, 0, 0, 0, 0, 0]),
                     active=True, source="bt", guide_step=3,
                     control_mode="assisted",
                     mask=np.ones(7, dtype=bool), meta={"tree": "t"})
    payload = step.to_dict()
    assert payload["source"] == "bt"
    assert payload["action"][0] == 0.5
    with pytest.raises(ValueError):
        GuideStep(action=np.zeros(6), active=True)
    with pytest.raises(ValueError):
        GuideStep(action=None, active=True)
    with pytest.raises(ValueError):
        GuideStep(action=np.zeros(7), active=False, control_mode="bad")


def test_adapter_mode_guard():
    bb = Blackboard(control_mode="raw_fcs")
    with pytest.raises(ValueError):
        AssistedAdapter().to_action_v1(
            {"flight": [0.5, 0, 0, 0], "weapon": [0, 0, 0]}, bb)
    action = RawFCSAdapter().to_action_v1(
        {"flight": [0.5, 0, 0, 0], "weapon": [1, 0, 0]}, bb)
    assert action.shape == (7,)
    assert action[4] == 1.0


def test_bt_provider_assisted_exhaust_and_reset():
    provider = BTGuideProvider(max_guide_steps=2, control_mode="assisted")
    obs = _obs()
    provider.reset(obs=obs, info={})
    first = provider.step(obs, {}, step=0)
    second = provider.step(obs, {}, step=1)
    third = provider.step(obs, {}, step=2)
    assert first.active and first.action is not None
    assert first.control_mode == "assisted"
    assert first.action.shape == (7,)
    assert second.active
    assert not third.active and third.action is None
    provider.reset(obs=obs, info={})
    assert provider.step(obs, {}, step=0).active


def test_bt_provider_with_v3_teacher_tree():
    provider = BTGuideProvider(
        tree=tactical_tree_v3_json("break_turn_defense"),
        max_guide_steps=2, control_mode="assisted")
    obs = _obs()
    provider.reset(obs=obs, info={})
    step = provider.step(obs, {}, step=0)
    assert step.active and step.action.shape == (7,)
    assert step.meta["tree_id"] == "bt_tactical_v3_break_turn_defense"


def test_bt_provider_raw_mode_and_bounds():
    provider = BTGuideProvider(max_guide_steps=1, control_mode="raw_fcs")
    obs = _obs()
    step = provider.step(obs, {}, step=0)
    assert step.active
    assert np.all(step.action >= 0.0)
    assert np.all(step.action <= 1.0)


def test_bt_provider_rejects_35_dim_policy_obs():
    provider = BTGuideProvider(max_guide_steps=1)
    with pytest.raises(ValueError, match="28"):
        provider.step(np.zeros(35, dtype=np.float32), {})


def test_bt_provider_rejects_control_mode_mismatch():
    class FakeEnv:
        control_mode = "raw_fcs"
        cfg = {}

    with pytest.raises(ValueError):
        BTGuideProvider(env=FakeEnv(), control_mode="assisted")


def test_action_dict_conversion_binarizes_edges():
    out = action_dict_to_v1(
        {"flight": [0.7, 0.2, -0.3, 0.0], "weapon": [0.6, 0.4, 0.9]},
        control_mode="assisted")
    assert out[4] == 1.0 and out[5] == 0.0 and out[6] == 1.0


def test_provider_runs_with_assisted_env():
    env = BVRCombatEnv(config={"control_mode": "assisted",
                               "compute_envelope": False,
                               "n_missiles": 0,
                               "enemy_fire_interval_s": 1e9})
    provider = BTGuideProvider(env=env, control_mode="assisted",
                               max_guide_steps=3)
    try:
        obs, info = env.reset(seed=0)
        provider.reset(obs, info)
        for i in range(3):
            step = provider.step(obs, info, i)
            assert step.active and step.action.shape == (7,)
            action = {"flight": step.action[:4],
                      "weapon": step.action[4:].astype(int)}
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
    finally:
        env.close()
