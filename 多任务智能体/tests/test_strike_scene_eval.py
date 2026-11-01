# -*- coding: utf-8 -*-
"""阶段三 P0/P1 场景与评估冒烟测试（不跑长训练）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.evaluate_strike import evaluate_model, parse_seeds, policy_input
from agents.ppo_agent import make_policy_env
from agents.task_agents.mission_env import scene_config
from common.errors import ValidationError


def test_scene_config_contract():
    assert scene_config("default") == {}
    cfg = scene_config("bvr80-120")
    assert cfg["scene_dist_min_m"] == 80e3
    assert cfg["scene_dist_max_m"] == 120e3
    assert cfg["max_steps"] == 600
    with pytest.raises(ValidationError):
        scene_config("missing")


def test_policy_observation_dims():
    for policy, dim in (("mlp", 28), ("hybrid", 35)):
        env = make_policy_env(policy, "strike", "bvr80-120")
        try:
            obs, _ = env.reset(seed=1)
            assert obs.shape == (dim,)
        finally:
            env.close()


def test_policy_input_shapes():
    obs = np.zeros(28, dtype=np.float32)
    info = {"own_missiles_left": 2, "radar_state": "TRACK", "dist_m": 40000.0,
            "r_min_own_m": 1000.0, "r_max_own_m": 50000.0}
    assert policy_input("mlp", obs, info, None).shape == (28,)
    assert policy_input("hybrid", obs, info, None).shape == (35,)
    assert parse_seeds("1-3,5") == [1, 2, 3, 5]


class _StubModel:
    def predict(self, obs, deterministic=True):
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        dtype=np.float32), None


def test_evaluate_model_smoke(tmp_path):
    agg = evaluate_model(_StubModel(), "hybrid", [1], scene="bvr80-120",
                         deterministic=True, out_dir=tmp_path)
    assert agg["n_episodes"] == 1
    assert agg["hit_rate"] in (0.0, 1.0)
    assert "termination_counts" in agg and "avg_steps" in agg
    assert (tmp_path / "eval_metrics.json").is_file()
    assert (tmp_path / "eval_metrics.csv").is_file()
    data = json.loads((tmp_path / "eval_metrics.json").read_text(encoding="utf-8"))
    assert data["episodes"][0]["situation"]["dist_km"] >= 80.0
