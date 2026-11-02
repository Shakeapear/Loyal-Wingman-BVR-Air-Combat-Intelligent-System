# -*- coding: utf-8 -*-
"""模型归档规格校验回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.model_spec import (load_model_spec, save_model_spec,  # noqa: E402
                               validate_model_artifact)


class HybridActorCriticPolicy:
    pass


def _model(dim: int, policy_cls):
    class _Model:
        def __init__(self):
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
            self.policy = policy_cls()

    return _Model()


def test_spec_roundtrip(tmp_path):
    path = save_model_spec(tmp_path / "model.zip",
                           {"task_id": "strike", "policy": "hybrid",
                            "scene": "bvr80-120",
                            "observation": {"dim": 35}})
    assert path.is_file()
    assert load_model_spec(tmp_path / "model.zip")["scene"] == "bvr80-120"


def test_matching_spec_passes(tmp_path):
    model_path = tmp_path / "hybrid.zip"
    save_model_spec(model_path, {"task_id": "strike", "policy": "hybrid",
                                 "scene": "bvr80-120",
                                 "observation": {"dim": 35}})
    validate_model_artifact(_model(35, HybridActorCriticPolicy),
                            policy="hybrid", task="strike",
                            scene="bvr80-120", path=model_path)


def test_policy_dimension_mismatch_raises():
    try:
        validate_model_artifact(_model(28, HybridActorCriticPolicy),
                                policy="hybrid")
    except ValueError as exc:
        assert "观测维度" in str(exc)
    else:
        raise AssertionError("hybrid 配 28 维模型应失败")


def test_spec_task_scene_mismatch_raises(tmp_path):
    model_path = tmp_path / "hybrid.zip"
    save_model_spec(model_path, {"task_id": "strike", "policy": "hybrid",
                                 "scene": "bvr80-120",
                                 "observation": {"dim": 35}})
    for kwargs, needle in (
        ({"task": "patrol"}, "任务不匹配"),
        ({"task": "strike", "scene": "default"}, "场景不匹配"),
    ):
        try:
            validate_model_artifact(_model(35, HybridActorCriticPolicy),
                                    policy="hybrid", path=model_path, **kwargs)
        except ValueError as exc:
            assert needle in str(exc)
        else:
            raise AssertionError(f"应触发 {needle}")
