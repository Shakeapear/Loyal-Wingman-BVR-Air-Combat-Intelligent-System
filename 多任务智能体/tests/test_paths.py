# -*- coding: utf-8 -*-
"""路径解析与掩码观测源回归测试（服务器部署相关）。"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.task_agents.mission_env import ScaledObservation  # noqa: E402
from common.paths import (ensure_writable_dir, jsbsim_root,  # noqa: E402
                          model_dir)


def _make_jsbsim_tree(root: Path) -> None:
    f104 = root / "aircraft" / "f104"
    (f104 / "Systems").mkdir(parents=True)
    (f104 / "f104.xml").write_text("<xml/>", encoding="utf-8")
    (f104 / "reset_cruise.xml").write_text("<xml/>", encoding="utf-8")
    (f104 / "Systems" / "autopilot-hold.xml").write_text("<xml/>", encoding="utf-8")


def test_jsbsim_root_override_and_validation(tmp_path, monkeypatch):
    _make_jsbsim_tree(tmp_path)
    monkeypatch.setenv("BVR_JSBSIM_ROOT", str(tmp_path))
    assert jsbsim_root(validate=True) == tmp_path.resolve()


def test_jsbsim_root_missing_files_raises(tmp_path):
    try:
        jsbsim_root(tmp_path, validate=True)
    except RuntimeError as exc:
        assert "BVR_JSBSIM_ROOT" in str(exc)
        assert "f104.xml" in str(exc)
    else:
        raise AssertionError("缺失 JSBSim 文件时应抛 RuntimeError")


def test_model_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("BVR_MODEL_DIR", str(tmp_path / "models_out"))
    assert model_dir() == (tmp_path / "models_out").resolve()


def test_ensure_writable_dir(tmp_path):
    target = ensure_writable_dir(tmp_path / "a" / "b")
    assert target.is_dir()
    assert not list(target.glob(".bvr_write_test_*"))


class _FakeEnv(gym.Env):
    """只用于验证 ScaledObservation 会保存尺度化前的原始观测。"""

    def __init__(self):
        self.observation_space = spaces.Box(
            low=np.full(28, -2000.0, dtype=np.float32),
            high=np.full(28, 2000.0, dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.cfg = {}
        self.raw = np.zeros(28, dtype=np.float32)
        self.raw[0] = 1000.0

    def reset(self, *, seed=None, options=None):
        return self.raw.copy(), {"ok": True}

    def step(self, action):
        return self.raw.copy(), 0.0, False, False, {"ok": True}


def test_masked_wrapper_uses_raw_observation(monkeypatch):
    from agents import hybrid_policy

    seen = {}

    def fake_mask(obs, info, env_cfg):
        seen["obs"] = np.asarray(obs, dtype=np.float32).copy()
        return np.ones(7, dtype=bool)

    monkeypatch.setattr(hybrid_policy, "compute_action_mask", fake_mask)
    wrapped = hybrid_policy.MaskedObsWrapper(ScaledObservation(_FakeEnv()))
    obs, _ = wrapped.reset(seed=0)
    assert obs.shape == (35,)
    # 掩码必须基于原始物理量 1000，而不是尺度化后的 ~[-1,1] 值。
    assert seen["obs"][0] == 1000.0
