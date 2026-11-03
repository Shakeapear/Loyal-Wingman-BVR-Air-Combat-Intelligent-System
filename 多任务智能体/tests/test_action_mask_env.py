# -*- coding: utf-8 -*-
"""P0 可选动作掩码基线回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.task_agents.mission_env import FlatTaskMissionEnv  # noqa: E402


def test_mask_actions_blocks_out_of_zone_fire():
    env = FlatTaskMissionEnv(
        mission="strike",
        config={"compute_envelope": False, "mask_actions": True,
                "n_missiles": 2, "enemy_fire_interval_s": 1e9})
    try:
        env.reset(seed=0)
        action = np.array([0.55, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["own_fired"] == 0, "未跟踪/不在攻击区时 fire 应被掩码清零"
    finally:
        env.close()
