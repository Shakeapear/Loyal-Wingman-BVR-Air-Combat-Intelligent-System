# -*- coding: utf-8 -*-
"""BT vs BT 观测契约冒烟测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))
sys.path.append(str(_ROOT.parent / "可视化工具"))

from common.constants import wrap_pi
from llm_bt.tactical_execution_tree import tactical_tree_json
from run_bt_vs_bt import BTVersusBTEnv


def test_enemy_obs_matches_obs_v1_contract():
    env = BTVersusBTEnv(
        tactical_tree_json(),
        config={"compute_envelope": False, "max_steps": 2,
                "n_missiles": 0, "enemy_fire_interval_s": 1e9},
    )
    try:
        env.reset(seed=1)
        obs, info = env._enemy_obs_info()
        state = env.enemy.state_dict()

        assert obs.shape == (28,) and obs.dtype == np.float32
        assert np.isclose(obs[0], state["h_sl_m"], atol=1.0)
        assert np.isclose(obs[1], state["vtrue_mps"], atol=1.0)
        assert np.isclose(obs[2], state["mach"], atol=1e-3)
        assert np.isclose(obs[3], state["phi_rad"], atol=1e-4)
        assert np.isclose(obs[4], state["theta_rad"], atol=1e-4)
        assert np.isclose(obs[5], wrap_pi(state["psi_rad"]), atol=1e-4)
        assert obs[12] == env.enemy_n_left
        assert obs[21] in (0.0, 1.0)
        assert obs[22] in (0.0, 1.0)
        assert obs[24] in (0.0, 1.0)
        assert set(info) >= {"in_zone", "maws_alarm", "rwr_alarm"}
    finally:
        env.close()
