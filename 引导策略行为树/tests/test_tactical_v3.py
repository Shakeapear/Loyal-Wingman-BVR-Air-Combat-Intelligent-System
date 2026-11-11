# -*- coding: utf-8 -*-
"""v3 对抗教师树与新派生条件测试（不依赖 JSBSim）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))

import numpy as np
import pytest

from llm_bt.executor import (Blackboard, COND_FUNCS, LAR_CONFIG, build_tree,
                             configure_lar, tick_once)
from llm_bt.maneuvers import ManeuverRegistry
from llm_bt.schema import OODA_ALL, ooda_coverage, validate_tree_json
from llm_bt.tactical_execution_tree import V3_PROFILES, tactical_tree_v3_json


class FakeEnv:
    def __init__(self, pos=(0.0, 0.0, 6000.0), missiles=None, cfg=None):
        self.pos = list(pos)
        self.missiles = list(missiles or [])
        self.cfg = cfg or {"escape_radius_m": 150000.0}

    def get_viz_frame(self):
        return {"own": {"pos": list(self.pos)},
                "own_missiles": list(self.missiles)}


def _obs(*, dist=30000.0, bearing=0.0, heading=math.pi, detected=1.0,
         psi=0.0, n_msl=2.0, fuel=0.8, h=6000.0, v=250.0, mach=0.8, nz=1.0):
    o = np.zeros(28, dtype=np.float64)
    o[0], o[1], o[2] = h, v, mach
    o[5], o[8], o[10], o[12] = psi, nz, fuel, n_msl
    o[17], o[18], o[20], o[21] = dist, bearing, heading, detected
    return o


def _info(**kw):
    base = {"in_zone": True, "r_max_own_m": 40000.0, "r_nez_own_m": 20000.0,
            "r_max_enemy_m": 40000.0, "r_min_enemy_m": 1000.0,
            "maws_alarm": False, "maws_tta_s": 0.0, "rwr_alarm": False}
    base.update(kw)
    return base


@pytest.fixture(autouse=True)
def _reset_lar():
    configure_lar(aspect_max_deg=60.0, closure_min_mps=400.0,
                  range_factor_min=0.70, max_range_m=45000.0)
    yield
    configure_lar(aspect_max_deg=60.0, closure_min_mps=400.0,
                  range_factor_min=0.70, max_range_m=45000.0)


class TestDerivedConditions:
    def test_target_aspect_hot_and_beam(self):
        bb = Blackboard(env=FakeEnv())
        bb.set_frame(_obs(heading=math.pi), _info(), 0)   # 迎面
        assert COND_FUNCS["is_target_hot"](bb) is True
        bb.set_frame(_obs(heading=math.pi / 2), _info(), 1)  # 90° beam
        assert COND_FUNCS["is_target_hot"](bb) is False

    def test_closure_high_and_opening(self):
        bb = Blackboard(env=FakeEnv())
        bb.set_frame(_obs(dist=30400.0), _info(), 0)
        bb.set_frame(_obs(dist=30000.0), _info(), 1)   # 400 m/s 闭合
        assert COND_FUNCS["is_closure_high"](bb) is True
        assert COND_FUNCS["is_opening"](bb) is False
        bb.set_frame(_obs(dist=30600.0), _info(), 2)   # 拉开 600 m/s
        assert COND_FUNCS["is_opening"](bb) is True

    def test_dynamic_lar_hot_vs_beam(self):
        bb = Blackboard(env=FakeEnv())
        bb.set_frame(_obs(dist=30400.0, heading=math.pi), _info(), 0)
        bb.set_frame(_obs(dist=30000.0, heading=math.pi), _info(), 1)
        assert COND_FUNCS["is_within_dynamic_lar"](bb) is True
        bb.set_frame(_obs(dist=33000.0, heading=math.pi / 2), _info(), 2)
        assert COND_FUNCS["is_within_dynamic_lar"](bb) is False

    def test_target_turning_and_shot_window(self):
        bb = Blackboard(env=FakeEnv())
        bb.set_frame(_obs(heading=0.0), _info(), 0)
        bb.set_frame(_obs(heading=math.radians(5.0)), _info(), 1)
        assert COND_FUNCS["is_target_turning"](bb) is True
        assert COND_FUNCS["is_shot_window_open"](bb) is False
        bb.set_frame(_obs(heading=math.radians(5.0)), _info(), 2)
        assert COND_FUNCS["is_target_turning"](bb) is False
        assert COND_FUNCS["is_shot_window_open"](bb) is True
        bb.set_frame(_obs(heading=math.radians(5.0)),
                     _info(maws_alarm=True, maws_tta_s=6.0), 3)
        assert COND_FUNCS["is_shot_window_open"](bb) is False

    def test_full_weapons(self):
        bb = Blackboard(env=FakeEnv())
        bb.set_frame(_obs(n_msl=2.0), _info(), 0)
        assert COND_FUNCS["has_full_weapons"](bb) is True
        bb.set_frame(_obs(n_msl=1.0), _info(), 1)
        assert COND_FUNCS["has_full_weapons"](bb) is False

    def test_boundary_missile_rwr_tta(self):
        env = FakeEnv(pos=(130000.0, 0.0, 6000.0), missiles=[object()],
                      cfg={"escape_radius_m": 150000.0})
        bb = Blackboard(env=env)
        bb.set_frame(_obs(), _info(rwr_alarm=True, maws_alarm=True,
                                   maws_tta_s=12.0), 0)
        assert COND_FUNCS["is_near_boundary"](bb) is True
        assert COND_FUNCS["has_missile_in_flight"](bb) is True
        assert COND_FUNCS["is_rwr_locked"](bb) is True
        assert COND_FUNCS["is_tta_long"](bb) is True


class TestV3Tree:
    def test_profiles_valid_and_ooda(self):
        ids = set(ManeuverRegistry.from_csv().ids())
        for profile in V3_PROFILES:
            tree = tactical_tree_v3_json(profile)
            assert validate_tree_json(tree, maneuver_ids=ids) == []
            coverage, missing = ooda_coverage(tree)
            assert coverage == 1.0 and missing == []
            assert set(OODA_ALL) <= {n["name"] for n in tree["nodes"]
                                     if n.get("name")}

    def test_first_shot_branch_fires(self):
        bb = Blackboard(env=FakeEnv())
        bb.set_frame(_obs(dist=30400.0), _info(), 0)
        bb.set_frame(_obs(dist=30000.0), _info(), 1)
        root = build_tree(tactical_tree_v3_json("balanced"), bb)
        action = tick_once(root, bb, bb.obs, bb.info, 1)
        assert bb.tick_counts["fire_missile"] == 1
        assert action["weapon"][0] == 1

    def test_weapons_in_flight_cranks(self):
        # 已无弹、但仍有导弹在飞：先手/补射分支失败，进入 crank
        bb = Blackboard(env=FakeEnv(missiles=[object()]))
        bb.set_frame(_obs(dist=30000.0, n_msl=0.0),
                     _info(in_zone=False), 0)
        root = build_tree(tactical_tree_v3_json("balanced"), bb)
        tick_once(root, bb, bb.obs, bb.info, 0)
        assert bb._maneuver_state.get("name") == "crank"

    def test_boundary_returns(self):
        bb = Blackboard(env=FakeEnv(pos=(130000.0, 0.0, 6000.0)))
        bb.set_frame(_obs(detected=0.0), _info(), 0)
        root = build_tree(tactical_tree_v3_json("balanced"), bb)
        tick_once(root, bb, bb.obs, bb.info, 0)
        assert bb._maneuver_state.get("name") == "fuel_rtb"

    def test_defense_selects_notch_for_long_tta(self):
        bb = Blackboard(env=FakeEnv())
        bb.set_frame(_obs(), _info(maws_alarm=True, maws_tta_s=12.0), 0)
        root = build_tree(tactical_tree_v3_json("balanced"), bb)
        tick_once(root, bb, bb.obs, bb.info, 0)
        # 防御分支先于进攻；evade_missile 节点被执行
        assert bb.tick_counts["evade_missile"] == 1




