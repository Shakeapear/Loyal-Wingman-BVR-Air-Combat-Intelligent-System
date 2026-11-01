# -*- coding: utf-8 -*-
"""
引导策略行为树/tests/test_guidance_2_6.py（步骤 2.6 单元测试）
================================================================
覆盖：6 类边界约束（触发/不触发/覆盖/限幅/滞回/日志）、战术执行子树
（JSON 校验、OODA 100%、执行）、GuideInterface（激活/耗尽/更新/reset/None）、
最小联调（run_guidance_sim.run_episode 3 步）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))

import numpy as np
import pytest

from llm_bt.boundary_constraints import BoundaryConstraints
from llm_bt.executor import Blackboard, build_tree, tick_once
from llm_bt.guide_interface import GuideInterface
from llm_bt.schema import OODA_ALL, ooda_coverage
from llm_bt.tactical_execution_tree import (build_tactical_tree,
                                            tactical_coverage,
                                            tactical_tree_json,
                                            validate_tactical_tree)
from llm_bt.tree_io import text_tree, tree_to_dot


def _obs(**kw):
    """构造 28 维观测：h, v, mach, phi, theta, psi, ..., nz, throttle, fuel, ..."""
    o = np.zeros(28, dtype=np.float64)
    o[0], o[1], o[2] = 6000.0, 240.0, 0.8
    o[5], o[8], o[9], o[10] = 0.0, 1.0, 0.6, 0.9
    for idx, val in kw.items():
        o[int(idx[1:])] = float(val)
    return o


def _bb(obs, info=None, step=0):
    bb = Blackboard(env=None)
    bb.set_frame(obs, info or {}, step)
    bb.action = {"flight": [0.55, 0.0, 0.0, 0.0], "weapon": [0, 0, 0]}
    return bb


def _guard(**cfg):
    return BoundaryConstraints(config=cfg)


class TestBoundaryConstraints:
    def test_level_flight_no_trigger(self):
        guard = _guard()
        bb = _bb(_obs())
        action = guard.apply(bb.action, bb)
        assert guard.summary()["boundary_trigger_count"] == 0
        assert action["flight"] == [0.55, 0.0, 0.0, 0.0]

    def test_min_altitude_pull_up_and_log(self):
        guard = _guard()
        bb = _bb(_obs(o0=300.0))
        action = guard.apply(bb.action, bb)
        assert guard.active["min_altitude"] is True
        assert action["flight"][0] == 1.0
        assert action["flight"][2] <= -0.15           # 负俯仰 = 抬头
        entry = guard.log[0]
        assert entry["constraint"] == "min_altitude"
        assert entry["event"] == "trigger"
        assert entry["value"] == 300.0 and entry["threshold"] == 500.0
        assert entry["action_override"] == "pull_up"

    def test_min_speed_accelerate(self):
        guard = _guard()
        bb = _bb(_obs(o1=80.0))
        action = guard.apply(bb.action, bb)
        assert guard.active["min_speed"] is True
        assert action["flight"][0] == 1.0
        assert action["flight"][2] >= 0.12

    def test_max_speed_decelerate(self):
        guard = _guard()
        bb = _bb(_obs(o2=1.8))
        action = guard.apply(bb.action, bb)
        assert guard.active["max_speed"] is True
        assert action["flight"][0] == 0.0
        assert action["flight"][2] <= -0.04

    def test_over_g_limits_rates(self):
        guard = _guard()
        bb = _bb(_obs(o8=6.0))
        bb.action["flight"][2] = 0.3
        action = guard.apply(bb.action, bb)
        assert guard.active["over_g"] is True
        assert guard.log[-1]["action_override"] == "limit_control_rate"
        # 上一步 pitch=0，限幅 0.08；放松拉杆 +0.06 后仍被限到 0.08
        assert action["flight"][2] == pytest.approx(0.08)

    def test_no_fly_circle_triggers(self):
        guard = _guard(no_fly_zones=[
            {"name": "z1", "type": "circle", "center": [0.0, 0.0], "radius_m": 5000.0}])
        bb = _bb(_obs(), info={"own_pos_enu_m": [100.0, 100.0]})
        action = guard.apply(bb.action, bb)
        assert guard.active["no_fly_zone"] is True
        assert guard.log[-1]["action_override"] == "exit_no_fly_zone"
        assert action["flight"][0] >= 0.8
        assert abs(action["flight"][1]) > 0.0

    def test_no_fly_polygon_triggers(self):
        guard = _guard(no_fly_zones=[
            {"name": "p1", "type": "polygon",
             "vertices": [[-1000, -1000], [1000, -1000], [1000, 1000], [-1000, 1000]]}])
        bb = _bb(_obs(), info={"own_pos_enu_m": [0.0, 0.0]})
        guard.apply(bb.action, bb)
        assert guard.active["no_fly_zone"] is True

    def test_fuel_low_forces_return_to_base(self):
        guard = _guard()
        bb = _bb(_obs(o10=0.1), info={"own_pos_enu_m": [30000.0, 0.0]})
        action = guard.apply(bb.action, bb)
        assert guard.active["fuel_low"] is True
        assert guard.log[-1]["action_override"] == "return_to_base"
        assert action["flight"][0] >= 0.75

    def test_hysteresis_avoids_chatter(self):
        guard = _guard()
        bb = _bb(_obs(o0=300.0))
        guard.apply(bb.action, bb)
        assert guard.active["min_altitude"] is True
        bb = _bb(_obs(o0=600.0), step=1)              # 回差区间内仍激活
        guard.apply(bb.action, bb)
        assert guard.active["min_altitude"] is True
        assert guard.counts["min_altitude"] == 1
        bb = _bb(_obs(o0=700.0), step=2)              # 超过退出阈值才清除
        guard.apply(bb.action, bb)
        assert guard.active["min_altitude"] is False
        assert [e["event"] for e in guard.log] == ["trigger", "clear"]

    def test_reset_clears_state(self):
        guard = _guard()
        bb = _bb(_obs(o0=300.0))
        guard.apply(bb.action, bb)
        assert guard.summary()["boundary_trigger_count"] == 1
        guard.reset()
        assert guard.summary()["boundary_trigger_count"] == 0
        assert not any(guard.active.values())


class TestTacticalExecutionTree:
    def test_tree_valid_and_full_ooda(self):
        assert validate_tactical_tree() == []
        coverage, missing = tactical_coverage()
        assert coverage == 1.0 and missing == []
        assert set(OODA_ALL) <= {n["name"] for n in tactical_tree_json()["nodes"]
                                 if n.get("name")}

    def test_text_and_dot(self):
        tree = tactical_tree_json()
        text = text_tree(tree)
        assert "BVR_Engagement" in text and "fire_missile" in text
        assert "Self_Defense" in text and "Human_Command_Response" in text
        dot = tree_to_dot(tree, "bt_tactical_2_6")
        assert dot.startswith("digraph") and "->" in dot

    def test_tree_executes_fire_branch(self):
        bb = _bb(_obs(o12=2.0, o21=1.0, o18=0.1), info={"in_zone": True})
        root = build_tactical_tree(bb)
        tick_once(root, bb, bb.obs, bb.info, 0)
        assert bb.tick_counts["fire_missile"] == 1
        assert bb.action["weapon"][0] == 1
        assert {"has_target_detected", "lock_target", "is_in_launch_zone",
                "has_weapon_remaining", "fire_missile"} <= set(bb.tick_counts)

    def test_tree_executes_search_default(self):
        bb = _bb(_obs(o21=0.0), info={"in_zone": False})
        root = build_tactical_tree(bb)
        tick_once(root, bb, bb.obs, bb.info, 0)
        assert "search_target" in bb.tick_counts
        assert "maintain_formation" in bb.tick_counts


class TestGuideInterface:
    @staticmethod
    def _policy(_obs):
        return {"flight": [0.5, 0.0, 0.0, 0.0], "weapon": [0, 0, 0]}

    def test_active_then_exhausted(self):
        guide = GuideInterface(max_guide_steps=2)
        a1, g1 = guide.get_action(_obs(), self._policy)
        a2, g2 = guide.get_action(_obs(), self._policy)
        a3, g3 = guide.get_action(_obs(), self._policy)
        assert g1 and g2 and a1 is not None and a2 is not None
        assert (a3, g3) == (None, False)
        assert guide.exhausted and guide.current_guide_step == 2

    def test_update_guide_steps_extends(self):
        guide = GuideInterface(max_guide_steps=1)
        assert guide.get_action(_obs(), self._policy)[1] is True
        assert guide.get_action(_obs(), self._policy)[1] is False
        guide.update_guide_steps(3)
        assert guide.get_action(_obs(), self._policy)[1] is True
        assert guide.current_guide_step == 2

    def test_update_zero_disables(self):
        guide = GuideInterface(max_guide_steps=5)
        guide.update_guide_steps(0)
        assert guide.guide_active is False
        assert guide.get_action(_obs(), self._policy) == (None, False)

    def test_reset_restores_initial_state(self):
        guide = GuideInterface(max_guide_steps=1)
        guide.get_action(_obs(), self._policy)
        guide.update_guide_steps(0)
        guide.reset()
        assert guide.current_guide_step == 0
        assert guide.max_guide_steps == 0
        assert guide.guide_active is True
        assert guide.get_action(_obs(), self._policy)[1] is False

    def test_none_action_not_counted(self):
        guide = GuideInterface(max_guide_steps=5)
        assert guide.get_action(_obs(), lambda _o: None) == (None, False)
        assert guide.current_guide_step == 0

    def test_inactive_returns_none(self):
        guide = GuideInterface(max_guide_steps=5, guide_active=False)
        assert guide.get_action(_obs(), self._policy) == (None, False)


class TestGuidanceIntegration:
    def test_run_episode_smoke(self):
        import run_guidance_sim
        rec, summary, events, tree = run_guidance_sim.run_episode(
            seed=11, max_steps=3, compute_envelope=False)
        assert summary["steps"] == 3
        assert summary["safety_violations"] == 0
        assert summary["logic_errors"] == 0
        assert summary["is_guided_ratio"] == 1.0
        assert summary["tactical_ooda_coverage"] == 1.0
        assert "boundary_trigger_count" in summary["boundary"]
        assert "per_constraint" in summary["boundary"]
        assert isinstance(events, list)
        assert len(rec) == summary["steps"] + 1
        assert tree["bt_id"] == "bt_tactical_2_6"
