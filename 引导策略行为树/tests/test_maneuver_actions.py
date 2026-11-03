# -*- coding: utf-8 -*-
"""execute_maneuver / ManeuverExecutors 回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))

import numpy as np
import pytest

from llm_bt.executor import Blackboard, build_tree, tick_once
from llm_bt.maneuver_actions import MANEUVER_EXECUTORS, run_maneuver
from llm_bt.maneuvers import ManeuverRegistry
from llm_bt.schema import ooda_coverage, validate_tree_json
from llm_bt.tactical_execution_tree import tactical_tree_80_120_json


def _obs(**kw):
    o = np.zeros(28, dtype=np.float64)
    o[0], o[1], o[2] = 6000.0, 250.0, 0.8
    o[9], o[10], o[11], o[12] = 0.55, 1.0, 0.0, 2
    o[13], o[18], o[21] = 0.0, 0.2, 1.0
    for key, value in kw.items():
        o[int(key[1:])] = value
    return o


def _info():
    return {"in_zone": True, "dist_m": 30000.0, "r_max_own_m": 45000.0,
            "ecm_on": False, "maws_alarm": False}


def test_all_catalog_maneuvers_have_executor():
    registry = ManeuverRegistry.from_csv()
    assert set(registry.ids()) <= set(MANEUVER_EXECUTORS)


def test_every_maneuver_runs_without_exception():
    registry = ManeuverRegistry.from_csv()
    allowed = {"SUCCESS", "RUNNING", "FAILURE"}
    for name in registry.ids():
        bb = Blackboard(control_mode="assisted")
        bb.set_frame(_obs(), _info(), 0)
        bb.maneuver_registry = registry
        status = run_maneuver(bb, name, {})
        assert status.name in allowed, name
        assert np.all(np.isfinite(np.asarray(bb.action["flight"], dtype=float)))


def test_assisted_and_raw_search_use_different_semantics():
    registry = ManeuverRegistry.from_csv()
    assisted = Blackboard(control_mode="assisted")
    assisted.set_frame(_obs(), _info(), 0)
    assisted.maneuver_registry = registry
    run_maneuver(assisted, "search_pattern", {"speed_mps": 240})
    raw = Blackboard(control_mode="raw_fcs")
    raw.set_frame(_obs(), _info(), 0)
    raw.maneuver_registry = registry
    run_maneuver(raw, "search_pattern", {"speed_mps": 240})
    assert abs(assisted.action["flight"][0] - raw.action["flight"][0]) > 0.05


def test_execute_maneuver_tree_action_and_unknown_name():
    tree = {
        "bt_id": "t", "mission": "m",
        "nodes": [
            {"id": "n1", "type": "Action", "name": "execute_maneuver",
             "args": {"maneuver": "crank", "parameters": {"offset_deg": 60}}},
        ],
        "edges": [],
    }
    assert validate_tree_json(tree) == []
    registry = ManeuverRegistry.from_csv()
    assert validate_tree_json(tree, maneuver_ids=set(registry.ids())) == []
    bb = Blackboard(control_mode="assisted")
    bb.set_frame(_obs(), _info(), 0)
    root = build_tree(tree, bb)
    tick_once(root, bb, _obs(), _info(), 0)
    assert root.status.name in {"SUCCESS", "RUNNING"}

    bad = {
        "bt_id": "t2", "mission": "m",
        "nodes": [
            {"id": "n1", "type": "Action", "name": "execute_maneuver",
             "args": {"maneuver": "not_a_maneuver", "parameters": {}}},
        ],
        "edges": [],
    }
    assert any("未注册招式" in e for e in
               validate_tree_json(bad, maneuver_ids=set(registry.ids())))
    bb2 = Blackboard(control_mode="assisted")
    bb2.set_frame(_obs(), _info(), 0)
    root2 = build_tree(bad, bb2)
    with pytest.raises(ValueError, match="未注册招式"):
        tick_once(root2, bb2, _obs(), _info(), 0)


def test_80_120_reference_tree_is_complete():
    tree = tactical_tree_80_120_json()
    registry = ManeuverRegistry.from_csv()
    assert validate_tree_json(tree, maneuver_ids=set(registry.ids())) == []
    coverage, missing = ooda_coverage(tree)
    assert coverage == 1.0 and missing == []
    names = {n.get("name") for n in tree["nodes"]}
    assert {"execute_maneuver", "is_in_nez", "evade_missile"} <= names
