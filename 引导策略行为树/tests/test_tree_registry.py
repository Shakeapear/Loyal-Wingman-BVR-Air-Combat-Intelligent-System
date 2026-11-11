# -*- coding: utf-8 -*-
"""场景教师树清单与路由测试。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))

from llm_bt.maneuvers import ManeuverRegistry
from llm_bt.schema import ooda_coverage, validate_tree_json
from llm_bt.tree_registry import load_manifest, resolve_tree


def test_manifest_has_both_scenes():
    manifest = load_manifest()
    assert "default" in manifest and "bvr80-120" in manifest
    assert manifest["bvr80-120"]["metrics"]["survival_rate"] >= 0.9


def test_resolve_and_validate_both_trees():
    ids = set(ManeuverRegistry.from_csv().ids())
    for scene in ("default", "bvr80-120"):
        tree = resolve_tree(scene)
        assert validate_tree_json(tree, maneuver_ids=ids) == []
        assert ooda_coverage(tree)[0] == 1.0


def test_resolve_teacher_v3_for_both_scenes():
    ids = set(ManeuverRegistry.from_csv().ids())
    for scene in ("teacher_v3_default", "teacher_v3_bvr80-120"):
        tree = resolve_tree(scene)
        assert tree["bt_id"] == "bt_tactical_v3_break_turn_defense"
        assert validate_tree_json(tree, maneuver_ids=ids) == []
        assert ooda_coverage(tree)[0] == 1.0


def test_resolve_teacher_speed260_for_both_scenes():
    ids = set(ManeuverRegistry.from_csv().ids())
    for scene in ("teacher_v3_speed260_default",
                  "teacher_v3_speed260_bvr80-120"):
        tree = resolve_tree(scene)
        assert tree["bt_id"] == "bt_tactical_v3_speed260"
        assert validate_tree_json(tree, maneuver_ids=ids) == []
        assert ooda_coverage(tree)[0] == 1.0
