# -*- coding: utf-8 -*-
"""ManeuverSpec / ManeuverRegistry 回归测试。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))

import pytest

from llm_bt.maneuvers import (DEFAULT_CATALOG, ManeuverRegistry,
                              ManeuverSpec, catalog_path)


def test_default_catalog_loads_and_has_core_moves():
    registry = ManeuverRegistry.from_csv()
    assert len(registry) >= 10
    for maneuver_id in ("search_pattern", "lock_and_track", "crank",
                        "notch", "launch_and_crank", "break_turn",
                        "barrel_roll", "defensive_spiral", "fuel_rtb"):
        assert registry.has(maneuver_id)
    assert registry.get("crank").phase == "act"
    assert "missing" not in registry.ids()


def test_registry_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "catalog.csv"
    row = {
        "maneuver_id": "x", "name_zh": "测试", "name_en": "x",
        "phase": "act", "preconditions": "p", "parameters": "p",
        "action_sequence_or_controller": "c", "exit_conditions": "e",
        "safety_constraints": "s", "source_type": "manual",
        "source_ref": "ref", "status": "design", "notes": "",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)
    with pytest.raises(ValueError, match="重复招式"):
        ManeuverRegistry.from_csv(path)


def test_spec_missing_required_field_raises():
    with pytest.raises(ValueError, match="缺少必填字段"):
        ManeuverSpec.from_row({"maneuver_id": "x", "name_zh": "测试"})


def test_catalog_path_env_override(tmp_path, monkeypatch):
    target = tmp_path / "catalog.csv"
    monkeypatch.setenv("BVR_MANEUVER_CATALOG", str(target))
    assert catalog_path() == target.resolve()


def test_default_catalog_exists():
    assert DEFAULT_CATALOG.is_file()
