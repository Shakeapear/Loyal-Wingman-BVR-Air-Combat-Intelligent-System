# -*- coding: utf-8 -*-
"""
引导策略行为树/tests/test_validation_2_7.py（步骤 2.7 冒烟测试）
================================================================
不跑 20 局；只验证：
- seed 解析 / 手写树与 LLM 树加载校验；
- node_trace 完整性校验（合法/缺失/非法节点）；
- 2 seeds × 3 步最小联调，summary 字段与 trace 完整。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))

import run_2_7_validation as v27
from llm_bt.schema import ACTIONS, OODA_ALL


class TestRunnerBasics:
    def test_parse_seeds(self):
        assert v27.parse_seeds("1-3,5,4-4") == [1, 2, 3, 4, 5]

    def test_load_handwritten_and_llm(self):
        hw, label, _ = v27.load_tree("handwritten", None, None)
        assert label == "handwritten" and hw["bt_id"] == "bt_tactical_2_6"
        llm, label2, source = v27.load_tree("llm", None, None)
        assert label2 == "llm" and llm["bt_id"] and Path(source).is_file()

    def test_node_trace_validation(self):
        ok_log = [(0, "search_target", "SUCCESS"),
                  (0, "has_human_command", "FAILURE"),
                  (1, "search_target", "SUCCESS")]
        ticks = {n: 1 for n in OODA_ALL}
        trace = v27.validate_node_trace(ok_log, 2, ticks)
        assert trace["ok"] is True and trace["ooda_complete"] is True
        assert trace["missing_steps"] == [] and trace["illegal_nodes"] == []

        gap = v27.validate_node_trace(ok_log, 3, ticks)
        assert gap["ok"] is False and gap["missing_steps"] == [2]

        illegal = v27.validate_node_trace([(0, "not_a_node", "SUCCESS")], 1, ticks)
        assert illegal["ok"] is False and illegal["illegal_nodes"] == ["not_a_node"]

    def test_two_seed_smoke(self):
        tree, label, source = v27.load_tree("handwritten", None, None)
        rows = []
        for seed in (1, 2):
            result = v27.run_episode(tree, seed, max_steps=3,
                                     compute_envelope=False)
            s = result["summary"]
            rows.append(s)
            assert s["seed"] == seed and s["steps"] == 3
            assert s["safety_violations"] == 0 and s["logic_errors"] == 0
            assert s["node_trace_ok"] is True
            assert s["is_guided_ratio"] == 1.0
            assert s["terminated_reason"] in (None, "timeout")
            assert result["trace"]["first_step"] == 0
            assert result["trace"]["last_step"] == 2
            assert set(result["trace"]["ooda_executed_missing"]) <= set(OODA_ALL)
        agg = v27.aggregate(rows, label, tree["bt_id"], source, [1, 2], 3,
                            tree)
        assert agg["n_episodes"] == 2
        assert agg["ooda_static_coverage"] == 1.0
        assert agg["survival_rate"] == 1.0
        assert agg["safety_violations"] == 0
        assert agg["trace_ok_episodes"] == 2
        assert agg["misfire_rate"] == 0.0


class TestActionCatalog:
    def test_legal_nodes_cover_schema(self):
        assert set(ACTIONS) <= v27.LEGAL_NODES
