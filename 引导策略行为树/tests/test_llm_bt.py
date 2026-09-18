# -*- coding: utf-8 -*-
"""
引导策略行为树/tests/test_llm_bt.py（步骤 2.5 单元测试）
================================================================
覆盖：JSON 规范校验、XML/bt_validator 静态校验复用、文本/DOT 可视化、
OODA 覆盖计算、执行器（条件/动作/边界安全）、Tester 评估与离线
Coder-Executor-Tester 两轮闭环（生成→测试→修正→收敛）。

运行（DC 环境，在 引导策略行为树/ 目录下）：
    python -m pytest tests/test_llm_bt.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT.parent / "多任务智能体"))   # 数据源：common 包

import numpy as np
import pytest

from llm_bt.executor import Blackboard, build_tree, tick_once
from llm_bt.llm_client import LLMClient, extract_json
from llm_bt.mock_llm import coder_response
from llm_bt.prompts import load_prompt
from llm_bt.schema import (ACTIONS, CONDITIONS, OODA_ALL, ooda_coverage,
                           tree_node_names, validate_tree_json)
from llm_bt.tester import default_test_cases, evaluate_case, run_case
from llm_bt.tree_io import text_tree, tree_to_dot, tree_to_xml
from llm_bt.validator_bridge import validate_tree_static


def _mock_tree(missing=()):
    return extract_json(coder_response({"missing": list(missing)}))


@pytest.fixture(scope="module")
def tree_attack_only():
    # 首轮（无反馈）生成的简化树：缺 Self_Defense / Fuel_Return 分支
    return _mock_tree(missing=())


@pytest.fixture(scope="module")
def tree_full():
    # 收到反馈后补齐 OODA 缺失分支的完整树
    return _mock_tree(missing=("evade_missile", "is_missile_incoming", "is_fuel_low"))


class TestSchema:
    def test_mock_tree_valid(self, tree_full):
        assert validate_tree_json(tree_full) == []

    def test_reject_unknown_action(self, tree_full):
        data = json.loads(json.dumps(tree_full))
        data["nodes"][0]["type"] = "Action"
        data["nodes"][0]["name"] = "do_a_barrel_roll"
        errors = validate_tree_json(data)
        assert any("名称非法" in e for e in errors)

    def test_reject_duplicate_id(self, tree_full):
        data = json.loads(json.dumps(tree_full))
        data["nodes"][1]["id"] = data["nodes"][0]["id"]
        assert any("重复" in e for e in validate_tree_json(data))

    def test_reject_leaf_with_child(self, tree_attack_only):
        data = json.loads(json.dumps(tree_attack_only))
        leaf = next(n for n in data["nodes"] if n["type"] == "Condition")
        extra = {"id": "orphan", "type": "Action", "name": "search_target"}
        data["nodes"].append(extra)
        data["edges"].append({"from": leaf["id"], "to": "orphan", "order": 0})
        assert any("叶子节点" in e for e in validate_tree_json(data))

    def test_reject_two_roots(self, tree_attack_only):
        data = json.loads(json.dumps(tree_attack_only))
        data["nodes"].append({"id": "solo", "type": "Action", "name": "search_target"})
        assert any("根节点" in e for e in validate_tree_json(data))


class TestStaticValidator:
    def test_bt_validator_reuse(self, tree_full):
        errors, warnings = validate_tree_static(tree_full)
        assert errors == [] and warnings == []

    def test_xml_selector_maps_to_fallback(self, tree_full):
        xml = tree_to_xml(tree_full)
        assert "<Fallback" in xml and "<Sequence" in xml
        assert "<fire_missile" in xml and "<navigate_to_waypoint" in xml
        assert "<Action" not in xml and "<Condition" not in xml   # 叶子用类名标签
        assert "BehaviorTree" in xml and "main_tree_to_execute" in xml

    def test_text_and_dot(self, tree_attack_only):
        text = text_tree(tree_attack_only)
        assert "BVR_Engagement" in text and "fire_missile" in text
        assert "Self_Defense" not in text          # 攻击型树无自保分支
        dot = tree_to_dot(tree_attack_only)
        assert dot.startswith("digraph") and "->" in dot


class TestCoverage:
    def test_attack_only_missing_ooda(self, tree_attack_only):
        coverage, missing = ooda_coverage(tree_attack_only)
        assert coverage < 0.95
        assert "evade_missile" in missing and "is_fuel_low" in missing

    def test_full_coverage(self, tree_full):
        coverage, missing = ooda_coverage(tree_full)
        assert coverage == 1.0 and missing == []
        assert set(OODA_ALL) <= set(tree_node_names(tree_full))


class TestExecutor:
    def test_tick_produces_valid_action(self, tree_full):
        """合成一帧观测 tick：动作类型/范围合法（不依赖 JSBSim）。"""
        bb = Blackboard(env=None)
        root = build_tree(tree_full, bb)
        obs = np.zeros(28, dtype=np.float32)
        obs[0] = 6000.0    # 高度
        obs[1] = 240.0     # 速度
        obs[5] = 0.0       # 航向
        obs[10] = 0.8      # 燃油
        obs[12] = 2.0      # 导弹
        info = {"in_zone": False, "maws_alarm": False}
        action = tick_once(root, bb, obs, info, 0)
        flight = np.asarray(action["flight"], dtype=float)
        assert flight.shape == (4,) and np.all(np.isfinite(flight))
        assert 0.0 <= flight[0] <= 1.0 and np.all(np.abs(flight[1:]) <= 1.0)
        assert len(action["weapon"]) == 3
        assert bb.tick_counts   # 执行日志已记录

    def test_fire_requires_zone_and_missiles(self, tree_full):
        bb = Blackboard(env=None)
        root = build_tree(tree_full, bb)
        obs = np.zeros(28, dtype=np.float32)
        obs[0], obs[1], obs[10], obs[12] = 6000.0, 240.0, 0.8, 2.0
        info = {"in_zone": False}
        action = tick_once(root, bb, obs, info, 0)
        assert action["weapon"][0] == 0            # 不在攻击区：不发射
        action = tick_once(root, bb, obs, {"in_zone": True}, 1)
        assert action["weapon"][0] == 1            # 攻击区内：发射
        action = tick_once(root, bb, obs, {"in_zone": True}, 2)
        assert action["weapon"][0] == 0            # 4 s 连发间隔内：不重复发射

    def test_run_case_no_logic_errors(self, tree_full):
        case = {"case_id": "t1", "seed": 7,
                "success_criteria": [{"metric": "no_logic_error", "op": "==", "value": 1}]}
        result = run_case(tree_full, case, max_steps=40, compute_envelope=False)
        assert result["logic_errors"] == 0
        assert result["steps"] >= 1
        assert result["executed_nodes"]


class TestTester:
    def test_default_cases(self):
        cases = default_test_cases()
        assert len(cases) == 5
        assert len({c["seed"] for c in cases}) == 5
        for c in cases:
            assert any(x["metric"] == "safety_violations" for x in c["success_criteria"])

    def test_evaluate_case(self):
        case = {"success_criteria": [
            {"metric": "safety_violations", "op": "==", "value": 0},
            {"metric": "no_crash", "op": "==", "value": 1}]}
        good = {"logic_errors": 0, "safety_violations": 0, "reason": "timeout",
                "steps": 60, "fired_own": 0, "in_zone_steps": 0, "enemy_fired": 0}
        bad = dict(good, safety_violations=2, reason="own_crash")
        assert evaluate_case(case, good)[0] is True
        assert evaluate_case(case, bad)[0] is False


class TestExtractJson:
    def test_fenced_and_prefixed(self, tree_full):
        raw = "好的，以下是行为树：\n```json\n" + json.dumps(tree_full) + "\n```\n以上。"
        assert extract_json(raw)["nodes"] == tree_full["nodes"]
        assert extract_json(json.dumps(tree_full))["bt_id"] == tree_full["bt_id"]


class TestPrompts:
    def test_templates_size_and_catalog(self):
        coder = load_prompt("coder_prompt.md")
        tester = load_prompt("tester_prompt.md")
        assert len(coder.encode("utf-8")) >= 1024   # 交付物要求：每个 ≥1KB
        assert len(tester.encode("utf-8")) >= 1024
        for name in list(ACTIONS) + list(CONDITIONS):   # 防止目录与模板漂移
            assert name in coder
        for metric in ("safety_violations", "no_logic_error", "engaged"):
            assert metric in tester


class TestSimRecording:
    def test_episode_recording_roundtrip(self, tmp_path):
        """行为树接入仿真→录制链路：CSV/ACMI 帧数与内容完整（接口统一验证）。"""
        import run_tree_sim
        from visualization.acmi import export_acmi

        tree = _mock_tree(missing=("evade_missile", "is_missile_incoming", "is_fuel_low"))
        rec, summary = run_tree_sim.run_episode(tree, seed=9, max_steps=25,
                                                compute_envelope=False)
        assert summary["steps"] >= 20
        assert summary["logic_errors"] == 0 and summary["safety_violations"] == 0
        assert len(rec) == summary["steps"] + 1          # 初始帧 + 每步一帧
        frames = rec.__class__.load_csv(rec.save_csv(tmp_path / "ep.csv"))
        assert len(frames) == len(rec)
        acmi = export_acmi(rec.frames, tmp_path / "ep.acmi")
        text = acmi.read_text(encoding="utf-8")
        assert "Coalition=Allies" in text and "Coalition=Enemies" in text
        assert text.count("\n#") >= summary["steps"] - 1
        assert "Type=Air+FixedWing" in text


class TestOfflineIteration:
    def test_two_round_convergence(self, tmp_path):
        """离线两轮闭环：v1 缺自保/燃油分支 → 反馈 → v2 补齐并收敛。"""
        from llm_bt_generator import run_iteration
        result = run_iteration(mission="离线测试任务", provider="mock", rounds=3,
                               max_steps=60, compute_envelope=False,
                               out_dir=tmp_path, verbose=False)
        history = result["history"]
        assert len(history) == 2
        assert history[0]["status"] == "not_converged"
        assert "evade_missile" in history[0]["missing"]
        assert history[1]["status"] == "converged"
        assert history[1]["coverage"] == 1.0
        assert history[1]["summary"]["pass_rate"] == 1.0
        assert history[1]["summary"]["safety_violations"] == 0
        run_dir = result["run_dir"]
        assert (run_dir / "bt_v2.json").is_file()
        assert (run_dir / "bt_final.xml").is_file()
        log = (run_dir / "iteration_log.md").read_text(encoding="utf-8")
        assert "## Round 1" in log and "## Round 2" in log
        assert "是否收敛：是" in log
