# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/tester.py（步骤 2.5 组件：Tester 仿真评估）
================================================================
- generate_test_cases : LLM 生成 5 组测试用例（mock 返回离线默认用例；
                        真实 LLM 输出解析失败时回退默认用例）；
- run_case / run_test_cases : 在 BVRCombatEnv（JSBSim）中执行行为树并评估；
- 指标：通过率、安全违规步数、OODA 关键节点覆盖（静态 + 实际执行）、
        发射/命中/交战等态势结果（收敛判据见 schema.CONVERGENCE）。

测试用例字段（与 tester_prompt.md 一致）：
    case_id, description, expected_behavior, seed,
    success_criteria = [{"metric": ..., "op": ..., "value": ...}]
环境初始态势由 seed 复现（距离 30~80 km、相对方位 ±60°、高度/速度随机），
详见 common/bvr_combat_env.reset()。
"""
from __future__ import annotations

import json
import math
import operator
import sys
from pathlib import Path

import numpy as np

from .executor import Blackboard, build_tree, tick_once
from .llm_client import extract_json
from .mock_llm import DEFAULT_TEST_SEEDS
from .prompts import SYSTEM_TESTER, render_tester_prompt
from .schema import OODA_ALL, ooda_coverage, tree_node_names

# 数据源：智能体库环境（引导策略行为树 → 多任务智能体/common，单向依赖）
_AGENT_LIB = Path(__file__).resolve().parents[2] / "多任务智能体"
if str(_AGENT_LIB) not in sys.path:
    sys.path.append(str(_AGENT_LIB))

from common.bvr_combat_env import BVRCombatEnv   # noqa: E402

OPS = {"==": operator.eq, "!=": operator.ne, ">=": operator.ge,
       "<=": operator.le, ">": operator.gt, "<": operator.lt}


def _b(x) -> int:
    return 1 if x else 0


# ---- 指标词汇表（对齐 tester_prompt.md；输入为单用例执行结果 dict）----
METRICS = {
    "no_logic_error": lambda r: _b(r["logic_errors"] == 0),
    "safety_violations": lambda r: r["safety_violations"],
    "no_crash": lambda r: _b(r["reason"] != "own_crash"),
    "steps_executed": lambda r: r["steps"],
    "fired_own": lambda r: r["fired_own"],
    "in_zone_steps": lambda r: r["in_zone_steps"],
    "enemy_hit": lambda r: _b(r["reason"] == "enemy_hit"),
    "own_not_hit": lambda r: _b(r["reason"] != "own_hit"),
    "engaged": lambda r: _b(r["fired_own"] > 0 or r["in_zone_steps"] > 0
                            or r["enemy_fired"] > 0),
}

_SAFE_CRITERIA = [
    {"metric": "no_logic_error", "op": "==", "value": 1},
    {"metric": "safety_violations", "op": "==", "value": 0},
    {"metric": "no_crash", "op": "==", "value": 1},
    {"metric": "steps_executed", "op": ">=", "value": 30},
]


def default_test_cases() -> list:
    """离线默认用例（mock Tester / 解析回退共用）。"""
    cases = []
    for i, seed in enumerate(DEFAULT_TEST_SEEDS, 1):
        cases.append({
            "case_id": f"case_{i}",
            "description": f"随机初始态势 seed={seed}（距离 30~80 km、相对方位 ±60°）",
            "expected_behavior": "探测→锁定→进入攻击区发射；受威胁时规避；全程无安全违规",
            "seed": int(seed),
            "success_criteria": [dict(c) for c in _SAFE_CRITERIA],
        })
    return cases


def _normalize_case(case: dict, index: int) -> dict:
    crit = [c for c in (case.get("success_criteria") or [])
            if isinstance(c, dict) and c.get("metric") in METRICS and c.get("op") in OPS]
    return {
        "case_id": str(case.get("case_id") or f"case_{index + 1}"),
        "description": str(case.get("description") or ""),
        "expected_behavior": str(case.get("expected_behavior") or ""),
        "seed": int(case.get("seed", DEFAULT_TEST_SEEDS[index % len(DEFAULT_TEST_SEEDS)])),
        "success_criteria": crit or [dict(c) for c in _SAFE_CRITERIA],
    }


def generate_test_cases(llm, tree_json: dict):
    """LLM 生成测试用例；返回 (cases, raw_text)。解析失败回退默认用例。"""
    raw = llm.complete(SYSTEM_TESTER, render_tester_prompt(tree_json),
                       task="tester", context={"tree": tree_json})
    try:
        data = extract_json(raw)
        raw_cases = data.get("test_cases") or []
        cases = [_normalize_case(c, i) for i, c in enumerate(raw_cases)
                 if isinstance(c, dict)]
        if not cases:
            raise ValueError("test_cases 为空")
        return cases, raw
    except Exception as e:   # LLM 输出不可解析：回退默认用例并记录原因
        return default_test_cases(), raw + f"\n[解析回退] {type(e).__name__}: {e}"


def _is_safety_violation(obs, cfg) -> bool:
    nz_g, mach, h_m, v_mps = float(obs[8]), float(obs[2]), float(obs[0]), float(obs[1])
    return (nz_g > cfg["safety_nz_max_g"] or nz_g < cfg["safety_nz_min_g"]
            or mach > cfg["safety_mach_max"] or h_m > cfg["safety_alt_max_m"]
            or v_mps < cfg["safety_vmin_mps"])


def run_case(tree_json: dict, case: dict, max_steps: int = 240,
             compute_envelope: bool = True) -> dict:
    """在 BVRCombatEnv 中执行单个测试用例，返回指标 dict。"""
    env = BVRCombatEnv(config={"compute_envelope": bool(compute_envelope)})
    bb = Blackboard(env)
    root = build_tree(tree_json, bb)
    obs, info = env.reset(seed=int(case["seed"]))
    logic_errors = safety = in_zone_steps = 0
    own_fired = enemy_fired = 0
    reason, steps = None, 0
    try:
        for step in range(int(max_steps)):
            action = tick_once(root, bb, obs, info, step)
            flight = np.asarray(action["flight"], dtype=float)
            if flight.shape != (4,) or not np.all(np.isfinite(flight)):
                logic_errors += 1
                flight = np.array([0.55, 0.0, 0.0, 0.0])
            flight[0] = float(np.clip(flight[0], 0.0, 1.0))
            flight[1:] = np.clip(flight[1:], -1.0, 1.0)
            weapon = np.clip(np.asarray(action["weapon"], dtype=int), 0, 1)
            try:
                obs, _reward, term, trunc, info = env.step(
                    {"flight": flight.astype(np.float32), "weapon": weapon})
            except Exception:            # 执行异常计入逻辑错误并终止本用例
                logic_errors += 1
                break
            steps = step + 1
            if info.get("in_zone"):
                in_zone_steps += 1
            own_fired = int(info.get("own_fired", own_fired))
            enemy_fired = int(info.get("enemy_fired", enemy_fired))
            if _is_safety_violation(obs, env.cfg):
                safety += 1
            if term or trunc:
                reason = info.get("terminated_reason")
                break
    finally:
        env.close()
    return {
        "case_id": case["case_id"], "seed": int(case["seed"]),
        "steps": steps, "reason": reason,
        "logic_errors": logic_errors, "safety_violations": safety,
        "in_zone_steps": in_zone_steps, "fired_own": own_fired,
        "enemy_fired": enemy_fired,
        "executed_nodes": sorted(bb.tick_counts),
        "tick_counts": dict(bb.tick_counts),
    }


def evaluate_case(case: dict, result: dict):
    """按 success_criteria 评估，返回 (passed, checks)。"""
    checks = []
    for crit in case.get("success_criteria", []):
        fn = METRICS.get(crit.get("metric"))
        op = OPS.get(crit.get("op"))
        actual = fn(result) if fn else None
        try:
            passed = bool(fn and op and op(actual, crit.get("value")))
        except Exception:
            passed = False
        checks.append({"criterion": crit, "actual": actual, "passed": passed})
    return all(c["passed"] for c in checks), checks


def run_test_cases(tree_json: dict, cases: list, max_steps: int = 240,
                   compute_envelope: bool = True) -> dict:
    """批量执行用例并汇总（通过率、安全违规、执行覆盖）。"""
    results, executed = [], set()
    for case in cases:
        result = run_case(tree_json, case, max_steps=max_steps,
                          compute_envelope=compute_envelope)
        passed, checks = evaluate_case(case, result)
        executed.update(result["executed_nodes"])
        results.append({"case": case, "passed": passed, "checks": checks, "result": result})
    n_pass = sum(1 for r in results if r["passed"])
    static_cov, static_missing = ooda_coverage(tree_json)
    hit = [n for n in OODA_ALL if n in executed]
    return {
        "cases": results,
        "pass_rate": n_pass / len(results) if results else 0.0,
        "n_pass": n_pass, "n_cases": len(results),
        "safety_violations": sum(r["result"]["safety_violations"] for r in results),
        "ooda_coverage": static_cov, "ooda_missing": static_missing,
        "ooda_executed_coverage": len(hit) / len(OODA_ALL),
        "ooda_executed_missing": [n for n in OODA_ALL if n not in executed],
    }
