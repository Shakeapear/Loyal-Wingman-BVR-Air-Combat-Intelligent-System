# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/mock_llm.py（离线模式：确定性 Coder/Tester）
================================================================
无 API Key 时用于框架开发、单测与演示：按确定性规则扮演 LLM 的
Coder/Tester 角色，接口与真实 LLM 完全一致（返回 JSON 文本）。

- Coder：按“上一轮反馈缺失的 OODA 节点”补全对应分支——第 1 轮生成
  仅攻击的简化树（缺 Self_Defense / Fuel_Return），收到反馈后第 2 轮
  补齐，真实复现「生成→测试→修正」闭环；
- Tester：返回 5 组固定种子 + 安全/逻辑判据的测试用例 JSON。

真实 LLM 接入见 llm_client.py（provider=deepseek/openai）。
"""
from __future__ import annotations

import json

DEFAULT_TEST_SEEDS = (11, 22, 33, 44, 55)

_SAFE_CRITERIA = [
    {"metric": "no_logic_error", "op": "==", "value": 1},
    {"metric": "safety_violations", "op": "==", "value": 0},
    {"metric": "no_crash", "op": "==", "value": 1},
    {"metric": "steps_executed", "op": ">=", "value": 30},
]


def _coder_tree(missing) -> dict:
    """确定性行为树生成：missing 中的 OODA 节点决定是否包含对应分支。"""
    nodes, edges = [], []

    def add(ntype, label=None, name=None, args=None):
        nid = f"n{len(nodes) + 1}"
        node = {"id": nid, "type": ntype}
        if label:
            node["label"] = label
        if name:
            node["name"] = name
        if args:
            node["args"] = args
        nodes.append(node)
        return nid

    def link(parent, child, order):
        edges.append({"from": parent, "to": child, "order": order})

    root = add("Selector", label="BVR_Mission")
    order = 0

    # L1 真人指令响应
    human = add("Sequence", label="Human_Command_Response")
    link(root, human, order); order += 1
    link(human, add("Condition", name="has_human_command"), 0)
    link(human, add("Action", name="navigate_to_waypoint",
                    args={"altitude_m": 8000, "speed_mps": 250, "heading_deg": 0}), 1)

    # L2 应急自保（第 1 轮缺失，收到反馈后补全）
    need_defense = ("evade_missile" in missing) or ("is_missile_incoming" in missing)
    if need_defense:
        defense = add("Sequence", label="Self_Defense")
        link(root, defense, order); order += 1
        link(defense, add("Condition", name="is_missile_incoming"), 0)
        link(defense, add("Action", name="evade_missile",
                          args={"maneuver_type": "break_turn"}), 1)

    # L2.5 燃油约束（第 1 轮缺失）
    if "is_fuel_low" in missing:
        fuel = add("Sequence", label="Fuel_Return")
        link(root, fuel, order); order += 1
        link(fuel, add("Condition", name="is_fuel_low"), 0)
        link(fuel, add("Action", name="return_to_base"), 1)

    # L3 超视距交战（覆盖 OODA：观察→判断→决策→行动）
    # 结构要点：检测成功后保持交战/跟踪——发射分支失败（未进攻击区/弹药耗尽/
    # 连发冷却）时回退到 lock_target 继续跟踪/机动，而不是落回默认巡逻
    # （否则巡逻动作会在同一步覆盖机动指令，见 executor 指令写入语义）。
    engage = add("Sequence", label="BVR_Engagement")
    link(root, engage, order); order += 1
    detect = add("Selector", label="Detection")
    link(engage, detect, 0)
    track = add("Sequence", label="Track_Target")
    link(detect, track, 0)
    link(track, add("Condition", name="has_target_detected"), 0)
    link(track, add("Action", name="lock_target"), 1)
    link(detect, add("Action", name="search_target", args={"radar_mode": "auto"}), 1)
    engage_or_track = add("Selector", label="Engage_or_Track")
    link(engage, engage_or_track, 1)
    launch = add("Sequence", label="Launch")
    link(engage_or_track, launch, 0)
    link(launch, add("Condition", name="is_in_launch_zone"), 0)
    fire = add("Sequence", label="Fire_Control")
    link(launch, fire, 1)
    link(fire, add("Condition", name="has_weapon_remaining"), 0)
    link(fire, add("Action", name="fire_missile"), 1)
    link(engage_or_track, add("Action", name="lock_target"), 1)

    # L4 默认巡逻
    patrol = add("Sequence", label="Default_Patrol")
    link(root, patrol, order)
    link(patrol, add("Action", name="search_target", args={"radar_mode": "auto"}), 0)
    link(patrol, add("Action", name="maintain_formation"), 1)

    return {"bt_id": "mock", "mission": "离线确定性生成（mock coder）",
            "nodes": nodes, "edges": edges}


def coder_response(context: dict) -> str:
    """Mock Coder：输入反馈上下文，输出行为树 JSON 文本。"""
    missing = list(context.get("missing") or [])
    feedback = str(context.get("feedback") or "")
    # 亦识别反馈文本中的节点名（贴近真实 LLM 的“按反馈修正”行为）
    for name in ("evade_missile", "is_missile_incoming", "is_fuel_low"):
        if name in feedback and name not in missing:
            missing.append(name)
    tree = _coder_tree(missing)
    return "```json\n" + json.dumps(tree, ensure_ascii=False, indent=2) + "\n```"


def tester_response(context: dict) -> str:
    """Mock Tester：输出 5 组测试用例 JSON 文本。"""
    cases = []
    for i, seed in enumerate(DEFAULT_TEST_SEEDS, 1):
        cases.append({
            "case_id": f"case_{i}",
            "description": f"随机初始态势 seed={seed}（距离 30~80 km、相对方位 ±60°）",
            "expected_behavior": "探测→锁定→进入攻击区发射；受威胁时规避；全程无安全违规",
            "seed": seed,
            "success_criteria": [dict(c) for c in _SAFE_CRITERIA],
        })
    return "```json\n" + json.dumps({"test_cases": cases}, ensure_ascii=False,
                                    indent=2) + "\n```"


def respond(task: str, context: dict) -> str:
    if task == "coder":
        return coder_response(context or {})
    if task == "tester":
        return tester_response(context or {})
    raise ValueError(f"mock LLM 不支持任务 {task!r}")
