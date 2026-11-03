# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/tactical_execution_tree.py（步骤 2.6 D2.6-2）
================================================================
第二优先级战术执行子树：以计划书 §2.2.2 的 BVR 行为树为骨架，用现有
`llm_bt.schema` 动作/条件节点实现，覆盖完整 OODA 环。

结构：
    Selector [BVR_Mission]
    ├── Sequence [Human_Command_Response]   has_human_command → navigate_to_waypoint
    ├── Sequence [Self_Defense]             is_missile_incoming → evade_missile
    ├── Sequence [Fuel_Return]              is_fuel_low → return_to_base
    ├── Sequence [BVR_Engagement]
    │   ├── Selector [Detection]            (has_target_detected → lock_target) / search_target
    │   └── Selector [Engagement]
    │       ├── Sequence [Launch]           is_in_launch_zone → (has_weapon_remaining → fire_missile)
    │       └── lock_target
    └── Sequence [Default_Patrol]           search_target → maintain_formation

计划书节点映射（未新增 schema 叶子，避免死节点）：
    ExecuteCommand        → has_human_command + navigate_to_waypoint
    AssessThreat          → is_missile_incoming
    SetOffensivePosture   → lock_target
    SelectWeaponType      → has_weapon_remaining（单武器槽，接口预留）
    ManeuverToLaunchZone  → lock_target（跟踪/占位）
    InNoEscapeZone        → is_in_launch_zone（环境攻击区）
    ConsiderFire          → fire_missile
    NavigateToPatrolPoint → navigate_to_waypoint
    ExecuteSearchPattern  → search_target

OODA 覆盖：search_target / has_target_detected / lock_target / is_in_launch_zone /
has_weapon_remaining / is_fuel_low / fire_missile / evade_missile 共 8/8。
"""
from __future__ import annotations

from .executor import build_tree
from .schema import OODA_ALL, ooda_coverage, validate_tree_json


def tactical_tree_json(maneuver_mode: bool = False) -> dict:
    """返回战术执行子树 JSON（确定性，不依赖在线 LLM）。

    maneuver_mode=True 时，搜索/接敌分支额外调用参数化招式：
    search_pattern / intercept_closure / lock_and_track；旧 OODA 节点保留，
    便于 80–120 km 场景复用同一验收口径。
    """
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

    # L2 应急自保
    defense = add("Sequence", label="Self_Defense")
    link(root, defense, order); order += 1
    link(defense, add("Condition", name="is_missile_incoming"), 0)
    link(defense, add("Action", name="evade_missile",
                      args={"maneuver_type": "defensive_spiral"
                            if maneuver_mode else "break_turn"}), 1)

    # L2.5 燃油约束
    fuel = add("Sequence", label="Fuel_Return")
    link(root, fuel, order); order += 1
    link(fuel, add("Condition", name="is_fuel_low"), 0)
    link(fuel, add("Action", name="return_to_base"), 1)

    # L3 超视距交战（OODA：观察→判断→决策→行动）
    engage = add("Sequence", label="BVR_Engagement")
    link(root, engage, order); order += 1

    detection = add("Selector", label="Detection")
    link(engage, detection, 0)
    track = add("Sequence", label="Track_Target")
    link(detection, track, 0)
    link(track, add("Condition", name="has_target_detected"), 0)
    link(track, add("Action", name="lock_target"), 1)
    if maneuver_mode:
        link(track, add("Action", name="execute_maneuver",
                        args={"maneuver": "intercept_closure",
                              "parameters": {"speed_mps": 280}}), 2)

    if maneuver_mode:
        search_seq = add("Sequence", label="Search_Pattern")
        link(detection, search_seq, 1)
        link(search_seq, add("Action", name="search_target",
                             args={"radar_mode": "auto"}), 0)
        link(search_seq, add("Action", name="execute_maneuver",
                             args={"maneuver": "search_pattern",
                                   "parameters": {"sweep_deg": 90,
                                                  "sweep_period_s": 20}}), 1)
    else:
        link(detection, add("Action", name="search_target",
                            args={"radar_mode": "auto"}), 1)

    engagement = add("Selector", label="Engagement")
    link(engage, engagement, 1)
    launch = add("Sequence", label="Launch")
    link(engagement, launch, 0)
    link(launch, add("Condition", name="is_in_launch_zone"), 0)
    fire = add("Sequence", label="Fire_Control")
    if maneuver_mode:
        link(launch, add("Condition", name="is_in_nez"), 1)
        link(launch, fire, 2)
    else:
        link(launch, fire, 1)
    link(fire, add("Condition", name="has_weapon_remaining"), 0)
    link(fire, add("Action", name="fire_missile"), 1)
    if maneuver_mode:
        link(engagement, add("Action", name="execute_maneuver",
                             args={"maneuver": "lock_and_track",
                                   "parameters": {}}), 1)
    else:
        link(engagement, add("Action", name="lock_target"), 1)

    # L4 默认巡逻
    patrol = add("Sequence", label="Default_Patrol")
    link(root, patrol, order)
    link(patrol, add("Action", name="search_target", args={"radar_mode": "auto"}), 0)
    link(patrol, add("Action", name="maintain_formation"), 1)

    return {"bt_id": "bt_tactical_2_6",
            "mission": "二级优先级引导：边界约束（外部守卫）+ 战术执行（OODA）",
            "nodes": nodes, "edges": edges}


def tactical_tree_80_120_json() -> dict:
    """80–120 km / 600 步参考树：在战术骨架上加入搜索/接敌招式。"""
    return tactical_tree_json(maneuver_mode=True)


def build_tactical_tree(blackboard):
    """构建 py_trees 战术执行树（根节点）。"""
    return build_tree(tactical_tree_json(), blackboard)


def tactical_coverage():
    """返回 (覆盖率, 缺失节点)。"""
    return ooda_coverage(tactical_tree_json())


def validate_tactical_tree() -> list:
    """静态校验战术树 JSON，返回错误列表（空 = 通过）。"""
    return validate_tree_json(tactical_tree_json())


__all__ = ["OODA_ALL", "tactical_tree_json", "tactical_tree_80_120_json",
           "build_tactical_tree", "tactical_coverage", "validate_tactical_tree"]
