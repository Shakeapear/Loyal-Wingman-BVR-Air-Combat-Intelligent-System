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


# ----------------------------------------------------------------------
# v3：招式丰富的对抗教师树（2026-11-04）
# ----------------------------------------------------------------------
V3_PROFILES = {
    # 先手用 fire_missile，NEZ 补射用 launch_and_crank；远距 notch、近距 break_turn
    "balanced": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "notch",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "intercept_speed_mps": 280.0,
        "crank_offset_deg": 60.0,
        "pressure_full_only": True,
    },
    # 老式两发 Rmax 压力射击 + v3 防御/边界/搜索（head-to-head 取向）
    "pressure_all": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "break_turn",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "intercept_speed_mps": 280.0,
        "crank_offset_deg": 60.0,
        "pressure_full_only": False,
    },
    # 先手直接用 launch_and_crank，NEZ 补射用 fire_missile
    "aggressive": {
        "first_shot_action": "launch_and_crank",
        "nez_action": "fire_missile",
        "defense_long": "notch",
        "defense_short": "break_turn",
        "search_sweep_deg": 60.0,
        "search_period_s": 18.0,
        "intercept_speed_mps": 300.0,
        "crank_offset_deg": 45.0,
    },
    # 防御 A/B：两侧都用防御螺旋
    "spiral_defense": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "defensive_spiral",
        "defense_short": "defensive_spiral",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "intercept_speed_mps": 280.0,
        "crank_offset_deg": 60.0,
    },
    # 防御 A/B：两侧都用 break_turn（当前手写树的有效方案）
    "break_turn_defense": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "break_turn",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "intercept_speed_mps": 280.0,
        "crank_offset_deg": 60.0,
    },
    # 速度 A/B：v3 实战中射手速度 ~300 m/s，手写树 ~240 m/s；
    # 数据表明“慢射手打快目标”Pk 更高，先扫 240/250/260。
    "speed240": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "break_turn",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "search_speed_mps": 240.0,
        "intercept_speed_mps": 240.0,
        "crank_offset_deg": 60.0,
    },
    "speed250": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "break_turn",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "search_speed_mps": 250.0,
        "intercept_speed_mps": 250.0,
        "crank_offset_deg": 60.0,
    },
    "speed260": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "break_turn",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "search_speed_mps": 260.0,
        "intercept_speed_mps": 260.0,
        "crank_offset_deg": 60.0,
    },
    # 只在“干净窗口”补射：无 MAWS、目标未转弯、本机无弹在飞。
    "shot_window": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "break_turn",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "search_speed_mps": 260.0,
        "intercept_speed_mps": 260.0,
        "crank_offset_deg": 60.0,
        "shot_window_gate": True,
    },
    "speed240_intercept": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "break_turn",
        "defense_short": "break_turn",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "search_speed_mps": 280.0,
        "intercept_speed_mps": 240.0,
        "crank_offset_deg": 60.0,
    },
    # 防御 A/B：两侧都用 notch
    "notch_defense": {
        "first_shot_action": "fire_missile",
        "nez_action": "launch_and_crank",
        "defense_long": "notch",
        "defense_short": "notch",
        "search_sweep_deg": 45.0,
        "search_period_s": 15.0,
        "intercept_speed_mps": 280.0,
        "crank_offset_deg": 60.0,
    },
}


def tactical_tree_v3_json(profile: str = "balanced") -> dict:
    """招式丰富、先手优先的对抗教师树。

    结构（优先级从高到低）：
        Human_Command_Response
        Self_Defense（MAWS → notch/break_turn + ECM）
        Boundary_Return（燃油/近边界 → fuel_rtb）
        Weapons_In_Flight（有弹在飞 → lock + crank）
        First_Shot（动态 LAR + 热姿态 + 高闭合 → fire/launch_and_crank）
        NEZ_Backup（近距离高闭合补射）
        Close_And_Track（lock + intercept_closure）
        Search_Default（search_target + search_pattern）

    OODA 8/8 保留；所有 execute_maneuver 名称均来自 ManeuverRegistry。
    """
    if profile not in V3_PROFILES:
        raise KeyError(f"未知 v3 profile: {profile}；可用 {sorted(V3_PROFILES)}")
    p = V3_PROFILES[profile]
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

    def maneuver(mid, params=None):
        return add("Action", name="execute_maneuver",
                   args={"maneuver": mid, "parameters": params or {}})

    def first_shot_action():
        if p["first_shot_action"] == "fire_missile":
            return add("Action", name="fire_missile")
        return maneuver("launch_and_crank",
                        {"offset_deg": p["crank_offset_deg"]})

    def nez_action():
        if p["nez_action"] == "fire_missile":
            return add("Action", name="fire_missile")
        return maneuver("launch_and_crank",
                        {"offset_deg": p["crank_offset_deg"]})

    root = add("Selector", label="BVR_Mission")
    order = 0

    # L1 真人指令响应
    human = add("Sequence", label="Human_Command_Response")
    link(root, human, order); order += 1
    link(human, add("Condition", name="has_human_command"), 0)
    link(human, add("Action", name="navigate_to_waypoint",
                    args={"altitude_m": 8000, "speed_mps": 250,
                          "heading_deg": 0}), 1)

    # L2 自保：远距 notch / 近距 break_turn + ECM
    defense = add("Sequence", label="Self_Defense")
    link(root, defense, order); order += 1
    link(defense, add("Condition", name="is_missile_incoming"), 0)
    defense_actions = add("Parallel", label="DefensiveActions")
    link(defense, defense_actions, 1)
    maneuver_selector = add("Selector", label="DefensiveManeuver")
    link(defense_actions, maneuver_selector, 0)
    long_tta = add("Sequence", label="NotchWhenLongTTA")
    link(maneuver_selector, long_tta, 0)
    link(long_tta, add("Condition", name="is_tta_long"), 0)
    link(long_tta, add("Action", name="evade_missile",
                        args={"maneuver_type": p["defense_long"]}), 1)
    link(maneuver_selector, add("Action", name="evade_missile",
                                args={"maneuver_type": p["defense_short"]}), 1)
    link(defense_actions, maneuver("ecm_on", {"duration_s": 5}), 1)

    # L2.5 边界/燃油返航（防止 80–120 km 逃逸）
    boundary = add("Sequence", label="Boundary_Return")
    link(root, boundary, order); order += 1
    boundary_trigger = add("Selector", label="BoundaryTrigger")
    link(boundary, boundary_trigger, 0)
    link(boundary_trigger, add("Condition", name="is_fuel_low"), 0)
    link(boundary_trigger, add("Condition", name="is_near_boundary"), 1)
    link(boundary, maneuver("fuel_rtb", {}), 1)

    # L3.0 Rmax 压力首发：满弹时在攻击区内先打一发，抢首发射击权。
    # 这一发不要求 45 km 精度门槛；命中最好，未命中也会迫使对方进入防御，
    # 为后续 45 km/NEZ 的高 Pk 补射创造窗口。
    pressure = add("Sequence", label="Pressure_Shot")
    link(root, pressure, order); order += 1
    link(pressure, add("Condition", name="has_target_detected"), 0)
    link(pressure, add("Condition", name="is_in_launch_zone"), 1)
    link(pressure, add("Condition",
                       name=("has_full_weapons"
                             if p.get("pressure_full_only", True)
                             else "has_weapon_remaining")), 2)
    link(pressure, first_shot_action(), 3)

    # L3.1 第二发压力射击（实验分支，默认关闭）：
    # 若首发后目标仍保持热姿态/高闭合，继续在 Rmax 打；否则由 45 km 精度分支接管。
    # 80–120 km 的脚本敌机会保持热姿态，开启该分支会浪费第二发，故默认关闭。
    if p.get("pressure2", False):
        pressure2 = add("Sequence", label="Pressure_Shot_2")
        link(root, pressure2, order); order += 1
        link(pressure2, add("Condition", name="has_target_detected"), 0)
        link(pressure2, add("Condition", name="is_in_launch_zone"), 1)
        link(pressure2, add("Condition", name="is_target_hot"), 2)
        link(pressure2, add("Condition", name="is_closure_high"), 3)
        link(pressure2, add("Condition", name="has_weapon_remaining"), 4)
        link(pressure2, first_shot_action(), 5)

    # L3 第一发：Rmax 热姿态先手（动态 LAR）
    # 注意：先手/补射故意放在“弹在飞 crank”之前，允许 shoot-look-shoot；
    # 否则第一发在飞期间会被 crank 分支长期占用，无法补第二发。
    first = add("Sequence", label="First_Shot")
    link(root, first, order); order += 1
    sub = 0
    for cond in ("has_target_detected", "is_in_launch_zone",
                 "is_within_dynamic_lar", "is_target_hot",
                 "is_closure_high"):
        link(first, add("Condition", name=cond), sub); sub += 1
    if p.get("shot_window_gate", False):
        link(first, add("Condition", name="is_shot_window_open"), sub)
        sub += 1
    link(first, add("Condition", name="has_weapon_remaining"), sub)
    link(first, first_shot_action(), sub + 1)

    # L3.5 兜底先手：不强制热姿态，只要闭合率足够高就发射。
    # 脚本敌机在初始态势可能是 beam 但高闭合，手写树在 Rmax 直接发射仍能命中；
    # 该分支保留“先手权”，避免为了等 hot 而晚 3–5 s 被对方先打。
    closure_shot = add("Sequence", label="First_Shot_Closure")
    link(root, closure_shot, order); order += 1
    sub = 0
    for cond in ("has_target_detected", "is_in_launch_zone",
                 "is_within_effective_range", "is_closure_high"):
        link(closure_shot, add("Condition", name=cond), sub); sub += 1
    if p.get("shot_window_gate", False):
        link(closure_shot, add("Condition", name="is_shot_window_open"), sub)
        sub += 1
    link(closure_shot, add("Condition", name="has_weapon_remaining"), sub)
    link(closure_shot, first_shot_action(), sub + 1)

    # L4 近距离补射：NEZ + 高闭合
    nez = add("Sequence", label="NEZ_Backup")
    link(root, nez, order); order += 1
    sub = 0
    for cond in ("has_target_detected", "is_in_launch_zone",
                 "is_in_nez", "is_closure_high"):
        link(nez, add("Condition", name=cond), sub); sub += 1
    if p.get("shot_window_gate", False):
        link(nez, add("Condition", name="is_shot_window_open"), sub)
        sub += 1
    link(nez, add("Condition", name="has_weapon_remaining"), sub)
    link(nez, nez_action(), sub + 1)

    # L5 弹在飞：保持跟踪并 crank（先手/补射失败后才进入）
    weapons = add("Sequence", label="Weapons_In_Flight")
    link(root, weapons, order); order += 1
    link(weapons, add("Condition", name="has_missile_in_flight"), 0)
    link(weapons, add("Action", name="lock_target"), 1)
    link(weapons, maneuver("crank", {"offset_deg": p["crank_offset_deg"],
                                     "speed_mps": 250.0}), 2)

    # L6 闭合/跟踪
    close = add("Sequence", label="Close_And_Track")
    link(root, close, order); order += 1
    link(close, add("Condition", name="has_target_detected"), 0)
    link(close, add("Action", name="lock_target"), 1)
    link(close, maneuver("intercept_closure",
                         {"speed_mps": p["intercept_speed_mps"]}), 2)

    # L7 边界感知搜索
    search = add("Sequence", label="Search_Default")
    link(root, search, order)
    link(search, add("Action", name="search_target",
                     args={"radar_mode": "auto"}), 0)
    link(search, maneuver("search_pattern",
                          {"sweep_deg": p["search_sweep_deg"],
                           "sweep_period_s": p["search_period_s"],
                           "speed_mps": p.get("search_speed_mps", 280.0)}), 1)

    return {
        "bt_id": f"bt_tactical_v3_{profile}",
        "mission": ("对抗教师树 v3：动态 LAR 先手发射、弹在飞 crank、"
                    "边界感知搜索、人类招式防御/返航，用于 JSRL 引导。"),
        "profile": profile,
        "nodes": nodes,
        "edges": edges,
    }


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
           "tactical_tree_v3_json", "V3_PROFILES", "build_tactical_tree",
           "tactical_coverage", "validate_tactical_tree"]
