# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/schema.py（步骤 2.5 交付物 D2.5-1 组件：行为树 JSON 规范与校验）
================================================================================
本模块是 LLM 输出行为树 JSON 的**唯一规范来源**：
- 节点/动作/条件目录（与《大创项目实施计划书》步骤 2.5 可用节点清单一致）；
- JSON 结构校验（结构合法、无环、单根、引用完整、参数合法）；
- OODA 关键节点覆盖定义与收敛判据；
- bt_validator（BTGenBot 纯 Python 重写）复用的节点库目录。

JSON 格式（nodes + edges，对齐《研究计划与实施方案》§2.1.2 输出格式）：

{
  "bt_id": "bt_v1",
  "mission": "任务描述",
  "nodes": [
    {"id": "n1", "type": "Selector", "label": "BVR_Mission"},
    {"id": "n2", "type": "Action", "name": "navigate_to_waypoint",
     "args": {"altitude_m": 8000, "speed_mps": 250, "heading_deg": 0}},
    {"id": "n3", "type": "Condition", "name": "has_target_detected"}
  ],
  "edges": [
    {"from": "n1", "to": "n2", "order": 0},
    {"from": "n1", "to": "n3", "order": 1}
  ]
}

语义约定：
- edges 方向为父→子，order 为兄弟顺序（缺省按数组顺序）；
- 组合节点（Selector/Sequence/Parallel）按 order 依次执行；
- Action/Condition 为叶子节点，不允许有子节点；
- 节点名来自下列目录，参数缺省使用默认值。
"""
from __future__ import annotations

from collections import Counter, defaultdict

SCHEMA_VERSION = "bt_v1"

NODE_TYPES = ("Selector", "Sequence", "Parallel", "Condition", "Action")

# ---- 可用动作（名称 → 参数名 → 默认值；数值参数类型为 float，其余为 str）----
ACTIONS = {
    "navigate_to_waypoint": {"altitude_m": 8000.0, "speed_mps": 250.0, "heading_deg": 0.0},
    "search_target": {"radar_mode": "auto"},
    "lock_target": {},
    "fire_missile": {},
    "evade_missile": {"maneuver_type": "break_turn"},
    "maintain_formation": {"offset_x_m": 0.0, "offset_y_m": 0.0, "offset_z_m": 0.0},
    "return_to_base": {},
}

# ---- 可用条件 ----
CONDITIONS = (
    "has_target_detected",
    "is_in_launch_zone",
    "is_missile_incoming",
    "has_weapon_remaining",
    "is_fuel_low",
    "has_human_command",
)

# ---- 组合节点子节点数下限（Selector/Sequence 至少 1，Parallel 至少 2）----
MIN_CHILDREN = {"Selector": 1, "Sequence": 1, "Parallel": 2}

# ---- OODA 环关键节点（阶段二验收：覆盖率 > 95% 即 8/8 全含）----
OODA_REQUIRED = {
    "observe": ("search_target", "has_target_detected"),
    "orient": ("lock_target", "is_in_launch_zone"),
    "decide": ("has_weapon_remaining", "is_fuel_low"),
    "act": ("fire_missile", "evade_missile"),
}
OODA_ALL = tuple(n for group in OODA_REQUIRED.values() for n in group)

# ---- 迭代收敛判据（计划书步骤 2.5：通过率 > 90%、OODA 覆盖 > 95%、安全违规 = 0）----
CONVERGENCE = {"pass_rate": 0.90, "ooda_coverage": 0.95, "safety_violations": 0}

# ---- bt_validator 节点库目录（XML 静态校验用；Selector 在 BT.CPP 中即 Fallback）----
VALIDATOR_CATALOG = {
    "actions": sorted(ACTIONS),
    "conditions": sorted(CONDITIONS),
    "controls": ["Sequence", "Fallback", "Parallel"],
}

_XML_TAG_BY_TYPE = {"Selector": "Fallback", "Sequence": "Sequence", "Parallel": "Parallel"}


def xml_tag_for(node_type: str) -> str:
    """JSON 节点类型 → BehaviorTree.CPP XML 标签（Selector → Fallback）。"""
    return _XML_TAG_BY_TYPE.get(node_type, node_type)


def node_name(node: dict) -> str:
    """节点显示名：Action/Condition 用动作/条件名，组合节点用 label 或类型。"""
    return node.get("name") or node.get("label") or node.get("type", "?")


def tree_node_names(data: dict) -> list:
    """树内全部 Action/Condition 名称（去重、保持出现顺序）。"""
    names, seen = [], set()
    for node in data.get("nodes", []):
        n = node.get("name")
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    return names


def ooda_coverage(data: dict):
    """返回 (覆盖率, 缺失节点列表)：关键节点在树中存在即计入（存在且可实例化）。"""
    present = set(tree_node_names(data))
    missing = [n for n in OODA_ALL if n not in present]
    coverage = (len(OODA_ALL) - len(missing)) / len(OODA_ALL)
    return coverage, missing


def _root_and_children(data: dict):
    """返回 (root_id, children: dict[parent] -> [(order, child_id)], errors)。"""
    errors = []
    nodes = data.get("nodes")
    edges = data.get("edges")
    ids = [n.get("id") for n in nodes]
    id_set = set(ids)
    if len(id_set) != len(ids):
        dup = [i for i, c in Counter(ids).items() if c > 1]
        errors.append(f"节点 id 重复: {dup}")
    children = defaultdict(list)
    parents = defaultdict(list)
    for e in edges:
        f, t = e.get("from"), e.get("to")
        if f not in id_set or t not in id_set:
            errors.append(f"边引用不存在的节点: {f} -> {t}")
            continue
        if f == t:
            errors.append(f"自环边: {f}")
            continue
        children[f].append((e.get("order", len(children[f])), t))
        parents[t].append(f)
    roots = [i for i in ids if not parents[i]]
    if len(roots) != 1:
        errors.append(f"应有且仅有一个根节点（无入边），实际 {len(roots)}: {roots}")
        root = roots[0] if roots else None
    else:
        root = roots[0]
    # 单父约束（树）
    multi = {i: p for i, p in parents.items() if len(p) > 1}
    if multi:
        errors.append(f"节点存在多个父节点（非树结构）: {multi}")
    for i in ids:
        children[i].sort(key=lambda x: x[0])
    return root, {k: [c for _, c in v] for k, v in children.items()}, errors


def root_and_children(data: dict):
    """公开入口：返回 (root_id, children: dict[parent] -> [child_id])。

    结构非法时 root_id 可能为 None；详细错误见 validate_tree_json()。
    """
    root, children, _ = _root_and_children(data)
    return root, children


def validate_tree_json(data) -> list:
    """校验行为树 JSON，返回错误列表（空列表 = 通过）。"""
    errors = []
    if not isinstance(data, dict):
        return ["行为树 JSON 顶层应为对象"]
    nodes, edges = data.get("nodes"), data.get("edges")
    if not isinstance(nodes, list) or not nodes:
        errors.append("缺少非空 nodes 列表")
    if not isinstance(edges, list):
        errors.append("缺少 edges 列表")
    if errors:
        return errors

    by_id = {}
    for n in nodes:
        nid, ntype = n.get("id"), n.get("type")
        if not isinstance(nid, str) or not nid:
            errors.append(f"节点缺少合法 id: {n}")
            continue
        by_id[nid] = n
        if ntype not in NODE_TYPES:
            errors.append(f"节点 {nid} 类型非法: {ntype!r}（可用 {NODE_TYPES}）")
            continue
        if ntype in ("Action", "Condition"):
            name = n.get("name")
            catalog = ACTIONS if ntype == "Action" else CONDITIONS
            if name not in catalog:
                errors.append(f"节点 {nid} 的 {ntype} 名称非法: {name!r}")
            elif ntype == "Action":
                spec = ACTIONS[name]
                args = n.get("args") or {}
                if not isinstance(args, dict):
                    errors.append(f"节点 {nid} args 应为对象")
                else:
                    unknown = [k for k in args if k not in spec]
                    if unknown:
                        errors.append(f"节点 {nid}（{name}）参数未知: {unknown}")
                    for k, default in spec.items():
                        if k in args and isinstance(default, float) and \
                                not isinstance(args[k], (int, float)):
                            errors.append(f"节点 {nid}（{name}）参数 {k} 应为数值")
    if errors:
        return errors

    root, children, struct_errors = _root_and_children(data)
    errors.extend(struct_errors)

    for nid, n in by_id.items():
        ntype = n.get("type")
        nchild = len(children.get(nid, []))
        if ntype in ("Action", "Condition") and nchild:
            errors.append(f"叶子节点 {nid}（{node_name(n)}）不应有子节点（实际 {nchild}）")
        elif ntype in MIN_CHILDREN and nchild < MIN_CHILDREN[ntype]:
            errors.append(f"组合节点 {nid}（{ntype}）子节点数 {nchild} < {MIN_CHILDREN[ntype]}")

    # 连通性（从根可达全部节点；配合单父约束可排除环）
    if root is not None:
        seen, stack = set(), [root]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(children.get(cur, []))
        unreachable = [i for i in by_id if i not in seen]
        if unreachable:
            errors.append(f"存在从根不可达的节点（或存在环）: {unreachable}")
    return errors
