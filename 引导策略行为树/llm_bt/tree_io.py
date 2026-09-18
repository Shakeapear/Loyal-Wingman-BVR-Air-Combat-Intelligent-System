# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/tree_io.py（步骤 2.5 组件：树格式转换与可视化）
================================================================
- tree_to_xml      : JSON → BehaviorTree.CPP XML（供 bt_validator 静态校验复用）
- text_tree        : JSON → 文本树形图（D2.5 可视化工具：终端可读）
- tree_to_dot      : JSON → graphviz .dot 源文件（可选渲染 PNG/PDF）
- describe_tree    : JSON → 馈入 Tester Prompt 的树描述文本
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from .schema import node_name, root_and_children, xml_tag_for


def tree_to_xml(data: dict, tree_id: str = "MainTree") -> str:
    """行为树 JSON → BehaviorTree.CPP XML 文本（Selector 映射为 Fallback）。"""
    root_id, children = root_and_children(data)
    if root_id is None:
        raise ValueError("行为树没有根节点，无法转换为 XML")
    by_id = {n["id"]: n for n in data["nodes"]}

    def build(nid):
        node = by_id[nid]
        ntype = node["type"]
        # 叶子节点在 BehaviorTree.CPP 中以动作/条件类名作为标签（如 <fire_missile/>）
        tag = node["name"] if ntype in ("Action", "Condition") else xml_tag_for(ntype)
        attrib = {}
        if node.get("label"):
            attrib["label"] = str(node["label"])
        if ntype == "Action":
            for k, v in (node.get("args") or {}).items():
                attrib[k] = str(v)
        elem = ET.Element(tag, attrib)
        for c in children.get(nid, []):
            elem.append(build(c))
        return elem

    bt = ET.Element("BehaviorTree", {"ID": tree_id})
    bt.append(build(root_id))
    root = ET.Element("root", {"main_tree_to_execute": tree_id})
    root.append(bt)
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def _arg_text(node: dict) -> str:
    args = node.get("args") or {}
    if not args:
        return ""
    return "(" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"


def _walk_text(data: dict, nid: str, children: dict, by_id: dict, prefix: str,
               is_last: bool, lines: list):
    node = by_id[nid]
    ntype = node["type"]
    if ntype in ("Action", "Condition"):
        head = f"{node_name(node)}{_arg_text(node)}"
    else:
        head = f"{ntype}" + (f" [{node['label']}]" if node.get("label") else "")
    connector = "└── " if is_last else "├── "
    lines.append(prefix + connector + head)
    kids = children.get(nid, [])
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, c in enumerate(kids):
        _walk_text(data, c, children, by_id, child_prefix, i == len(kids) - 1, lines)


def text_tree(data: dict) -> str:
    """行为树 JSON → 文本树形图（含动作参数）。"""
    root_id, children = root_and_children(data)
    if root_id is None:
        return "（空树：没有根节点）"
    by_id = {n["id"]: n for n in data["nodes"]}
    root = by_id[root_id]
    head = root["type"] + (f" [{root['label']}]" if root.get("label") else "")
    lines = [head]
    kids = children.get(root_id, [])
    for i, c in enumerate(kids):
        _walk_text(data, c, children, by_id, "", i == len(kids) - 1, lines)
    return "\n".join(lines)


def describe_tree(data: dict) -> str:
    """给 Tester Prompt 的树描述：保留顺序与动作参数（文本树形图即满足）。"""
    mission = data.get("mission")
    body = text_tree(data)
    return f"任务: {mission}\n{body}" if mission else body


def tree_to_dot(data: dict, name: str = "bt") -> str:
    """行为树 JSON → graphviz DOT 文本（组合节点方形、动作圆角、条件菱形近似）。"""
    root_id, children = root_and_children(data)
    if root_id is None:
        raise ValueError("行为树没有根节点，无法导出 DOT")
    by_id = {n["id"]: n for n in data["nodes"]}
    lines = [f'digraph "{name}" {{', "  rankdir=TB;", "  node [fontname=\"Microsoft YaHei\"];"]
    for nid, node in by_id.items():
        label = node_name(node) + _arg_text(node).replace('"', "'")
        if node["type"] in ("Action", "Condition"):
            shape = "ellipse" if node["type"] == "Action" else "diamond"
        elif node["type"] == "Parallel":
            shape = "parallelogram"
        else:
            shape = "box"
        lines.append(f'  "{nid}" [label="{label}", shape={shape}];')
    for parent, kids in children.items():
        for c in kids:
            lines.append(f'  "{parent}" -> "{c}";')
    lines.append("}")
    return "\n".join(lines)
