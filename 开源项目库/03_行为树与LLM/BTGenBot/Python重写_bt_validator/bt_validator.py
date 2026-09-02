# -*- coding: utf-8 -*-
"""
bt_validator.py
===============
BTGenBot bt_validator（原版 C++/ROS2 + BehaviorTree.CPP）的纯 Python 重写。

重写说明：
- 原版 bt_validator（开源项目库/03_行为树与LLM/BTGenBot/源码/bt_validator/）使用
  C++(BehaviorTree.CPP) 在 ROS2 中"执行"行为树并在仿真中验证；
- 本项目只需其**静态校验逻辑**：XML 良构性、节点是否在节点库中、树结构合法性
  （单根、组合节点子节点数约束）、动作/条件节点叶子约束；
- 动态执行验证由本项目 JSBSim 环境承担，本模块预留 evaluator 接口。

用法（DC 环境）：
    python bt_validator.py <行为树XML或目录> [--catalog 节点库json]
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# 节点库（源自原版 bt_client 注册节点，C++ 头文件同名类）
DEFAULT_CATALOG = {
    "actions": ["FollowAruco", "GenerateNextDestination", "MoveManipulator",
                "MoveTo", "MoveToWithTimeout"],
    "conditions": ["isExplorationComplete", "isGoalReachable", "Done", "Fail"],
    "controls": ["Sequence", "Fallback", "SequenceStar", "ReactiveSequence",
                 "ReactiveFallback", "Parallel", "Decorator", "Inverter",
                 "RetryUntilSuccessful", "KeepRunningUntilFailure", "Delay",
                 "Timeout", "ForceSuccess", "ForceFailure", "SubTree"],
}

# 组合/装饰节点允许的子节点数范围（下限, 上限, 是否要求叶子）
CONTROL_RULES = {
    "Sequence": (1, None), "Fallback": (1, None), "SequenceStar": (1, None),
    "ReactiveSequence": (1, None), "ReactiveFallback": (1, None),
    "Parallel": (2, None), "Decorator": (1, 1), "Inverter": (1, 1),
    "RetryUntilSuccessful": (1, 1), "KeepRunningUntilFailure": (1, 1),
    "Delay": (1, 1), "Timeout": (1, 1), "ForceSuccess": (1, 1),
    "ForceFailure": (1, 1), "SubTree": (0, 1),
}


class ValidationReport:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def ok(self):
        return len(self.errors) == 0

    def __str__(self):
        lines = [f"校验{'通过' if self.ok() else '失败'}: "
                 f"{len(self.errors)} 错误, {len(self.warnings)} 警告"]
        for e in self.errors:
            lines.append(f"  [错误] {e}")
        for w in self.warnings:
            lines.append(f"  [警告] {w}")
        return "\n".join(lines)


def load_catalog(path=None):
    if path is None:
        return dict(DEFAULT_CATALOG)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_tree_file(xml_path, catalog, report):
    """校验单个行为树 XML 文件（对应原版 validator 的静态检查部分）。

    支持 BehaviorTree.CPP 格式（<root main_tree_to_execute="..."> 内含多个
    <BehaviorTree ID="...">）与单树 <root> 格式。
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        report.errors.append(f"{xml_path.name}: XML 解析失败 {e}")
        return

    all_known = set(catalog.get("actions", []) + catalog.get("conditions", [])
                    + catalog.get("controls", []))

    def classify(tag):
        if tag in catalog.get("controls", []):
            return "controls"
        if tag in catalog.get("conditions", []):
            return "conditions"
        if tag in catalog.get("actions", []):
            return "actions"
        return "unknown"

    def walk(node):
        tag = node.tag.split("}")[-1]
        kind = classify(tag)
        if kind == "unknown":
            report.warnings.append(f"{xml_path.name}: 未知节点类型 <{tag}>（需确认是否已在节点库注册）")

        children = list(node)
        if kind in ("actions", "conditions") and children:
            report.errors.append(f"{xml_path.name}: 叶子节点 <{tag}> 不应有子节点（实际 {len(children)}）")
        if tag in CONTROL_RULES:
            lo, hi = CONTROL_RULES[tag]
            if len(children) < lo or (hi is not None and len(children) > hi):
                report.errors.append(f"{xml_path.name}: <{tag}> 子节点数 {len(children)} 超出 [{lo},{hi}]")
        for c in children:
            walk(c)

    def walk_bt(bt):
        children = list(bt)
        if len(children) != 1:
            report.errors.append(f"{xml_path.name}: <BehaviorTree ID={bt.get('ID')}> "
                                 f"下应有且仅有一个子树根，实际 {len(children)} 个")
        for c in children:
            walk(c)

    if root.tag.split("}")[-1].lower() == "root":
        bts = [c for c in root if c.tag.split("}")[-1] == "BehaviorTree"]
        if bts:
            ids = [bt.get("ID") for bt in bts]
            main = root.get("main_tree_to_execute")
            if main and main not in ids:
                report.errors.append(f"{xml_path.name}: main_tree_to_execute={main} 不在树列表 {ids} 中")
            for bt in bts:
                walk_bt(bt)
        else:
            children = list(root)
            if len(children) != 1:
                report.errors.append(f"{xml_path.name}: 根下应有且仅有一个子树根，实际 {len(children)} 个")
            for c in children:
                walk(c)
    else:
        report.errors.append(f"{xml_path.name}: 根节点应为 <root>，实际 <{root.tag}>")


def validate(path, catalog_path=None):
    """入口：path 为 XML 文件或目录；返回 ValidationReport。"""
    catalog = load_catalog(catalog_path)
    report = ValidationReport()
    p = Path(path)
    files = sorted(p.glob("*.xml")) if p.is_dir() else [p]
    if not files:
        report.errors.append(f"未找到 XML 文件: {path}")
    for f in files:
        validate_tree_file(f, catalog, report)
    return report


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    catalog_path = None
    if "--catalog" in argv:
        i = argv.index("--catalog")
        catalog_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    report = validate(argv[1], catalog_path)
    print(report)
    return 0 if report.ok() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
