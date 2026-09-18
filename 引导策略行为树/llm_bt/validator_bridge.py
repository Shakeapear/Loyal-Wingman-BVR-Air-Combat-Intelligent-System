# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/validator_bridge.py（步骤 2.5 组件：bt_validator 复用）
================================================================
静态校验流水线（对应计划书步骤 2.5「复用 Python 版 bt_validator 做结构校验」）：

1. JSON 规范校验（llm_bt.schema.validate_tree_json：节点目录/引用/单根/无环）；
2. JSON → BehaviorTree.CPP XML（Selector→Fallback，动作参数转为 XML 属性）；
3. 载入 BTGenBot 的纯 Python bt_validator（按文件路径导入，非包依赖），
   以本项目的 BVR 节点库目录（VALIDATOR_CATALOG）执行 XML 结构校验。

返回 (errors, warnings)：errors 非空即校验失败，errors+warnings 可直接作为
下一轮 Coder 反馈的一部分。
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from .schema import VALIDATOR_CATALOG, validate_tree_json
from .tree_io import tree_to_xml

_REPO_ROOT = Path(__file__).resolve().parents[2]
BT_VALIDATOR_PATH = (_REPO_ROOT / "开源项目库" / "03_行为树与LLM" / "BTGenBot"
                     / "Python重写_bt_validator" / "bt_validator.py")


def load_bt_validator():
    """按文件路径加载 BTGenBot bt_validator（纯 Python 重写）。"""
    if not BT_VALIDATOR_PATH.is_file():
        raise FileNotFoundError(
            f"未找到 bt_validator（{BT_VALIDATOR_PATH}）；"
            f"见 开源项目库/03_行为树与LLM/BTGenBot/Python重写_bt_validator/README.md")
    spec = importlib.util.spec_from_file_location("btgenbot_bt_validator", BT_VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_tree_static(data: dict):
    """JSON + XML(bt_validator) 双层静态校验，返回 (errors, warnings)。"""
    errors = validate_tree_json(data)
    if errors:
        return errors, []
    xml_text = tree_to_xml(data)
    validator = load_bt_validator()
    with tempfile.TemporaryDirectory(prefix="bt_static_") as tmp:
        xml_path = Path(tmp) / "bt.xml"
        catalog_path = Path(tmp) / "catalog.json"
        xml_path.write_text(xml_text, encoding="utf-8")
        catalog_path.write_text(json.dumps(VALIDATOR_CATALOG, ensure_ascii=False),
                                encoding="utf-8")
        report = validator.validate(str(xml_path), str(catalog_path))
    return list(report.errors), list(report.warnings)
