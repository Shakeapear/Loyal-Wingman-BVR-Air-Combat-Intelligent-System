# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/prompts.py（步骤 2.5 交付物 D2.5-2 装载器）
================================================================
读取 prompts/*.md 模板并填充占位符；模板文件即交付物本体（≥1KB/份）。
"""
from __future__ import annotations

from pathlib import Path

from .tree_io import describe_tree

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

SYSTEM_CODER = "你是超视距空战行为树设计专家。严格只输出一个 JSON 对象，不要输出任何解释文字。"
SYSTEM_TESTER = "你是空战仿真测试工程师。严格只输出一个 JSON 对象（含 test_cases），不要输出任何解释文字。"


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt 模板缺失: {path}")
    return path.read_text(encoding="utf-8")


def render_coder_prompt(mission: str, feedback: str = "") -> str:
    """Coder Prompt：任务描述 + 上一轮反馈（首轮传空）。"""
    return (load_prompt("coder_prompt.md")
            .replace("{mission_description}", mission)
            .replace("{feedback}", feedback or "（无，首轮生成）"))


def render_tester_prompt(tree_json: dict) -> str:
    """Tester Prompt：注入行为树描述（文本树形图 + 动作参数）。"""
    return (load_prompt("tester_prompt.md")
            .replace("{bt_description}", describe_tree(tree_json)))
