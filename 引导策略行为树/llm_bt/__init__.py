# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt（步骤 2.5 交付物 D2.5-1 框架实现包）
================================================================
LLM 行为树生成器-测试器框架（Coder-Executor-Tester）：

- schema            行为树 JSON 规范、目录、校验、OODA 覆盖与收敛判据
- prompts           Prompt 模板装载（prompts/*.md = D2.5-2）
- llm_client        LLM 接入（deepseek / openai / mock 离线）
- mock_llm          离线确定性 Coder/Tester（无 Key 开发与单测）
- executor          JSON → py_trees → BVRCombatEnv 执行器与指令回路
- tester            LLM 测试用例生成 + 仿真执行评估
- validator_bridge  JSON → XML → 复用 BTGenBot bt_validator 静态校验
- tree_io           文本树形图 / DOT / BehaviorTree.CPP XML 转换
"""

from .executor import Blackboard, build_tree, tick_once
from .schema import CONVERGENCE, OODA_ALL, validate_tree_json
from .validator_bridge import validate_tree_static

__all__ = [
    "Blackboard", "build_tree", "tick_once",
    "CONVERGENCE", "OODA_ALL", "validate_tree_json", "validate_tree_static",
]
