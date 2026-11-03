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

包级路径引导（接口规范：sys.path 操作集中一处）：本包依赖智能体库与项目级
契约模块（`common.*`，位于 多任务智能体/）。
"""
import sys as _sys
from pathlib import Path as _Path

_AGENT_LIB = _Path(__file__).resolve().parents[2] / "多任务智能体"
if _AGENT_LIB.is_dir() and str(_AGENT_LIB) not in _sys.path:
    _sys.path.append(str(_AGENT_LIB))

from .executor import Blackboard, build_tree, tick_once
from .guide_provider import BTGuideProvider
from .guide_types import GuideStep
from .maneuvers import ManeuverRegistry, ManeuverSpec
from .schema import CONVERGENCE, OODA_ALL, validate_tree_json
from .tree_registry import load_manifest, resolve_tree
from .validator_bridge import validate_tree_static

__all__ = [
    "Blackboard", "build_tree", "tick_once",
    "BTGuideProvider", "GuideStep",
    "ManeuverRegistry", "ManeuverSpec",
    "load_manifest", "resolve_tree",
    "CONVERGENCE", "OODA_ALL", "validate_tree_json", "validate_tree_static",
]
