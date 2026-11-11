# -*- coding: utf-8 -*-
"""测试路径引导：无论从仓库根还是本目录运行 pytest，都可导入 agents/common。"""
import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[1]
_ROOT = _MODULE.parent
for _path in (_MODULE, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
