# -*- coding: utf-8 -*-
"""场景 → 教师树清单（scene-specific tree routing）。

一个专家智能体可以拥有多棵场景树；本模块只负责按 scene 取树，
不做训练。默认清单：引导策略行为树/tree_manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

from .tactical_execution_tree import tactical_tree_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "tree_manifest.json"


def load_manifest(path: str | Path | None = None) -> dict:
    target = Path(path) if path else DEFAULT_MANIFEST
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"树清单必须为对象：{target}")
    return data


def resolve_tree(scene: str, manifest: dict | None = None) -> dict:
    manifest = manifest or load_manifest()
    if scene not in manifest:
        raise KeyError(f"场景 {scene!r} 未配置教师树；可用 {sorted(manifest)}")
    entry = manifest[scene]
    if entry.get("builder") == "tactical_tree_json":
        return tactical_tree_json()
    path = entry.get("path")
    if not path:
        raise ValueError(f"场景 {scene!r} 缺少 path/builder")
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return json.loads(target.read_text(encoding="utf-8"))


__all__ = ["DEFAULT_MANIFEST", "load_manifest", "resolve_tree"]
