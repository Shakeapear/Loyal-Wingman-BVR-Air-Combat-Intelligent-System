# -*- coding: utf-8 -*-
"""agents/model_spec.py
======================
模型归档规格校验：任务、策略类型、观测维度与场景必须与当前评估/续训入口一致。

模型 .zip 是 SB3 pickle，单靠文件名无法判断它来自 P0(mlp/28 维) 还是
P1(hybrid/35 维)。错误组合会在首次 predict 时才以维度错误崩溃，或在同维
不同场景下静默给出错误评估；本模块在 load 后立即失败并给出明确原因。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POLICY_OBS_DIM = {"mlp": 28, "hybrid": 35}


def _candidates(path: str | Path) -> list[Path]:
    p = Path(path)
    paths = [Path(str(p) + ".spec.json"), p.with_suffix(".spec.json")]
    return list(dict.fromkeys(paths))


def load_model_spec(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    for candidate in _candidates(path):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                return data
    return None


def save_model_spec(path: str | Path, spec: dict) -> Path:
    p = Path(path)
    if p.name.endswith(".spec.json"):
        target = p
    elif p.suffix == ".zip":
        target = Path(str(p) + ".spec.json")
    else:
        target = p.with_suffix(".spec.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def normalize_policy(policy: str | None) -> str | None:
    if policy is None:
        return None
    text = str(policy).lower()
    if "hybrid" in text:
        return "hybrid"
    if "mlp" in text or "actorcritic" in text:
        return "mlp"
    return text


def infer_policy(model) -> str:
    name = type(getattr(model, "policy", model)).__name__.lower()
    return "hybrid" if "hybrid" in name else "mlp"


def validate_model_artifact(model, *, policy: str | None = None,
                            task: str | None = None,
                            scene: str | None = None,
                            control_mode: str | None = None,
                            path: str | Path | None = None) -> dict | None:
    """校验已加载模型；不匹配时抛 ValueError。返回读到的 spec（可能为 None）。"""
    expected_policy = normalize_policy(policy)
    actual_policy = infer_policy(model)
    if expected_policy is not None and actual_policy != expected_policy:
        raise ValueError(
            f"模型策略类型不匹配：期望 {expected_policy}，实际 {actual_policy}。"
            "请检查 --policy 与 checkpoint 是否对应。")

    space = getattr(model, "observation_space", None)
    obs_dim = None
    if space is not None and getattr(space, "shape", None) is not None:
        obs_dim = int(np.prod(space.shape))
    expected_dim = POLICY_OBS_DIM.get(expected_policy or actual_policy)
    if expected_dim is not None and obs_dim != expected_dim:
        raise ValueError(
            f"模型观测维度不匹配：期望 {expected_dim}，实际 {obs_dim}。"
            "P0(mlp) 应为 28 维，P1(hybrid) 应为 35 维。")

    spec = load_model_spec(path) if path is not None else None
    if not spec:
        return None

    mission = spec.get("mission") or spec.get("task_id")
    if task is not None and mission is not None and str(mission) != str(task):
        raise ValueError(
            f"模型任务不匹配：期望 task={task}，spec 记录 {mission}。")
    spec_policy = normalize_policy(spec.get("policy"))
    if spec_policy is not None and expected_policy is not None \
            and spec_policy != expected_policy:
        raise ValueError(
            f"模型 spec 策略不匹配：期望 {expected_policy}，spec 记录 {spec_policy}。")
    obs_spec = spec.get("observation") or {}
    spec_dim = obs_spec.get("dim")
    if spec_dim is not None and obs_dim is not None and int(spec_dim) != obs_dim:
        raise ValueError(
            f"模型 spec 观测维度不匹配：模型 {obs_dim}，spec {spec_dim}。")
    spec_scene = spec.get("scene")
    if scene is not None and spec_scene is not None and str(spec_scene) != str(scene):
        raise ValueError(
            f"模型场景不匹配：期望 scene={scene}，spec 记录 {spec_scene}。")
    spec_cm = spec.get("control_mode")
    if control_mode is not None and spec_cm is not None \
            and str(spec_cm) != str(control_mode):
        raise ValueError(
            f"模型控制模式不匹配：期望 {control_mode}，spec 记录 {spec_cm}。")
    return spec


__all__ = ["POLICY_OBS_DIM", "infer_policy", "load_model_spec",
           "normalize_policy", "save_model_spec", "validate_model_artifact"]
