# -*- coding: utf-8 -*-
"""行为树/Tester/Generator 共用的场景定义。"""
from __future__ import annotations

SCENES = {
    "default": {
        "label": "30–80 km",
        "max_steps": 240,
    },
    "bvr80-120": {
        "label": "80–120 km",
        "max_steps": 600,
        "scene_dist_min_m": 80000.0,
        "scene_dist_max_m": 120000.0,
    },
}


def scene_spec(scene: str) -> dict:
    if scene not in SCENES:
        raise ValueError(f"未知场景 {scene!r}；可用 {sorted(SCENES)}")
    return SCENES[scene]


def scene_label(scene: str) -> str:
    return str(scene_spec(scene)["label"])


def scene_max_steps(scene: str) -> int:
    return int(scene_spec(scene)["max_steps"])


def env_config(scene: str, *, compute_envelope: bool = True,
               max_steps: int | None = None) -> dict:
    spec = scene_spec(scene)
    cfg = {
        "compute_envelope": bool(compute_envelope),
        "max_steps": int(max_steps or spec["max_steps"]),
    }
    for key in ("scene_dist_min_m", "scene_dist_max_m"):
        if key in spec:
            cfg[key] = float(spec[key])
    return cfg


__all__ = ["SCENES", "env_config", "scene_label", "scene_max_steps",
           "scene_spec"]
