# -*- coding: utf-8 -*-
"""仿真入口共享工具（仅服务 run_*.py 编排层）。

把四个入口脚本重复的种子解析、EpisodeRecorder 初始化、边界守卫构造和
ACMI 导出收敛到一处；不改变 llm_bt 运行时契约，也不改变各脚本 CLI。
"""
from __future__ import annotations

import json
from pathlib import Path

from common.constants import GEO_LAT0_DEG, GEO_LON0_DEG

from .boundary_constraints import BoundaryConstraints


def parse_seeds(spec: str) -> list[int]:
    """解析 ``1-20,25`` 形式的种子表达式，返回去重升序列表。"""
    seeds: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def geo_meta() -> dict:
    """EpisodeRecorder 使用的经纬原点元数据。"""
    return {"lat0_deg": GEO_LAT0_DEG, "lon0_deg": GEO_LON0_DEG}


def make_recorder():
    """创建带统一 GEO 元数据的逐帧记录器（惰性导入可视化包）。"""
    from visualization.recorder import EpisodeRecorder

    return EpisodeRecorder(meta=geo_meta())


def build_guard(env, config: dict | None = None) -> BoundaryConstraints:
    """按环境安全阈值构造边界守卫；config 仅覆盖守卫自身参数。"""
    return BoundaryConstraints(config=config, env_cfg=env.cfg)


def export_episode_acmi(frames, path, title: str):
    """用统一 GEO 原点导出 TacView ACMI（惰性导入可视化包）。"""
    from visualization.acmi import export_acmi

    return export_acmi(frames, path, lat0_deg=GEO_LAT0_DEG,
                       lon0_deg=GEO_LON0_DEG, title=title)


def write_json(path, payload) -> Path:
    """以仓库统一的 UTF-8/中文不转义/2 空格缩进格式写 JSON。"""
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
