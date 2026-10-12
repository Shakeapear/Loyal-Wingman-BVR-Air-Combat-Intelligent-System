# -*- coding: utf-8 -*-
"""
多任务智能体/common/contracts.py（统一数据契约层，接口规范 §2.2/§2.4/§2.5）
================================================================
本模块是跨项目数据契约的**唯一来源**（帧字典 / info / 结果记录 / 版本号 /
命名与路径工具）。契约字段名规则：`<概念>_<单位>`，帧键 = CSV 列名 = JSON 字段名
（单位不同的转换层除外，如 CSV 角度列 `*_deg` ↔ 帧 `*_rad`）。

包含：
- 版本常量 VERSIONS 与各数据格式版本（写入产物元数据）；
- 帧字典契约：FRAME_KEYS_V1 / validate_frame / migrate_frame_keys；
- info 契约：INFO_KEYS_STEP（reset 与 step 键集一致）/ validate_info；
- 技能级 info 契约：JSBSIM_INFO_KEYS；
- CSV 契约：CSV_VERSION / CSV_REQUIRED_COLUMNS；
- 单局结果记录：EpisodeResult（含 record_version）；
- 命名/路径工具：sanitize_tag / unique_dir / ensure_within。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .errors import ExecutionError, IO_PATH_DENIED, ValidationError

# ---------------------------------------------------------------------------
# 版本（写入产物；加字段=小版本，改语义/删字段=大版本）
# ---------------------------------------------------------------------------
FRAME_VERSION = "frame_v1"
CSV_VERSION = "csv_v1"
RESULT_RECORD_VERSION = "record_v1"
TEST_CASE_VERSION = "test_case_v1"
INFO_VERSION = "info_v1"
VERSIONS = {
    "frame": FRAME_VERSION,
    "csv": CSV_VERSION,
    "info": INFO_VERSION,
    "result_record": RESULT_RECORD_VERSION,
    "test_case": TEST_CASE_VERSION,
}

# ---------------------------------------------------------------------------
# CSV 录制契约（EpisodeRecorder；列名 = 帧键名，角度列 deg 为显式单位转换）
# ---------------------------------------------------------------------------
CSV_VERSION_COLUMN = "csv_version"
CSV_REQUIRED_COLUMNS = (
    "steps", "t",
    "own_e_m", "own_n_m", "own_h_m", "own_psi_deg", "own_v_mps",
    "enemy_e_m", "enemy_n_m", "enemy_psi_deg", "enemy_v_mps",
    "dist_m", "events_json", "own_missiles_json", "enemy_missiles_json",
)

# ---------------------------------------------------------------------------
# 枚举取值（保留字符串值以兼容落盘文件）
# ---------------------------------------------------------------------------
TERMINATION_REASONS = ("enemy_hit", "own_hit", "own_crash", "enemy_crash",
                       "fuel_out", "escape", "timeout")


class RadarState(str, Enum):
    TRACK = "TRACK"
    LOST = "LOST"


# ---------------------------------------------------------------------------
# 帧字典契约（BVRCombatEnv.get_viz_frame）
# ---------------------------------------------------------------------------
FRAME_KEYS_V1 = (
    "steps", "t", "own", "enemy", "own_missiles", "enemy_missiles", "dist_m",
    "radar_state", "enemy_radar_state", "rwr_alarm", "rwr_bearing_rad",
    "maws_alarm", "maws_tta_s", "in_zone",
    "r_max_own_m", "r_min_own_m", "r_nez_own_m", "r_max_enemy_m", "r_min_enemy_m",
    "radar_az_limit_deg", "reward", "terminated_reason", "events",
)
FRAME_REQUIRED_KEYS = ("steps", "t", "own", "enemy", "own_missiles",
                       "enemy_missiles", "dist_m", "events")
AIRFRAME_REQUIRED_KEYS = ("pos", "psi_rad", "vtrue_mps", "mach", "n_left")

# 旧键 → 规范键（v1 迁移用；读取历史内存帧/第三方帧时可用 migrate_frame_keys）
FRAME_KEY_ALIASES = {
    "rwr_bearing": "rwr_bearing_rad",
    "maws_tta": "maws_tta_s",
    "r_max_own": "r_max_own_m",
    "r_min_own": "r_min_own_m",
    "r_nez_own": "r_nez_own_m",
    "r_max_enemy": "r_max_enemy_m",
    "r_min_enemy": "r_min_enemy_m",
}


def migrate_frame_keys(frame: dict) -> dict:
    """把旧键名的帧就地补齐为规范键（已存在的规范键优先，不覆盖）。"""
    for old, new in FRAME_KEY_ALIASES.items():
        if new not in frame and old in frame:
            frame[new] = frame[old]
    return frame


def validate_frame(frame) -> list:
    """校验帧字典契约，返回错误列表（空 = 通过）。"""
    errors = []
    if not isinstance(frame, dict):
        return ["frame 应为 dict"]
    for key in FRAME_REQUIRED_KEYS:
        if key not in frame:
            errors.append(f"缺少必需帧键: {key}")
    for key in ("own", "enemy"):
        ac = frame.get(key)
        if not isinstance(ac, dict):
            errors.append(f"{key} 应为 dict")
            continue
        for sub in AIRFRAME_REQUIRED_KEYS:
            if sub not in ac:
                errors.append(f"{key}.{sub} 缺失")
        if isinstance(ac.get("pos"), (list, tuple)) and len(ac.get("pos", [])) != 3:
            errors.append(f"{key}.pos 应为长度 3")
    if "terminated_reason" in frame and frame["terminated_reason"] not in (None, *TERMINATION_REASONS):
        errors.append(f"terminated_reason 取值非法: {frame['terminated_reason']!r}")
    return errors


# ---------------------------------------------------------------------------
# info 契约（BVRCombatEnv：reset 与 step 键集一致）
# ---------------------------------------------------------------------------
INFO_KEYS_STEP = (
    "steps", "dist_m", "reward", "terminated_reason",
    "h_sl_m", "vtrue_mps", "mach", "fuel_lbs",
    "own_missiles_left", "enemy_missiles_left", "own_fired", "enemy_fired",
    "radar_state", "enemy_radar_state",
    "rwr_alarm", "rwr_bearing_rad", "maws_alarm", "maws_tta_s", "ecm_on",
    "in_zone", "r_max_own_m", "r_min_own_m", "r_nez_own_m",
    "r_max_enemy_m", "r_min_enemy_m",
)
JSBSIM_INFO_KEYS = ("steps", "h_sl_m", "vtrue_mps", "reward", "crash")


def validate_info(info, keys=INFO_KEYS_STEP) -> list:
    """校验 info 键集完备性（reset 与 step 必须一致）。"""
    missing = [k for k in keys if k not in (info or {})]
    return [f"缺少 info 键: {k}" for k in missing]


# ---------------------------------------------------------------------------
# 单局结果记录（run_case / run_tree_sim / 后续 2.7 聚合）
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    """一局对抗/评估的结构化结果（字段名与既有 summary.json 保持兼容 + 版本号）。"""

    seed: int = 0
    steps: int = 0
    terminated_reason: str | None = None
    logic_errors: int = 0
    safety_violations: int = 0
    own_fired: int = 0
    enemy_fired: int = 0
    in_zone_steps: int = 0
    frames: int = 0
    wall_s: float = 0.0
    tick_counts: dict = field(default_factory=dict)
    tree_id: str = ""
    tag: str = ""
    csv: str | None = None
    acmi: str | None = None
    record_version: str = RESULT_RECORD_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_versions"] = dict(VERSIONS)
        return d


# ---------------------------------------------------------------------------
# 命名与路径工具
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def sanitize_tag(tag, fallback: str = "run") -> str:
    """清洗文件名标签（仅保留字母/数字/._-），空则用 fallback。"""
    cleaned = _TAG_RE.sub("_", str(tag or "")).strip("._-")
    return cleaned or fallback


def unique_dir(root, stem: str) -> Path:
    """返回 root/stem；若已存在则追加 _1/_2…（避免同秒运行目录冲突）。"""
    root = Path(root)
    path = root / stem
    i = 1
    while path.exists():
        path = root / f"{stem}_{i}"
        i += 1
    return path


def ensure_within(path, root) -> Path:
    """确保输出路径位于 root 之内（接口规范 §2.6 路径安全）。"""
    p, r = Path(path).resolve(), Path(root).resolve()
    if r != p and r not in p.parents:
        raise ExecutionError(IO_PATH_DENIED, "输出路径越界", detail=f"{p} 不在 {r} 内")
    return p
