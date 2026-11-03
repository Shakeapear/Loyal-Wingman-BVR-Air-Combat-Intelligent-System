# -*- coding: utf-8 -*-
"""人类招式库（ManeuverSpec / ManeuverRegistry）。

招式库是“人类知识”与行为树/JSRL 之间的数据层：

- 行为树只通过 `execute_maneuver(name, params)` 选择招式（Phase 2 实现）；
- 每个招式必须有前置条件、参数、动作序列/控制器、退出条件、安全边界、
  数据来源；缺少这些的条目不进入正式库；
- 不在本模块执行飞行动作，也不修改 ActionV1 维度。

默认目录：
    飞行数据集资料库/05_人类战术动作与招式库/04_标注与招式表/maneuver_catalog_v1.csv
可用 BVR_MANEUVER_CATALOG 指向其他 CSV。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = (
    PROJECT_ROOT / "飞行数据集资料库" / "05_人类战术动作与招式库"
    / "04_标注与招式表" / "maneuver_catalog_v1.csv"
)
REQUIRED_FIELDS = (
    "maneuver_id", "name_zh", "name_en", "phase", "preconditions",
    "parameters", "action_sequence_or_controller", "exit_conditions",
    "safety_constraints", "source_type", "source_ref", "status",
)


def catalog_path(path: str | Path | None = None) -> Path:
    if path is not None:
        target = Path(path)
    else:
        env = os.environ.get("BVR_MANEUVER_CATALOG")
        target = Path(env) if env else DEFAULT_CATALOG
    return target.expanduser().resolve()


@dataclass(frozen=True)
class ManeuverSpec:
    maneuver_id: str
    name_zh: str
    name_en: str
    phase: str
    preconditions: str
    parameters: str
    action_sequence_or_controller: str
    exit_conditions: str
    safety_constraints: str
    source_type: str
    source_ref: str
    status: str
    notes: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def validate(self):
        missing = [name for name in REQUIRED_FIELDS
                   if not str(getattr(self, name, "")).strip()]
        if missing:
            raise ValueError(
                f"招式 {self.maneuver_id or '?'} 缺少必填字段：{', '.join(missing)}")
        return self

    def to_dict(self) -> dict:
        data = {name: getattr(self, name) for name in REQUIRED_FIELDS}
        data["notes"] = self.notes
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_row(cls, row: dict) -> "ManeuverSpec":
        known = set(REQUIRED_FIELDS) | {"notes"}
        extra = {key: value for key, value in row.items()
                 if key not in known and value not in (None, "")}
        spec = cls(
            maneuver_id=str(row.get("maneuver_id", "")).strip(),
            name_zh=str(row.get("name_zh", "")).strip(),
            name_en=str(row.get("name_en", "")).strip(),
            phase=str(row.get("phase", "")).strip(),
            preconditions=str(row.get("preconditions", "")).strip(),
            parameters=str(row.get("parameters", "")).strip(),
            action_sequence_or_controller=str(
                row.get("action_sequence_or_controller", "")).strip(),
            exit_conditions=str(row.get("exit_conditions", "")).strip(),
            safety_constraints=str(row.get("safety_constraints", "")).strip(),
            source_type=str(row.get("source_type", "")).strip(),
            source_ref=str(row.get("source_ref", "")).strip(),
            status=str(row.get("status", "")).strip(),
            notes=str(row.get("notes", "")).strip(),
            extra=extra,
        )
        return spec.validate()


class ManeuverRegistry:
    """只读招式注册表；ID 必须唯一。"""

    def __init__(self, specs):
        self._specs = {}
        for spec in specs:
            if not isinstance(spec, ManeuverSpec):
                raise TypeError("specs 必须为 ManeuverSpec")
            if spec.maneuver_id in self._specs:
                raise ValueError(f"重复招式 ID：{spec.maneuver_id}")
            self._specs[spec.maneuver_id] = spec

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> "ManeuverRegistry":
        target = catalog_path(path)
        if not target.is_file():
            raise FileNotFoundError(
                f"招式库 CSV 不存在：{target}\n"
                "可设置 BVR_MANEUVER_CATALOG 指向 maneuver_catalog_v1.csv。")
        with target.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return cls(ManeuverSpec.from_row(row) for row in rows)

    def get(self, maneuver_id: str) -> ManeuverSpec:
        try:
            return self._specs[maneuver_id]
        except KeyError as exc:
            raise KeyError(f"未注册招式：{maneuver_id}") from exc

    def has(self, maneuver_id: str) -> bool:
        return maneuver_id in self._specs

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def by_phase(self, phase: str) -> tuple[ManeuverSpec, ...]:
        return tuple(spec for spec in self._specs.values()
                     if spec.phase == phase)

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    def __contains__(self, maneuver_id: str) -> bool:
        return maneuver_id in self._specs


def load_default_registry() -> ManeuverRegistry:
    return ManeuverRegistry.from_csv()


__all__ = ["DEFAULT_CATALOG", "REQUIRED_FIELDS", "ManeuverSpec",
           "ManeuverRegistry", "catalog_path", "load_default_registry"]
