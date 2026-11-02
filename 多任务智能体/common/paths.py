# -*- coding: utf-8 -*-
"""common/paths.py
=================
项目路径的唯一解析入口。

目标：训练/评估脚本不再各自拼 ``Path(__file__).parents[...]``，也不再依赖
当前工作目录（CWD）。所有路径均可通过环境变量覆盖，便于在 Linux 服务器、
只读代码目录或非 ASCII 路径下运行：

- ``BVR_JSBSIM_ROOT``：JSBSim 数据根目录（需包含 ``aircraft/f104``）
- ``BVR_MODEL_DIR``：模型输出目录
- ``BVR_RESULTS_DIR``：验证/评估结果目录
- ``BVR_DOCS_ASSETS_DIR``：汇报图表输出目录
- ``BVR_VIZ_DIR``：可视化工具目录

相对路径一律按当前工作目录解析；建议在服务器上使用绝对路径。
"""
from __future__ import annotations

import os
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1]          # 多任务智能体/
PROJECT_DIR = LIB_DIR.parent                            # 仓库根目录

# 兼容旧代码/文档中的命名
PROJECT_ROOT = PROJECT_DIR

_JSBSIM_REQUIRED = (
    Path("aircraft") / "f104" / "f104.xml",
    Path("aircraft") / "f104" / "reset_cruise.xml",
    Path("aircraft") / "f104" / "Systems" / "autopilot-hold.xml",
)


def _resolve(raw: str | os.PathLike[str], default: Path) -> Path:
    """把环境变量值解析为绝对路径；空值回退到默认路径。"""
    text = str(raw).strip() if raw is not None else ""
    path = Path(text).expanduser() if text else default
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def env_path(name: str, default: Path) -> Path:
    """读取路径覆盖环境变量 ``name``，否则返回 ``default`` 的绝对路径。"""
    return _resolve(os.environ.get(name, ""), default)


def jsbsim_root(root: str | os.PathLike[str] | None = None,
                *, validate: bool = False) -> Path:
    """返回 JSBSim 根目录；``root``/``BVR_JSBSIM_ROOT``/默认 ``<repo>/JSBSim``。

    ``validate=True`` 时检查 F-104 模型文件，缺失时抛出带修复提示的
    ``RuntimeError``，避免在服务器上落到难懂的 JSBSim C++ 错误。
    """
    path = _resolve(root, env_path("BVR_JSBSIM_ROOT", PROJECT_DIR / "JSBSim"))
    if validate:
        missing = [str(rel) for rel in _JSBSIM_REQUIRED if not (path / rel).is_file()]
        if missing:
            raise RuntimeError(
                "JSBSim 数据目录不完整或未随代码部署："
                f"{path}\n缺失文件：{', '.join(missing)}\n"
                "修复：1) 把仓库 JSBSim/ 一并上传；或 2) 设置 "
                "BVR_JSBSIM_ROOT 指向包含 aircraft/f104 的目录。"
            )
    return path


def model_dir() -> Path:
    return env_path("BVR_MODEL_DIR", LIB_DIR / "models")


def results_dir() -> Path:
    return env_path("BVR_RESULTS_DIR", LIB_DIR / "validation" / "results")


def docs_assets_dir() -> Path:
    return env_path("BVR_DOCS_ASSETS_DIR", PROJECT_DIR / "docs" / "assets")


def visualization_dir() -> Path:
    return env_path("BVR_VIZ_DIR", PROJECT_DIR / "可视化工具")


def ensure_writable_dir(path: str | os.PathLike[str], *, label: str = "输出目录") -> Path:
    """创建并验证目录可写；失败时给出明确路径与权限建议。"""
    target = _resolve(path, Path.cwd())
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"{label}无法创建：{target}\n"
            "请检查目录权限，或用环境变量（如 BVR_RESULTS_DIR / BVR_MODEL_DIR）"
            "指向可写目录。"
        ) from exc
    probe = target / f".bvr_write_test_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"{label}不可写：{target}\n"
            "请检查目录权限，或改用可写目录/环境变量。"
        ) from exc
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
    return target


# 兼容常量（导入时解析；需要随环境变量动态变化时请调用上面的函数）
MODEL_DIR = model_dir()
RESULTS_DIR = results_dir()
DOCS_ASSETS_DIR = docs_assets_dir()
VISUALIZATION_DIR = visualization_dir()
JSB_ROOT = jsbsim_root()

__all__ = [
    "LIB_DIR", "PROJECT_DIR", "PROJECT_ROOT",
    "JSB_ROOT", "MODEL_DIR", "RESULTS_DIR", "DOCS_ASSETS_DIR",
    "VISUALIZATION_DIR", "env_path", "jsbsim_root", "model_dir",
    "results_dir", "docs_assets_dir", "visualization_dir",
    "ensure_writable_dir",
]
