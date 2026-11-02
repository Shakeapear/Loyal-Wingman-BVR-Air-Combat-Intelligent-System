# -*- coding: utf-8 -*-
"""validation/preflight_server.py
================================
服务器/新机器启动前的预检：路径、依赖、JSBSim 数据、权限与 CPU 环境。

在仓库根目录或任意目录运行（脚本自行定位 ``多任务智能体/``）：

    python 多任务智能体/validation/preflight_server.py
    python 多任务智能体/validation/preflight_server.py --run-smoke
    python 多任务智能体/validation/preflight_server.py --json

返回码：0=全部通过；1=存在失败项（逐条打印修复建议）。
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIB = _HERE.parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from common.paths import (  # noqa: E402
    DOCS_ASSETS_DIR, JSB_ROOT, MODEL_DIR, PROJECT_DIR, RESULTS_DIR,
    ensure_writable_dir, jsbsim_root, visualization_dir,
)

REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("gymnasium", "gymnasium"),
    ("stable_baselines3", "stable-baselines3"),
    ("jsbsim", "jsbsim"),
    ("py_trees", "py_trees"),
    ("yaml", "PyYAML"),
    ("cloudpickle", "cloudpickle"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("PIL", "Pillow"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("tensorboard", "tensorboard"),
    ("pytest", "pytest"),
]

RUNTIME_MODULES = [
    "agents.ppo_agent",
    "agents.evaluate_strike",
    "agents.task_agents.mission_env",
    "common.bvr_combat_env",
    "common.jsbsim_bridge",
]


def _version(dist: str):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _check_import(module: str) -> dict:
    try:
        spec = importlib.util.find_spec(module)
        if spec is None:
            return {"ok": False, "error": f"找不到模块 {module}"}
        __import__(module)
        return {"ok": True}
    except Exception as exc:  # 依赖/动态库导入失败都要给出可读原因
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _check_writable(path: Path, label: str) -> dict:
    try:
        ensure_writable_dir(path, label=label)
        return {"ok": True, "path": str(path)}
    except Exception as exc:
        return {"ok": False, "path": str(path), "error": str(exc)}


def _check_jsbsim(path: Path, run_smoke: bool) -> dict:
    try:
        root = jsbsim_root(path, validate=True)
    except Exception as exc:
        return {"ok": False, "path": str(path), "error": str(exc)}
    result = {"ok": True, "path": str(root)}
    if run_smoke:
        try:
            from common.jsbsim_bridge import JSBSimGymBridge
            bridge = JSBSimGymBridge(root=root)
            bridge.reset(seed=0, h_ft=15000.0, mach=0.8, heading_deg=0.0,
                         autopilot=False, warmup_frames=2)
            bridge.step_manual(0.6, 0.0, 0.0, 0.0, 1)
            bridge.state_dict()
            result["smoke"] = True
        except Exception as exc:
            result["ok"] = False
            result["smoke"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _check_env() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "cpu_count": os.cpu_count(),
        "BVR_JSBSIM_ROOT": os.environ.get("BVR_JSBSIM_ROOT", ""),
        "BVR_MODEL_DIR": os.environ.get("BVR_MODEL_DIR", ""),
        "BVR_RESULTS_DIR": os.environ.get("BVR_RESULTS_DIR", ""),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "MPLBACKEND": os.environ.get("MPLBACKEND", ""),
    }


def run(json_out: bool, run_smoke: bool, jsbsim_root_arg: str | None) -> int:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(PROJECT_DIR),
        "lib_dir": str(_LIB),
        "environment": _check_env(),
        "checks": {},
        "warnings": [],
    }
    checks = report["checks"]

    checks["python"] = ({"ok": True, "version": sys.version.split()[0]}
                        if sys.version_info >= (3, 10) else
                        {"ok": False, "error": "需要 Python >= 3.10（推荐 3.12）"})

    deps = {}
    for module, dist in REQUIRED_MODULES:
        item = _check_import(module)
        item["version"] = _version(dist)
        deps[dist] = item
    checks["dependencies"] = deps

    runtime = {module: _check_import(module) for module in RUNTIME_MODULES}
    checks["runtime_imports"] = runtime

    root = Path(jsbsim_root_arg) if jsbsim_root_arg else JSB_ROOT
    checks["jsbsim"] = _check_jsbsim(root, run_smoke)
    checks["writable"] = {
        "models": _check_writable(MODEL_DIR, "模型输出目录"),
        "results": _check_writable(RESULTS_DIR, "结果输出目录"),
        "docs_assets": _check_writable(DOCS_ASSETS_DIR, "汇报图表目录"),
    }

    if not checks["writable"]["docs_assets"]["ok"]:
        report["warnings"].append(
            "docs/assets 不可写：不影响训练；行为树演示/汇报图请用 --out-dir "
            "或设置 BVR_DOCS_ASSETS_DIR。")
    if not visualization_dir().is_dir():
        report["warnings"].append(
            f"可视化工具目录不存在：{visualization_dir()}（训练不受影响；render 时才会用到）")
    if sys.platform.startswith("win") and run_smoke:
        report["warnings"].append(
            "Windows 下 SubprocVecEnv 使用 spawn；长训练请在 Linux 服务器运行，"
            "或显式加 --vec-env dummy。")
    if not sys.version_info >= (3, 10):
        report["warnings"].append("Python 版本过低，jsbsim/SB3 可能无法安装。")

    failed = []
    if not checks["python"]["ok"]:
        failed.append("python")
    for dist, item in deps.items():
        if not item["ok"]:
            failed.append(f"dependency:{dist}")
    for module, item in runtime.items():
        if not item["ok"]:
            failed.append(f"import:{module}")
    if not checks["jsbsim"]["ok"]:
        failed.append("jsbsim")
    for name, item in checks["writable"].items():
        if not item["ok"] and name != "docs_assets":
            failed.append(f"writable:{name}")
    report["ok"] = not failed
    report["failed"] = failed

    if json_out:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=" * 68)
        print("阶段三训练服务器预检")
        print("=" * 68)
        env = report["environment"]
        print(f"Python: {env['python']}  ({env['executable']})")
        print(f"平台: {env['platform']}")
        print(f"项目: {report['project_dir']}")
        print(f"编码: {env['filesystem_encoding']}  CPU: {env['cpu_count']}")
        print("-" * 68)
        print("[依赖]")
        for dist, item in deps.items():
            state = "OK" if item["ok"] else "FAIL"
            ver = item.get("version") or "?"
            extra = "" if item["ok"] else f"  {item.get('error')}"
            print(f"  {state:4s} {dist:20s} {ver}{extra}")
        print("[运行期导入]")
        for module, item in runtime.items():
            print(f"  {'OK' if item['ok'] else 'FAIL':4s} {module}"
                  + ("" if item["ok"] else f"  {item.get('error')}"))
        print("[路径/权限]")
        js = checks["jsbsim"]
        print(f"  {'OK' if js['ok'] else 'FAIL':4s} JSBSim {js.get('path')}"
              + ("" if js["ok"] else f"\n       {js.get('error')}"))
        for name, item in checks["writable"].items():
            print(f"  {'OK' if item['ok'] else 'FAIL':4s} {name}: {item['path']}"
                  + ("" if item["ok"] else f"\n       {item.get('error')}"))
        if report["warnings"]:
            print("[警告]")
            for warning in report["warnings"]:
                print(f"  - {warning}")
        print("-" * 68)
        print("结论: " + ("通过，可以启动训练。" if report["ok"]
                         else "未通过，请先修复上述 FAIL 项。"))
        if failed:
            print("失败项: " + ", ".join(failed))
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阶段三训练服务器预检")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--run-smoke", action="store_true",
                    help="额外执行一次 JSBSim 初始化+reset+step 冒烟")
    ap.add_argument("--jsbsim-root", default=None,
                    help="覆盖 BVR_JSBSIM_ROOT（通常不需要）")
    args = ap.parse_args(argv)
    return run(args.json, args.run_smoke, args.jsbsim_root)


if __name__ == "__main__":
    sys.exit(main())
