# -*- coding: utf-8 -*-
"""
validation/run_expert_validation.py（阶段三专家智能体 DRL 底座验证）
==================================================================
对 4 个任务级专家智能体（strike/patrol/defense/ew）逐项执行：

1. 实例化 + ActionSpec 检查；
2. get_action/predict 形状/dtype/范围（ActionV1）；
3. 小规模 PPO 训练（默认 512 步，非完整训练）；
4. 训练指标有限性 + 推理动作统计；
5. 保存/加载后同观测推理一致性；
6. 输出 JSON/TXT 报告（固定 seed，记录耗时/步数/动作统计）。

用法（任意目录；脚本自行定位 多任务智能体/）：
    python 多任务智能体/validation/run_expert_validation.py
    ... --tasks strike,patrol --timesteps 512
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = _HERE.parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from agents.policy_interface import ACTION_V1_HIGH, ACTION_V1_LOW  # noqa: E402
from agents.task_agents import TASK_AGENTS  # noqa: E402
from common.paths import MODEL_DIR, ensure_writable_dir, results_dir  # noqa: E402

RESULTS_DIR = results_dir() / "stage3"


def _check_action(action, task_id):
    arr = np.asarray(action)
    checks = {
        "shape_ok": tuple(arr.shape) == (7,),
        "dtype_ok": str(arr.dtype) == "float32",
        "finite": bool(np.all(np.isfinite(arr))),
        "range_ok": bool(np.all(arr >= ACTION_V1_LOW - 1e-6)
                         and np.all(arr <= ACTION_V1_HIGH + 1e-6)),
    }
    checks["ok"] = all(checks.values())
    return checks


def validate_one(task_id: str, timesteps: int, eval_steps: int, seed: int,
                 env_config: dict, model_dir: Path) -> dict:
    cls = TASK_AGENTS[task_id]
    t0 = time.time()
    agent = cls(env_config=dict(env_config),
                train_config={"n_steps": 256, "batch_size": 64, "n_epochs": 4},
                seed=seed, model_path=model_dir / f"task_{task_id}_ppo")
    instantiate_s = time.time() - t0
    spec = agent.action_spec()

    t_build = time.time()
    agent.build_model()
    build_s = time.time() - t_build
    obs, info = agent.build_env().reset(seed=seed)
    out = agent.get_action(obs, info=info, deterministic=True)
    pred = agent.predict(obs, deterministic=True)
    action_check = _check_action(out.action, task_id)
    pred_check = _check_action(pred, task_id)

    train_info = {"timesteps": 0, "wall_s": 0.0, "metrics": {}, "all_finite": None}
    if timesteps > 0:
        train_info = agent.train(timesteps=timesteps)

    eval_info = agent.evaluate(n_steps=eval_steps)

    # 保存 → 重新实例化 → 加载 → 同观测推理一致性
    obs_check = obs.copy()
    out_before = agent.get_action(obs_check, info=info, deterministic=True).action.copy()
    save_path = agent.save()
    agent2 = cls(env_config=dict(env_config),
                 train_config={"n_steps": 256, "batch_size": 64, "n_epochs": 4},
                 seed=seed, model_path=model_dir / f"task_{task_id}_ppo")
    agent2.load(save_path)
    out_after = agent2.get_action(obs_check, info=info, deterministic=True)
    load_consistent = bool(np.allclose(out_before, out_after.action, atol=1e-6))
    agent.close()
    agent2.close()

    return {
        "task_id": task_id,
        "mission_name_zh": spec.get("mission_name_zh"),
        "action_version": spec.get("version"),
        "action_check": action_check,
        "predict_check": pred_check,
        "instantiate_s": round(instantiate_s, 3),
        "build_s": round(build_s, 3),
        "train": {**train_info, "metrics": _compact_metrics(train_info.get("metrics", {}))},
        "eval": eval_info,
        "save_path": str(save_path),
        "load_consistent": load_consistent,
        "load_action_diff": float(np.max(np.abs(out_before - out_after.action))),
        "ok": bool(action_check["ok"] and pred_check["ok"]
                   and (timesteps == 0 or train_info.get("all_finite", False))
                   and load_consistent and eval_info.get("reward_finite", False)),
    }


def _compact_metrics(metrics: dict) -> dict:
    keep = [k for k in metrics if any(s in k for s in
                                      ("loss", "reward", "entropy", "approx_kl", "explained"))]
    return {k: round(float(metrics[k]), 8) for k in keep}


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三任务级专家智能体验证（小规模训练）")
    ap.add_argument("--tasks", default="strike,patrol,defense,ew")
    ap.add_argument("--timesteps", type=int, default=512, help="每个智能体训练步数")
    ap.add_argument("--eval-steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-env-steps", type=int, default=64,
                    help="验证环境单局最大步数（短局加速，非最终场景）")
    ap.add_argument("--envelope", action="store_true",
                    help="启用攻击区解算（默认关闭以加速验证）")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_AGENTS]
    if unknown:
        print(f"[错误] 未知任务: {unknown}；可选 {list(TASK_AGENTS)}", file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir)
    out_dir = ensure_writable_dir(args.out_dir, label="专家验证输出目录")
    model_dir = ensure_writable_dir(model_dir, label="模型输出目录")
    env_config = {"max_steps": int(args.max_env_steps),
                  "compute_envelope": bool(args.envelope)}

    print(f"[阶段三验证] Python={platform.python_version()} "
          f"tasks={tasks} timesteps={args.timesteps} eval={args.eval_steps} "
          f"env={env_config}")
    results = []
    t_all = time.time()
    for task_id in tasks:
        print(f"\n== {task_id} ==")
        try:
            row = validate_one(task_id, args.timesteps, args.eval_steps, args.seed,
                               env_config, model_dir)
        except Exception as exc:  # 报告失败但继续其他智能体
            import traceback
            traceback.print_exc()
            row = {"task_id": task_id, "ok": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)
        if row.get("ok"):
            tr = row.get("train", {})
            ev = row.get("eval", {})
            print(f"  [通过] action={row['action_check']} "
                  f"train={tr.get('timesteps')}步/{tr.get('wall_s')}s "
                  f"finite={tr.get('all_finite')} eval_reward={ev.get('reward_mean')} "
                  f"save/load={row.get('load_consistent')}")
        else:
            print(f"  [失败] {row.get('error', row.get('action_check'))}")

    total_s = time.time() - t_all
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "timesteps_per_agent": args.timesteps,
        "eval_steps": args.eval_steps,
        "env_config": env_config,
        "total_wall_s": round(total_s, 3),
        "passed": sum(1 for r in results if r.get("ok")),
        "total": len(results),
        "results": results,
    }
    json_path = out_dir / "stage3_expert_validation.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    lines = [
        "阶段三任务级专家智能体 DRL 底座验证报告（小规模训练，非完整训练）",
        f"生成时间: {summary['generated_at']}",
        f"Python: {summary['python']} | seed={args.seed} | "
        f"timesteps/agent={args.timesteps} | eval_steps={args.eval_steps}",
        f"环境: {env_config}",
        "",
        "task_id | 结果 | 训练步数 | 耗时s | 损失有限 | 奖励有限 | 动作形状 | dtype | 范围 | 保存加载一致",
        "---|---|---|---|---|---|---|---|---|---",
    ]
    for r in results:
        tr = r.get("train", {})
        ev = r.get("eval", {})
        ac = r.get("action_check", {})
        lines.append(
            f"{r['task_id']} | {'通过' if r.get('ok') else '失败'} | "
            f"{tr.get('timesteps', 0)} | {tr.get('wall_s', 0)} | "
            f"{tr.get('all_finite')} | {ev.get('reward_finite')} | "
            f"{ev.get('action_shape')} | {ev.get('action_dtype')} | "
            f"{ac.get('range_ok')} | {r.get('load_consistent')}")
    lines += ["", f"汇总: {summary['passed']}/{summary['total']} 通过，"
                  f"总耗时 {summary['total_wall_s']}s", f"JSON: {json_path}"]
    txt_path = out_dir / "stage3_expert_validation.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
