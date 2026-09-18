# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt_generator.py（步骤 2.5 交付物 D2.5-1）
================================================================
LLM 行为树生成器-测试器框架（Coder-Executor-Tester 迭代闭环）：

    Round N: Coder(LLM: Prompt→JSON) → Executor(JSON→py_trees→JSBSim 执行)
                  ▲                                    │
                  └──────── 反馈（缺失节点/失败用例） ◀── Tester(用例生成+评估)

收敛判据（计划书步骤 2.5）：用例通过率 > 90%、OODA 关键节点覆盖 > 95%、
安全违规步数 = 0。

用法（DC 环境，在 引导策略行为树/ 目录下）：
    # 离线确定性验证（无需 API Key，含 生成→测试→修正 完整两轮日志）
    python llm_bt_generator.py --provider mock --rounds 3

    # 真实 LLM（DeepSeek，需 DEEPSEEK_API_KEY；OpenAI 同理）
    python llm_bt_generator.py --provider deepseek --rounds 3

    # 自定义任务描述 / 快速模式（不解算攻击区，用于冒烟）
    python llm_bt_generator.py --mission "巡逻并拦截入侵目标" --fast

产物：outputs/<运行时间戳>/ 下每轮 bt_vN.json/.xml、tree_vN.txt/.dot、
test_cases_vN.json、test_report_vN.json、llm 原始输出，
以及汇总 iteration_log.md（D2.5-3 迭代收敛日志样例）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))

from llm_bt.llm_client import LLMClient, LLMConfigError, extract_json
from llm_bt.prompts import SYSTEM_CODER, render_coder_prompt
from llm_bt.schema import CONVERGENCE, ooda_coverage
from llm_bt.tester import generate_test_cases, run_test_cases
from llm_bt.tree_io import text_tree, tree_to_dot, tree_to_xml
from llm_bt.validator_bridge import validate_tree_static

OUTPUTS_DIR = _HERE / "outputs"
DEFAULT_MISSION = (
    "超视距空战拦截：雷达搜索并锁定目标，进入攻击区后发射中距弹（双发间隔≥4 s），"
    "导弹来袭时优先规避，燃油不足时返航，全程满足安全包线约束")


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _feedback_text(round_idx: int, static_errors, static_warnings,
                   coverage: float, missing, summary) -> str:
    """构造下一轮 Coder 的反馈文本（静态错误 / 缺失节点 / 失败用例）。"""
    lines = [f"第 {round_idx} 轮测试反馈："]
    if static_errors:
        lines.append("- 静态校验错误：" + "；".join(static_errors))
    if static_warnings:
        lines.append("- 静态校验警告：" + "；".join(static_warnings))
    if missing:
        lines.append(f"- OODA 关键节点缺失（覆盖率 {coverage:.0%}）：" + "、".join(missing))
    if summary:
        lines.append(f"- 用例通过率 {summary['pass_rate']:.0%}"
                     f"（{summary['n_pass']}/{summary['n_cases']}），"
                     f"安全违规 {summary['safety_violations']} 步")
        for c in summary["cases"]:
            if not c["passed"]:
                failed = [f"{k['criterion']['metric']}{k['criterion']['op']}"
                          f"{k['criterion']['value']}（实际 {k['actual']}）"
                          for k in c["checks"] if not k["passed"]]
                lines.append(f"  · 未通过 {c['case']['case_id']}"
                             f"（seed={c['case']['seed']}）：" + "；".join(failed))
    return "\n".join(lines)


def run_iteration(mission: str = DEFAULT_MISSION, provider: str = "auto",
                  model: str | None = None, rounds: int = 3, max_steps: int = 240,
                  compute_envelope: bool = True, out_dir=None, verbose: bool = True):
    """执行 Coder-Executor-Tester 迭代，返回 {run_dir, history, final_tree, ...}。"""
    llm = LLMClient(provider=provider, model=model)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir or OUTPUTS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_md = [
        "# 迭代收敛日志（D2.5-3）", "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- provider：{llm.provider}（model={llm.model or '离线 mock'}）",
        f"- 任务：{mission}",
        f"- 收敛判据：通过率≥{CONVERGENCE['pass_rate']:.0%}，"
        f"OODA 覆盖≥{CONVERGENCE['ooda_coverage']:.0%}，"
        f"安全违规≤{CONVERGENCE['safety_violations']}", ""]
    feedback, missing = "", []
    final_tree, history = None, []
    for r in range(1, int(rounds) + 1):
        t0 = time.time()
        raw_coder = llm.complete(
            SYSTEM_CODER, render_coder_prompt(mission, feedback), task="coder",
            context={"mission": mission, "feedback": feedback,
                     "missing": missing, "round": r})
        _write(run_dir / f"llm_coder_raw_v{r}.txt", raw_coder)
        tree, static_errors, static_warnings = None, [], []
        try:
            tree = extract_json(raw_coder)
            tree["bt_id"] = f"bt_v{r}"
            tree.setdefault("mission", mission)
            static_errors, static_warnings = validate_tree_static(tree)
        except Exception as e:
            static_errors = [f"JSON 解析失败: {type(e).__name__}: {e}"]

        if static_errors:
            history.append({"round": r, "status": "static_failed",
                            "errors": static_errors})
            log_md += [f"## Round {r} — 静态校验未通过", "",
                       *[f"- {e}" for e in static_errors], ""]
            feedback = _feedback_text(r, static_errors, static_warnings, 0.0,
                                      missing, None)
            if verbose:
                print(f"[Round {r}] 静态校验未通过：{static_errors}", flush=True)
            continue

        coverage, missing_now = ooda_coverage(tree)
        cases, raw_tester = generate_test_cases(llm, tree)
        _write(run_dir / f"llm_tester_raw_v{r}.txt", raw_tester)
        summary = run_test_cases(tree, cases, max_steps=max_steps,
                                 compute_envelope=compute_envelope)
        converged = (summary["pass_rate"] >= CONVERGENCE["pass_rate"]
                     and coverage >= CONVERGENCE["ooda_coverage"]
                     and summary["safety_violations"] <= CONVERGENCE["safety_violations"])

        _write(run_dir / f"bt_v{r}.json", json.dumps(tree, ensure_ascii=False, indent=2))
        _write(run_dir / f"bt_v{r}.xml", tree_to_xml(tree))
        _write(run_dir / f"tree_v{r}.txt", text_tree(tree))
        _write(run_dir / f"tree_v{r}.dot", tree_to_dot(tree, f"bt_v{r}"))
        _write(run_dir / f"test_cases_v{r}.json",
               json.dumps({"test_cases": cases}, ensure_ascii=False, indent=2))
        report = {k: summary[k] for k in (
            "pass_rate", "n_pass", "n_cases", "safety_violations",
            "ooda_coverage", "ooda_missing",
            "ooda_executed_coverage", "ooda_executed_missing")}
        report["cases"] = [{"case_id": c["case"]["case_id"],
                            "seed": c["case"]["seed"],
                            "passed": c["passed"],
                            "checks": c["checks"],
                            "result": c["result"]} for c in summary["cases"]]
        report["converged"] = converged
        report["elapsed_s"] = round(time.time() - t0, 1)
        _write(run_dir / f"test_report_v{r}.json",
               json.dumps(report, ensure_ascii=False, indent=2))

        history.append({"round": r, "status": "converged" if converged else "not_converged",
                        "coverage": coverage, "missing": missing_now, "summary": report})
        log_md += [
            f"## Round {r}", "",
            f"- 静态校验：通过（警告 {len(static_warnings)} 条）",
            f"- OODA 覆盖率：{coverage:.0%}"
            f"（缺失：{'、'.join(missing_now) or '无'}）",
            f"- 用例通过率：{summary['pass_rate']:.0%}"
            f"（{summary['n_pass']}/{summary['n_cases']}）",
            f"- 安全违规：{summary['safety_violations']} 步",
            f"- 是否收敛：{'是' if converged else '否'}", "",
            "```", text_tree(tree), "```", ""]
        if verbose:
            print(f"[Round {r}] 覆盖 {coverage:.0%}，通过率 {summary['pass_rate']:.0%}，"
                  f"安全违规 {summary['safety_violations']}，"
                  f"{'收敛' if converged else '未收敛（进入下一轮修正）'}", flush=True)
        final_tree = tree
        if converged:
            break
        feedback = _feedback_text(r, static_errors, static_warnings,
                                  coverage, missing_now, summary)
        # 反馈累积：已缺失过的节点不再遗忘（mock 与真实 LLM 共用）
        missing = sorted(set(missing) | set(missing_now))

    if final_tree is not None:
        _write(run_dir / "bt_final.json", json.dumps(final_tree, ensure_ascii=False, indent=2))
        _write(run_dir / "bt_final.xml", tree_to_xml(final_tree))
    _write(run_dir / "iteration_log.md", "\n".join(log_md))
    return {"run_dir": run_dir, "history": history, "final_tree": final_tree,
            "provider": llm.provider, "mock": llm.is_mock}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="LLM 行为树生成器-测试器框架（步骤 2.5 交付物 D2.5-1）")
    p.add_argument("--mission", default=DEFAULT_MISSION, help="任务描述（Coder Prompt 注入）")
    p.add_argument("--provider", default="auto",
                   choices=["auto", "deepseek", "openai", "mock"],
                   help="LLM 提供方；auto=有 DEEPSEEK_API_KEY 用 deepseek，否则 mock")
    p.add_argument("--model", default=None, help="模型名（默认 provider 推荐模型）")
    p.add_argument("--rounds", type=int, default=3, help="最大迭代轮数")
    p.add_argument("--max-steps", type=int, default=240, help="单用例最大决策步（1 Hz）")
    p.add_argument("--fast", action="store_true",
                   help="不复算攻击区（compute_envelope=False），用于冒烟/快速验证")
    p.add_argument("--out", default=None, help="输出根目录（默认 outputs/）")
    p.add_argument("--quiet", action="store_true", help="不打印每轮进度")
    args = p.parse_args(argv)

    try:
        result = run_iteration(
            mission=args.mission, provider=args.provider, model=args.model,
            rounds=args.rounds, max_steps=args.max_steps,
            compute_envelope=not args.fast, out_dir=args.out,
            verbose=not args.quiet)
    except LLMConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2
    hist = result["history"]
    status = hist[-1]["status"] if hist else "无轮次"
    if result["final_tree"] is None:
        print(f"[完成] 未产出可用行为树（{status}），详见 {result['run_dir']}")
        return 1
    print(f"[完成] {'收敛' if status == 'converged' else '未收敛（已达最大轮数）'}"
          f"；产物目录：{result['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
