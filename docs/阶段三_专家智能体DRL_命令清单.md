# 阶段三专家智能体 DRL 底座：从零复现命令清单

> 适用环境：Windows + PowerShell 7，conda 环境 `DC`（Python 3.12，CPU 可跑）。
> 所有命令在仓库根目录执行；不执行完整训练（禁止 1e6/1e7 步），不 commit/push。

## 0. 进入仓库并确认环境

```powershell
$REPO = "C:\Users\36266\Desktop\大创 开发"
Set-Location $REPO
$PY = "C:\Users\36266\.conda\envs\DC\python.exe"

& $PY -c "import sys, torch, gymnasium, stable_baselines3, py_trees, jsbsim; print(sys.version); print('DC OK')"

# 确认工作区（本轮不自动提交）
git status --short
```

本轮未新增第三方依赖；如环境缺失，按 `多任务智能体/README.md` 安装 torch(CPU)、gymnasium、stable-baselines3、py_trees、jsbsim、numpy、matplotlib。

## 1. 接口/适配层/注册表/智能体测试

```powershell
& $PY -m pytest "多任务智能体\tests\test_expert_agents.py" -v
```

预期：24 passed（冷启动约 15 s，复跑约 11 s）。

## 2. 4 个任务级智能体：512 步小规模训练 + 保存加载

```powershell
& $PY -X utf8 "多任务智能体\validation\run_expert_validation.py" --timesteps 512 --eval-steps 64
```

预期：

- 4/4 通过；
- 训练损失/评估奖励有限；
- 保存/加载推理一致；
- 报告：`多任务智能体\validation\results\stage3\stage3_expert_validation.{json,txt}`；
- 模型：`多任务智能体\models\task_{strike,patrol,defense,ew}_ppo.zip`（不入库）。

可选更小规模冒烟（约 3 s）：

```powershell
& $PY -X utf8 "多任务智能体\validation\run_expert_validation.py" --tasks strike --timesteps 64 --eval-steps 8
```

## 3. 行为树最小演示（4 个任务级智能体）

```powershell
& $PY -X utf8 "多任务智能体\validation\run_expert_bt_demo.py" --tasks strike,patrol,defense,ew --steps 12
```

预期：4 个任务均 `SUCCESS`，每任务 12 步；产物在：

```text
docs\assets\stage3_expert_demo\
├── strike.csv / patrol.csv / defense.csv / ew.csv
├── bt_expert_demo.csv
├── demo_summary.json
├── demo_log.json
└── demo_screenshot.png
```

默认加载第 2 步产出的极小模型；若模型不存在则使用随机初始化策略，仅演示接口闭环：

```powershell
& $PY -X utf8 "多任务智能体\validation\run_expert_bt_demo.py" --tasks strike,patrol --steps 12 --no-load-models
```

## 4. 全部回归测试

```powershell
# 多任务智能体（环境/导弹/雷达/契约/新增专家）
& $PY -m pytest "多任务智能体\tests" -q

# 引导策略行为树
& $PY -m pytest "引导策略行为树\tests" -q

# 可视化工具
& $PY -m pytest "可视化工具\tests" -q
```

预期：80 + 19 + 23 = **122 passed**，0 failed。

## 4.1 稳定飞行基线（assisted，240 步）

```powershell
# 4 任务 × seeds 7/17/27 × safe/model；隔离敌方发射；额外跑正常战斗配置（仅记录）
& $PY -X utf8 "多任务智能体\validation\run_stability_baseline.py" --combat
```

预期：**24/24 通过，0 次 `own_crash`**，全部 240 步 `timeout`，约 11 s。产物：

```text
多任务智能体\validation\results\stage3_stability\
├── stability_summary.json
├── stability_report.txt
├── combat_summary.json
└── <task>_<policy>_seed<seed>.csv
```

raw_fcs 对照诊断（预期 45 步 `own_crash`，用于证明根因）：

```powershell
& $PY -X utf8 "多任务智能体\validation\run_stability_baseline.py" --tasks strike --seeds 42 --policies safe --control-mode raw_fcs --out-dir "多任务智能体\validation\results\stage3_diagnosis"
```

## 5. 技能级回归抽查（cruise_hold）

```powershell
# 先备份既有 5 任务基线报告
Copy-Item "多任务智能体\validation\results\validation_report.txt" "$env:TEMP\validation_report_backup.txt" -Force

# 只复跑 cruise_hold（L1 冒烟 + L2 建模 + L3 2048 步极小训练）
& $PY -X utf8 "多任务智能体\validation\run_validation.py" --tasks cruise_hold

# 恢复原 5 任务基线报告
Copy-Item "$env:TEMP\validation_report_backup.txt" "多任务智能体\validation\results\validation_report.txt" -Force
Get-Content "多任务智能体\validation\results\validation_report.txt"
```

预期：`cruise_hold: 通过`，2048 步约 9 s（实测 8.8 s）。

## 6. 查看产物与工作区改动

```powershell
# 验证报告
Get-Content "多任务智能体\validation\results\stage3\stage3_expert_validation.txt"

# 演示截图/汇总
Start-Process "docs\assets\stage3_expert_demo\demo_screenshot.png"
Get-Content "docs\assets\stage3_expert_demo\demo_summary.json" -TotalCount 30

# 工作区状态（不提交）
git status --short
git diff --stat
```

> 新增文件为 untracked 时 `git diff --stat` 不显示其行数；以 `git status --short` 的文件清单为准。
> 本轮按约束**未执行** `git add` / `git commit` / `git push`。

## 7. 一键顺序（复制到 PowerShell）

```powershell
$REPO = "C:\Users\36266\Desktop\大创 开发"
Set-Location $REPO
$PY = "C:\Users\36266\.conda\envs\DC\python.exe"

& $PY -m pytest "多任务智能体\tests\test_expert_agents.py" -q
& $PY -X utf8 "多任务智能体\validation\run_expert_validation.py" --timesteps 512 --eval-steps 64
& $PY -X utf8 "多任务智能体\validation\run_expert_bt_demo.py" --tasks strike,patrol,defense,ew --steps 12
& $PY -X utf8 "多任务智能体\validation\run_stability_baseline.py" --combat
& $PY -m pytest "多任务智能体\tests" -q
& $PY -m pytest "引导策略行为树\tests" -q
& $PY -m pytest "可视化工具\tests" -q
git status --short
```

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: agents` | 确认在仓库根目录执行；脚本已自行把 `多任务智能体/` 加入 `sys.path` |
| 模型不存在 | 先执行第 2 步；或演示加 `--no-load-models` |
| 训练/评估中 `own_crash` | 任务级默认 `control_mode=assisted`，240 步基线已 24/24 通过；若显式 `raw_fcs` 则无配平会坠地，属对照预期 |
| 稳定飞行脚本全部 `FAIL` | 确认未误用 `--control-mode raw_fcs`；assisted 产物见 `validation/results/stage3_stability/` |
| `Box low's precision lowered` 警告 | 既有技能级 `JSBSimFlightEnv` 边界 dtype 提示，不影响测试 |
| 需要 GPU | 本底座默认 CPU；不依赖 GPU，显存/驱动不满足也可复现 |
