# 阶段三专家智能体 DRL 底座测试报告

> 日期：2026-10-30｜环境：Windows + PowerShell 7｜conda `DC`（Python 3.12.13）｜CPU 训练
>
> 结论：**4/4 任务级智能体通过接口/小规模训练/保存加载验证；assisted 240 步隔离飞行 24/24 通过（无 `own_crash`）；新增 24 项测试通过；三套既有测试全绿（合计 122 项）；行为树成功调度 4 个任务级智能体完成最小闭环。**

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows（win32） |
| Shell | PowerShell 7（`pwsh`） |
| Python | 3.12.13（`C:\Users\36266\.conda\envs\DC\python.exe`） |
| 关键依赖 | torch CPU、gymnasium、stable-baselines3、py_trees、jsbsim 1.3.1、numpy、matplotlib |
| 固定随机种子 | 42（接口/训练验证）；7/17/27（240 步稳定飞行）；演示 7~10 |
| 训练规模 | 每任务 512 步（非完整训练） |
| 环境近似 | `compute_envelope=False`；验证 `max_steps=64`、稳定飞行 `max_steps=240`；任务级默认 `control_mode=assisted` |

---

## 2. 执行的命令

```powershell
# 1) 新增测试
& "C:\Users\36266\.conda\envs\DC\python.exe" -m pytest "多任务智能体\tests\test_expert_agents.py" -q

# 2) 4 个任务级智能体：实例化/动作/512 步训练/评估/保存加载
& "C:\Users\36266\.conda\envs\DC\python.exe" -X utf8 "多任务智能体\validation\run_expert_validation.py" --timesteps 512 --eval-steps 64

# 3) 行为树选择 4 个任务级智能体（每任务 12 步）
& "C:\Users\36266\.conda\envs\DC\python.exe" -X utf8 "多任务智能体\validation\run_expert_bt_demo.py" --tasks strike,patrol,defense,ew --steps 12

# 4) 回归：多任务智能体全部测试（含新增）
& "C:\Users\36266\.conda\envs\DC\python.exe" -m pytest "多任务智能体\tests" -q

# 5) 回归：行为树测试
& "C:\Users\36266\.conda\envs\DC\python.exe" -m pytest "引导策略行为树\tests" -q

# 6) 回归：可视化工具测试
& "C:\Users\36266\.conda\envs\DC\python.exe" -m pytest "可视化工具\tests" -q

# 7) 技能级抽查复跑（cruise_hold；先备份 validation_report.txt，跑完恢复）
Copy-Item "多任务智能体\validation\results\validation_report.txt" "$env:TEMP\validation_report_backup.txt" -Force
& "C:\Users\36266\.conda\envs\DC\python.exe" -X utf8 "多任务智能体\validation\run_validation.py" --tasks cruise_hold
Copy-Item "$env:TEMP\validation_report_backup.txt" "多任务智能体\validation\results\validation_report.txt" -Force

# 8) assisted 240 步稳定飞行基线（4 任务 × 3 seeds × safe/model）
& "C:\Users\36266\.conda\envs\DC\python.exe" -X utf8 "多任务智能体\validation\run_stability_baseline.py" --combat

# 9) raw_fcs 对照诊断（预期 own_crash，证明根因）
& "C:\Users\36266\.conda\envs\DC\python.exe" -X utf8 "多任务智能体\validation\run_stability_baseline.py" --tasks strike --seeds 42 --policies safe --control-mode raw_fcs --out-dir "多任务智能体\validation\results\stage3_diagnosis"
```

---

## 3. 任务级智能体验证结果

数据文件：`多任务智能体/validation/results/stage3/stage3_expert_validation.json`、`stage3_expert_validation.txt`。

| 智能体 | 实例化/动作 | `get_action` 形状/dtype/范围 | 小规模训练 | 训练耗时 | 损失有限 | 评估步数 | 评估奖励均值 | 保存/加载一致（最大差） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `strike` | 通过 | (7,) float32 / 界内 | 512 步 | 1.288 s | 是 | 64 | +0.6318 | 是（0.0） |
| `patrol` | 通过 | (7,) float32 / 界内 | 512 步 | 1.364 s | 是 | 64 | -0.0225 | 是（0.0） |
| `defense` | 通过 | (7,) float32 / 界内 | 512 步 | 1.464 s | 是 | 64 | +0.5018 | 是（0.0） |
| `ew` | 通过 | (7,) float32 / 界内 | 512 步 | 1.498 s | 是 | 64 | +0.0228 | 是（0.0） |

汇总：**4/4 通过，脚本总耗时 8.861 s**（不含 pytest 与既有技能回归）。本轮验证环境默认 `control_mode=assisted`，4 个智能体评估均为 64 步 `timeout`，**无 `own_crash`**。

### 3.1 训练损失/奖励（关键值）

| 智能体 | value_loss | policy_gradient_loss | entropy_loss | approx_kl | 评估奖励有限 |
|---|---:|---:|---:|---:|---:|
| `strike` | 5463.45 | -0.00788 | -9.9261 | 0.001579 | 是 |
| `patrol` | 917.81 | -0.01573 | -9.9309 | 0.008422 | 是 |
| `defense` | 1947.66 | -0.01113 | -9.9368 | 0.005743 | 是 |
| `ew` | 4749.79 | -0.00627 | -9.9263 | 0.001871 | 是 |

说明：

- 损失/奖励全部为有限值；`value_loss` 量级与 BVR 奖励尺度（命中 ±200、攻击区 +5/步）一致，512 步远未收敛，不作为性能指标；
- 评估 64 步全部为 `timeout`，不再出现上一版 raw_fcs 的 36~39 步 `own_crash`；
- 确定性评估 `fire_rate=0.0`：策略尚未学会发射，且动作掩码会禁用不满足条件的发射；符合“不要求任务性能达标”的本轮口径。

### 3.2 ActionV1 动作统计（评估片段）

| 智能体 | throttle 范围 | roll 范围 | pitch 范围 | yaw 范围 | 武器维 |
|---|---|---|---|---|---|
| `strike` | [0.098, 0.139] | [-0.050, -0.002] | [-0.052, 0.006] | [0.013, 0.076] | 全 0 |
| `patrol` | [0.027, 0.127] | [0.030, 0.079] | [0.082, 0.121] | [0.082, 0.116] | 全 0 |
| `defense` | [0.000, 0.000] | [-0.058, 0.029] | [0.042, 0.077] | [-0.017, 0.024] | 全 0 |
| `ew` | [0.000, 0.056] | [-0.007, 0.091] | [0.012, 0.053] | [0.062, 0.109] | 全 0 |

所有值均在 ActionV1 范围内，dtype 为 float32；`assisted` 下这些值解释为 AP 速度/滚转/高度设定，而非直接舵面。

---

## 4. 行为树最小闭环

命令：`run_expert_bt_demo.py --tasks strike,patrol,defense,ew --steps 12`

| 任务 | 行为树状态 | 步数 | 累计奖励 | 结局 |
|---|---|---:|---:|---|
| `strike` | SUCCESS | 12 | 60.105 | 未终止（演示截断） |
| `patrol` | SUCCESS | 12 | 0.100 | 未终止 |
| `defense` | SUCCESS | 12 | 18.544 | 未终止 |
| `ew` | SUCCESS | 12 | 60.344 | 未终止 |

产物（`docs/assets/stage3_expert_demo/`）：

- `strike.csv`、`patrol.csv`、`defense.csv`、`ew.csv`：逐帧 ActionV1/奖励/态势；
- `bt_expert_demo.csv`：合并记录；
- `demo_summary.json`：每任务步数/奖励/状态/模型加载；
- `demo_log.json`：行为树 `mission_start/mission_end` 事件；
- `demo_screenshot.png`：动作/奖励指标图。

演示总耗时 **7.87 s**，4 个任务均 `SUCCESS`，证明行为树可按任务名选择并驱动全部 4 个任务级智能体；每任务 12 步仅为接口闭环，不代表战术性能。

### 4.1 assisted 240 步稳定飞行基线

命令：`run_stability_baseline.py --combat`（4 任务 × seeds 7/17/27 × safe/model）。

- **隔离飞行**：24/24 通过，0 次 `own_crash`，全部 240 步 `timeout`；
- 安全动作高度最低 4587.9 m（初始约 5266 m，AP 高度保持静差约 100 m）；
- 512 步模型高度最低 3987.5 m，速度最低 113.7 m/s，仍无 `own_crash`；
- 正常战斗配置（n_missiles=2，seed 7）：4 任务均在 54 步 `own_hit`，`own_crash=False`，仅记录、不作硬性通过；
- 详细对照与根因见 `docs/阶段三_稳定飞行基线_实现与测试报告.md`。

### 4.2 raw_fcs 对照诊断

命令：`run_stability_baseline.py --tasks strike --seeds 42 --policies safe --control-mode raw_fcs --out-dir .../stage3_diagnosis`

- **45 步** `own_crash`，高度 -2.1 m，最大 |theta| 0.689 rad；
- 证明问题根因是 raw_fcs 无配平/俯仰保持，而非单纯“策略未训练/近零油门”。

---

## 5. pytest 汇总

| 测试套件 | 结果 | 耗时 |
|---|---:|---:|
| 新增 `多任务智能体/tests/test_expert_agents.py` | **24 passed** | 11.47 s（复跑） |
| `多任务智能体/tests`（含新增） | **80 passed** | 29.06 s（复跑） |
| `引导策略行为树/tests` | **19 passed** | 2.34 s |
| `可视化工具/tests` | **23 passed** | 1.96 s |
| **合计** | **122 passed，0 failed** | — |

警告：`gymnasium.spaces.box` 对技能级 `Box(3)` 的 float64 边界降精度提示，属于既有环境（`jsbsim_env.py`）历史行为，不影响测试结果；本轮未修改该接口。

### 5.1 新增测试覆盖

`test_expert_agents.py` 覆盖：

- ActionSpec/ActionV1/PolicyOutput/命令通道声明；
- `to_env_action`/`to_flat_action`/`from_env_action`/`to_ap_command` 边界与阈值；
- `compute_action_mask` 与发射条件（无弹/未跟踪/超包线/信息缺失）；
- `control_mode=raw_fcs/assisted` 切换、assisted 映射、240 步稳定性；
- ActionSpec 的 `control_mode`/映射字段；
- 注册表 9 项、tier、中文别名、实例化；
- 4 个任务级智能体实例化、`get_action`/`predict` 形状/范围/dtype；
- `guide_action/guide_steps` 透传与掩码；
- 4 个任务级智能体各 32 步小规模训练 + 保存/加载一致性；
- 任务环境标签/奖励/动作空间（扁平 + Dict），默认 assisted；
- py_trees 按中文任务名选择 4 个任务级智能体（假环境，验证调度逻辑）；
- 技能级 `cruise_hold` 的 command_v1 兼容通道可调用。

---

## 6. 技能级回归抽查

- 既有基线：`多任务智能体/validation/results/validation_report.txt`（5/5 通过，已恢复原文件）；
- 抽查 `cruise_hold` 复跑（`run_validation.py --tasks cruise_hold`）：
  - L1 环境冒烟 60 步：通过；
  - L2 PPO 模型构建：通过；
  - L3 极小训练 2048 步：**8.8 s**，通过；
  - 退出后原 5 任务报告已恢复，未破坏基线文件。
- 结论：5 个技能智能体的接口、环境与 CLI 未受本轮新增代码影响。

---

## 7. 失败分析

| 现象 | 原因 | 影响/处理 |
|---|---|---|
| raw_fcs 评估 36~45 步 `own_crash` | 本机 `autopilot=False` + `step_manual` 直接驱动 F-104 舵面，无配平/俯仰保持；零舵面时代码持续低头，不是单纯策略未训练 | 已由 `control_mode=assisted` 修复；raw_fcs 保留为兼容/对照，诊断见 §4.2 |
| assisted 240 步高度静差约 100 m | JSBSim AP 高度环为比例保持，存在稳态误差；安全基线足够（远高于 30 m 坠毁阈值） | 记录；后续可用积分/前馈调参减小静差 |
| `fire_rate=0` | 策略未学会发射；确定性策略武器维未过 0.5；掩码同时限制非法发射 | 预期；不要求本轮命中率 |
| `value_loss` 较大（~1e3~5e3） | 奖励尺度和 512 步未收敛；损失有限 | 记录为已知现象，后续奖励缩放/完整训练处理 |
| pytest 的 Box float64 警告 | 既有 `JSBSimFlightEnv` 动作边界 dtype | 历史行为，不影响 122 项测试通过 |
| 行为树演示每任务 12 步 | 中期检查最小闭环，避免长仿真 | 演示不声明任务性能 |
| 正常战斗配置 54 步 `own_hit` | 安全基线不做规避/发射战术；仅记录 | 不作飞行稳定性失败；后续训练评估 |

---

## 8. 复现与产物

- 验证报告：`多任务智能体/validation/results/stage3/stage3_expert_validation.{json,txt}`
- 稳定飞行：`多任务智能体/validation/results/stage3_stability/{stability_summary.json,stability_report.txt,combat_summary.json}`
- raw_fcs 诊断：`多任务智能体/validation/results/stage3_diagnosis/`
- 演示产物：`docs/assets/stage3_expert_demo/`
- 模型（不入库）：`多任务智能体/models/task_{strike,patrol,defense,ew}_ppo.zip` 及 `*.spec.json`
- 详细命令见 `docs/阶段三_专家智能体DRL_命令清单.md`。
