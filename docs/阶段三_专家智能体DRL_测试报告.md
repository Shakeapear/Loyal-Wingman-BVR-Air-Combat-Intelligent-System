# 阶段三专家智能体 DRL 底座测试报告

> 日期：2026-10-30｜环境：Windows + PowerShell 7｜conda `DC`（Python 3.12.13）｜CPU 训练
>
> 结论：**4/4 任务级智能体通过接口/小规模训练/保存加载验证；新增 22 项测试通过；三套既有测试全绿（合计 120 项）；行为树成功调度 4 个任务级智能体完成最小闭环。**

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows（win32） |
| Shell | PowerShell 7（`pwsh`） |
| Python | 3.12.13（`C:\Users\36266\.conda\envs\DC\python.exe`） |
| 关键依赖 | torch CPU、gymnasium、stable-baselines3、py_trees、jsbsim 1.3.1、numpy、matplotlib |
| 固定随机种子 | 42（验证脚本）；演示种子 7~10 |
| 训练规模 | 每任务 512 步（非完整训练） |
| 环境近似 | `compute_envelope=False`、`max_steps=64`（加速验证；非最终场景） |

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
```

---

## 3. 任务级智能体验证结果

数据文件：`多任务智能体/validation/results/stage3/stage3_expert_validation.json`、`stage3_expert_validation.txt`。

| 智能体 | 实例化/动作 | `get_action` 形状/dtype/范围 | 小规模训练 | 训练耗时 | 损失有限 | 评估步数 | 评估奖励均值 | 保存/加载一致（最大差） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `strike` | 通过 | (7,) float32 / 界内 | 512 步 | 1.540 s | 是 | 39 | +0.0079 | 是（0.0） |
| `patrol` | 通过 | (7,) float32 / 界内 | 512 步 | 1.564 s | 是 | 37 | -0.2624 | 是（0.0） |
| `defense` | 通过 | (7,) float32 / 界内 | 512 步 | 1.506 s | 是 | 39 | -0.1220 | 是（0.0） |
| `ew` | 通过 | (7,) float32 / 界内 | 512 步 | 1.530 s | 是 | 36 | -0.2519 | 是（0.0） |

汇总：**4/4 通过，脚本总耗时 9.231 s**（不含 pytest 与既有技能回归）。

### 3.1 训练损失/奖励（关键值）

| 智能体 | value_loss | policy_gradient_loss | entropy_loss | approx_kl | 评估奖励有限 |
|---|---:|---:|---:|---:|---:|
| `strike` | 2007.53 | -0.01286 | -9.9230 | 0.007422 | 是 |
| `patrol` | 2935.45 | -0.00775 | -9.9201 | 0.001481 | 是 |
| `defense` | 2637.51 | -0.00878 | -9.9320 | 0.004848 | 是 |
| `ew` | 2117.48 | -0.00813 | -9.9403 | 0.002651 | 是 |

说明：

- 损失/奖励全部为有限值；`value_loss` 量级较大与 BVR 奖励尺度（命中 ±200、攻击区 +5/步）一致，512 步远未收敛，不作为性能指标；
- 评估中 4 个智能体均在 36~39 步因 `own_crash` 终止（随机/极小训练策略近零油门导致），属于预期的“底座可运行”现象，不是接口失败；
- 确定性评估 `fire_rate=0.0`：策略尚未学会发射，且动作掩码会禁用不满足条件的发射；符合“不要求任务性能达标”的本轮口径。

### 3.2 ActionV1 动作统计（评估片段）

| 智能体 | throttle 范围 | roll 范围 | pitch 范围 | yaw 范围 | 武器维 |
|---|---|---|---|---|---|
| `strike` | [0.025, 0.064] | [-0.034, 0.064] | [-0.098, -0.054] | [0.045, 0.143] | 全 0 |
| `patrol` | [0.000, 0.033] | [-0.093, -0.042] | [-0.100, -0.009] | [0.019, 0.049] | 全 0 |
| `defense` | [0.000, 0.000] | [-0.034, -0.005] | [0.001, 0.077] | [0.038, 0.074] | 全 0 |
| `ew` | [0.000, 0.000] | [-0.047, -0.013] | [0.079, 0.130] | [-0.045, -0.026] | 全 0 |

所有值均在 ActionV1 范围内，dtype 为 float32。

---

## 4. 行为树最小闭环

命令：`run_expert_bt_demo.py --tasks strike,patrol,defense,ew --steps 12`

| 任务 | 行为树状态 | 步数 | 累计奖励 | 结局 |
|---|---|---:|---:|---|
| `strike` | SUCCESS | 12 | 60.106 | 未终止（演示截断） |
| `patrol` | SUCCESS | 12 | 0.095 | 未终止 |
| `defense` | SUCCESS | 12 | 18.545 | 未终止 |
| `ew` | SUCCESS | 12 | 60.345 | 未终止 |

产物（`docs/assets/stage3_expert_demo/`）：

- `strike.csv`、`patrol.csv`、`defense.csv`、`ew.csv`：逐帧 ActionV1/奖励/态势；
- `bt_expert_demo.csv`：合并记录；
- `demo_summary.json`：每任务步数/奖励/状态/模型加载；
- `demo_log.json`：行为树 `mission_start/mission_end` 事件；
- `demo_screenshot.png`：动作/奖励指标图。

演示总耗时 **20.4 s**，4 个任务均 `SUCCESS`，证明行为树可按任务名选择并驱动全部 4 个任务级智能体；每任务 12 步仅为接口闭环，不代表战术性能。

---

## 5. pytest 汇总

| 测试套件 | 结果 | 耗时 |
|---|---:|---:|
| 新增 `多任务智能体/tests/test_expert_agents.py` | **22 passed** | 7.29 s（复跑） |
| `多任务智能体/tests`（含新增） | **78 passed** | 18.21 s（复跑） |
| `引导策略行为树/tests` | **19 passed** | 2.34 s |
| `可视化工具/tests` | **23 passed** | 1.96 s |
| **合计** | **120 passed，0 failed** | — |

警告：`gymnasium.spaces.box` 对技能级 `Box(3)` 的 float64 边界降精度提示，属于既有环境（`jsbsim_env.py`）历史行为，不影响测试结果；本轮未修改该接口。

### 5.1 新增测试覆盖

`test_expert_agents.py` 覆盖：

- ActionSpec/ActionV1/PolicyOutput/命令通道声明；
- `to_env_action`/`to_flat_action`/`from_env_action`/`to_ap_command` 边界与阈值；
- `compute_action_mask` 与发射条件（无弹/未跟踪/超包线/信息缺失）；
- 注册表 9 项、tier、中文别名、实例化；
- 4 个任务级智能体实例化、`get_action`/`predict` 形状/范围/dtype；
- `guide_action/guide_steps` 透传与掩码；
- 4 个任务级智能体各 32 步小规模训练 + 保存/加载一致性；
- 任务环境标签/奖励/动作空间（扁平 + Dict）；
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
| 评估片段 36~39 步 `own_crash` | 512 步极小训练策略近零油门/随机姿态；不是接口异常 | 本轮只验收“可训练、有限、可保存加载”；任务性能后续完整训练 |
| `fire_rate=0` | 策略未学会发射；确定性策略武器维未过 0.5；掩码同时限制非法发射 | 预期；不要求本轮命中率 |
| `value_loss` 较大（~2e3） | 奖励尺度和 512 步未收敛；损失有限 | 记录为已知现象，后续奖励缩放/完整训练处理 |
| pytest 的 Box float64 警告 | 既有 `JSBSimFlightEnv` 动作边界 dtype | 历史行为，不影响 120 项测试通过 |
| 行为树演示每任务 12 步 | 中期检查最小闭环，避免长仿真 | 演示不声明任务性能 |

---

## 8. 复现与产物

- 验证报告：`多任务智能体/validation/results/stage3/stage3_expert_validation.{json,txt}`
- 演示产物：`docs/assets/stage3_expert_demo/`
- 模型（不入库）：`多任务智能体/models/task_{strike,patrol,defense,ew}_ppo.zip` 及 `*.spec.json`
- 详细命令见 `docs/阶段三_专家智能体DRL_命令清单.md`。
