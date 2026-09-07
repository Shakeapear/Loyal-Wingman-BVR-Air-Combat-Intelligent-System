# 多任务智能体库（任务专精 DRL 智能体）

基于已入库开源项目（stable-baselines3 / CleanRL / tianshou / gym-jsbsim / py_trees）构建的**任务专精多任务 DRL 智能体代码**。设计原则：**不同任务使用不同 DRL 模型**，模型输入输出与项目数据格式（trajectory_v1）对齐。

## 目录结构（按类别组织）

```
多任务智能体\
├── README.md                     ← 本文件
├── common\
│   ├── jsbsim_env.py             ← JSBSim 单机 Gym 环境（步骤 2.3 桥接雏形，1Hz 决策 + 自驾仪内环）
│   ├── tasks.py                  ← 5 个任务定义与奖励函数（含虚拟目标）
│   └── config.py                 ← 网络结构/超参（节点数以文档为准）
├── agents\                       ← 任务专精智能体（每任务一个文件）
│   ├── base_agent.py             ← 基类：环境/模型/训练/推理/保存统一接口
│   ├── cruise_hold_ppo.py        ← 任务1 巡航保持 → PPO
│   ├── turn_sac.py               ← 任务2 盘旋机动 → SAC
│   ├── climb_descent_td3.py      ← 任务3 爬升/下降 → TD3
│   ├── pursuit_dqn.py            ← 任务4 接敌追击 → DQN（离散动作）
│   └── evasion_ppo.py            ← 任务5 威胁规避 → PPO
├── models\                       ← 验证性运行产出的模型（不入库，见 .gitignore）
├── validation\
│   ├── run_validation.py         ← 验证性运行（L1 冒烟 / L2 建模型 / L3 极小训练）
│   └── results\                  ← 验证报告与运行日志
└── visualization\                ← 步骤 2.4 可视化（2D 战术显示 + CSV/GIF 回放 + ACMI 导出，
                                    见 visualization/README.md；demo：python -m visualization.demo）
```

## 任务-模型-网络对照表（节点数依据开源项目文档）

| 任务 | DRL 模型 | net_arch | 激活 | 依据 |
|------|---------|:---:|:---:|------|
| 巡航保持 cruise_hold | PPO (SB3) | [64,64] | tanh | SB3/CleanRL 简单连续任务文档默认 |
| 盘旋机动 turn | SAC (SB3) | [128,128] | relu | 中等复杂度，维度协调更多 |
| 爬升/下降 climb_descent | TD3 (SB3) | [128,128] | relu | 能量管理任务，中等 |
| 接敌追击 pursuit | DQN (SB3) | [128,128] | relu | 离散 48 动作 × 17 维观测 |
| 威胁规避 evasion | PPO (SB3) | [256,256] | tanh | tianshou MuJoCo 文档默认（奖励稀疏、态势复杂） |

## 输入输出（与 trajectory_v1 对齐）

- **观测输入**：12 维标准状态（h/v/mach/欧拉角/迎角/过载/油门/油量/垂速）+ 任务扩展（追击/规避含目标相对位置 4-5 维），SI 单位；
- **动作输出（连续）**：`[速度指令 m/s, 滚转角指令 rad, 高度指令 m]` —— 与轨迹文件 `action_speed_cmd_mps / action_roll_cmd_rad / action_alt_cmd_m` 完全一致；
- **动作输出（追击 DQN）**：48 离散动作 = 16 航向档 × 3 油门档；
- **决策频率**：1 Hz（内环 dt=1/120 s 由 F-104 自驾仪 autopilot-hold.xml 执行）。

## 环境配置（DC，conda）

```powershell
# 已完成：conda 创建 DC 环境（Python 3.12）+ 依赖安装（见安装日志）
# 依赖：torch(CPU) numpy pandas matplotlib scipy pyyaml gymnasium
#       stable-baselines3 py_trees tianshou jsbsim
& "C:\Users\36266\.conda\envs\DC\python.exe" -c "import torch, stable_baselines3, jsbsim, py_trees; print('DC 环境 OK')"
```

## 验证性运行（当前阶段，不执行完整训练）

```powershell
# 全部任务三级验证（冒烟 → 建模 → 极小训练 1024~2048 步）
& "C:\Users\36266\.conda\envs\DC\python.exe" "多任务智能体\validation\run_validation.py"

# 只验证部分任务 / 跳过训练
& "C:\Users\36266\.conda\envs\DC\python.exe" "多任务智能体\validation\run_validation.py" --tasks cruise_hold --skip-train
```

**完整训练**（待验证通过后另行执行，各任务 50 万~200 万步，见 config.py `total_timesteps_full`）：
```powershell
& "C:\Users\36266\.conda\envs\DC\python.exe" -m agents.cruise_hold_ppo
```

## 非 Python 语言部分的重写

下载的开源项目中仅两处含 C++，处理如下：
1. **BTGenBot bt_validator（C++/ROS2）** → 已按原逻辑重写为纯 Python：
   `开源项目库\03_行为树与LLM\BTGenBot\Python重写_bt_validator\bt_validator.py`（XML 良构/节点库/结构约束校验，含 10 个原版测试树样例）；
2. **MARLlib 的 hanabi C++ 卡牌引擎** → 与空战领域无关（纸牌游戏环境依赖），不做重写，仅记录说明。
