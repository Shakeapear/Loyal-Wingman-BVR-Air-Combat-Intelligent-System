# BVRGym（超视距空战 RL 环境 + 行为树对手，已下载）

## 基本信息

| 项目 | 内容 |
|------|------|
| 仓库 | https://github.com/xcwoid/BVRGym |
| 论文 | arXiv:2403.17533 "BVR Gym: A Reinforcement Learning Environment for Beyond-Visual-Range Air Combat"（KTH + SAAB） |
| 动力学 | JSBSim（F-16 等机型 + F-22/F-15 模型目录） |
| 武器 | BVR 中距导弹（比例导引 PN），F-16 模型配高层自驾仪控制器 |
| 依赖 | gymnasium、py_trees、stable_baselines3、jsbsim |
| 许可 | 开源（仓库 LICENSE 文件已随包） |

## 与本项目最相关的内容（重点学习）

**1. 三级架构**（源码 `jsb_gym\`）：
- Level 1 `simObjects`：JSBSim 交互对象（aircraft.py、missiles.py、FDMObject.py）；
- Level 2 `agents`：智能体封装（飞机+导弹）；
- Level 3 `envs`：Gym 环境与场景配置（WvrEnv.py、config\）。

**2. 行为树实现**（源码 `jsb_gym\bts\BVR` 与 `bts\WVR`，py_trees 框架）——正是本项目"LLM 行为树生成"的**节点库参考**：
- 条件节点（conditions.py）：`MAW_own_condition`（本机导弹是否还在飞行）、`MAW_condition`（是否遭导弹威胁）、`Launch_condition`（发射条件）、`Pursue_condition`（追击判定）等；
- 动作节点（actions.py）：`MAW_guide_evade_action`（引导规避：敌方位+80°）、`MAW_evade_action`（180° 急转规避）、`Guide_own_action`（接敌引导：敌方位+45°）、发射导弹动作等；
- 另有 `test_BTvsBT` 数据输出模板（行为树对行为树对抗测试）。

**3. 场景与试验**（论文 + `tests\`）：
- 单/双导弹场景、BVR 狗斗（双方各 2 枚中距弹、最长 16 分钟）；
- 行为树（BT）策略作为 RL 训练对手与基线——与本项目阶段二"引导策略行为树"定位一致。

## 快速了解（组员）

1. 先读 `源码\README.md`（有动图与命令）；
2. 重点看 `jsb_gym\bts\BVR\actions.py / conditions.py`——节点命名与返回状态（SUCCESS/FAILURE/RUNNING）写法；
3. `data_output\test_BTvsBT`、`test_wvr` 是输出字段模板，可与本项目数据输出指令对照。

## 运行要求

Python 3.11 + `pip install gymnasium jsbsim pymap3d pandas py_trees stable_baselines3 tensorboard`；可视化可选 FlightGear（Linux 优先）。本项目 Windows 环境可只读代码借鉴，不强制运行。
