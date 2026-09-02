# CloseAirCombat（JSBSim 空战环境，多机，已下载）

## 基本信息

| 项目 | 内容 |
|------|------|
| 仓库 | https://github.com/Chengc1/CloseAirCombat（中文团队开源） |
| 论文 | "Light Aircraft Game" 系列（详见仓库 README） |
| 动力学 | JSBSim 1.1.6（F-16 等机型） |
| 武器 | 比例导引（PN）导弹模型（自实现，含文档 `docs/missile_engine`） |
| 算法 | PPO、MAPPO（`algorithms\` 目录），支持自博弈与对基线训练 |
| 许可 | 开源（使用前核对仓库 LICENSE） |

## 提供的任务（场景）

| 场景 | 规模 | 内容 |
|------|------|------|
| SingleControl | 单机 | 航向/高度/速度跟踪控制任务（训练底层控制策略） |
| SingleCombat 1v1 | 双机对抗 | NoWeapon 态势占位（咬尾）、Missile 发射/规避 |
| MultiCombat 2v2 | 四机对抗 | 协同对抗，MAPPO 多智能体训练 |

**与本项目高度契合的设计点：**
1. **分层框架**：高层策略输出（航向/高度/速度指令）→ 底层控制用 SingleControl 训练好的模型执行——这正是本项目计划书"行为树决策 → 姿态控制"的分层思路；
2. **比例导引导弹 + 命中判定**：对应计划书步骤 2.2 的导弹模型；
3. **1v1/2v2 任务配置**：对应计划书阶段四效能评估的场景矩阵（1v1/1v2/2v1）。

## 目录说明

- `源码\algorithms\ppo`、`algorithms\mappo`：PPO/MAPPO 实现（训练参考）；
- `源码\README.md`：安装与训练脚本说明；
- ⚠ 本包**不含环境代码**：仓库的 `envs/JSBSim` 是 git 子模块，需按 README 执行 `git submodule init; git submodule update` 获取（完整包也可从 gitee 镜像 https://gitee.com/xiaohuiduan/CloseAirCombat 克隆）。

## 复现步骤（组员）

```shell
git clone --recurse-submodules https://github.com/Chengc1/CloseAirCombat.git
conda create -n jsbsim python=3.8
pip install torch pymap3d jsbsim==1.1.6 geographiclib gym==0.20.0
cd scripts && bash train_heading.sh   # 先训练底层控制，再训 1v1/2v2
```
