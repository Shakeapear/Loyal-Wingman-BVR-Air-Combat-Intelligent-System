# MARLlib（多智能体强化学习算法库）

## 简介

MARLlib 是统一的多智能体强化学习算法库（1.3k★，MIT，PyTorch），把 MAPPO、HAPPO、MATRPO、QMIX、MADDPG 等主流 MARL 算法统一为"策略类"接口，基于 Ray/RLlib 构建，支持自定义多智能体环境。

- 仓库：https://github.com/Replicable-MARL/MARLlib
- 许可证：MIT
- 源码位置：`源码\marl\algos\`（各算法）、`源码\examples\`

## 选择理由

1. 阶段四效能评估含 **1v2 / 2v2** 多机场景，MAPPO/MATRPO 是本项目多机协同训练的候选算法；
2. 统一接口便于在"单机 PPO（SB3）→ 多机 MAPPO（MARLlib）"间平滑迁移；
3. 论文配套库（可复现性），学术引用完整。

## 运行说明

```powershell
# 环境要求：Linux 优先（Ray 依赖），Windows 建议 WSL2
pip install marllib    # 或按仓库 README 源码安装

# 官方示例（多智能体 MuJoCo / MPE）：
python marl/exp_scripts/envs/mpe/mappo.py    # 参照仓库 examples 启动脚本
```

> 备注：若仅需 MAPPO 参考实现而不引入 Ray 依赖，可优先阅读 `飞行数据集资料库\02_多机空战与协同\CloseAirCombat_JSBSim空战环境` 内的 `algorithms\mappo`（纯 PyTorch，无 Ray），两处对照学习。
