# CleanRL（单文件 PPO 实现）

## 简介

CleanRL 以"每个算法一个自包含 Python 文件"著称（10.3k★，MIT 许可，活跃维护），PPO/SAC/TDQN 等算法的训练循环 + 网络 + 日志全部在单文件内，是深度定制 DRL 算法的首选起点。

- 仓库：https://github.com/vwxyzjn/cleanrl
- 许可证：MIT
- 源码位置：`源码\cleanrl\`（全部单文件算法，含 ppo.py、ppo_continuous_action.py 等 50+ 算法）

## 本目录文件

| 路径 | 内容 |
|------|------|
| `源码\cleanrl\ppo.py` | 离散动作 PPO（官方旗舰文件） |
| `源码\cleanrl\ppo_continuous_action.py` | 连续动作 PPO（**与本项目动作空间一致，重点参考**） |
| `源码\docs\` | 官方文档与算法说明 |

## 选择理由

1. 本项目动作空间为**连续**（油门/坡度/俯仰指令），`ppo_continuous_action.py` 是单文件连续控制 PPO 的标准参考，改奖励/观测/网络只需编辑一个文件；
2. 代码行数少、注释清楚，适合组员对照 SB3 学习 PPO 内部机制；
3. 内置 wandb/tensorboard 实验管理与 `--env-id` 通用接口，便于挂接自定义 Gym 环境。

## 运行说明

```powershell
# 官方示例（连续动作，MuJoCo）：
python cleanrl_ppo_continuous_action.py --env-id HalfCheetah-v5 --total-timesteps 1000000

# 换成本项目环境（注册自定义 gym id 后）：
python cleanrl_ppo_continuous_action.py --env-id JSBSimEnv-v0 --total-timesteps 1000000 --track
```

> 组员学习路径：先读 `ppo_continuous_action.py` 顶部注释（超参数说明）→ `Agent.learn()` 内的 PPO 更新循环。
