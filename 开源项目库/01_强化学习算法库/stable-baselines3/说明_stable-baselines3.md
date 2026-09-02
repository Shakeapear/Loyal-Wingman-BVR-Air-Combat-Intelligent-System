# stable-baselines3（PPO/SAC/TD3 主流实现）

## 简介

Stable Baselines3 (SB3) 是目前最主流的 PyTorch 深度强化学习算法库，提供 PPO、SAC、TD3、A2C、DQN 等可靠实现，是官方推荐的 RL 基线库。本包为 GitHub master 分支源码（13.7k★，MIT 许可，2026-08 仍在活跃更新）。

- 仓库：https://github.com/DLR-RM/stable-baselines3
- 许可证：MIT
- 源码位置：`源码\stable_baselines3\`（算法实现）、`源码\docs\`（教程）

## 选择理由

1. 本项目计划书阶段三的 **PPO 算法**以此为标准实现；`jumpstart-rl`（JSRL）也基于 SB3 开发，两者可无缝配合；
2. 提供 `gymnasium` 接口封装，与本项目自建环境（步骤 2.3 Gym 桥接）直接兼容；
3. 社区最活跃、文档最全、bug 最少，是"可运行、维护活跃"的标杆。

## 运行说明

```powershell
# 1. 安装（官方 PyPI 包，与本目录源码同源）
pip install stable-baselines3

# 2. 最小示例：在自定义环境上训练 PPO（伪代码，环境换成我们的 JSBSimGym）
import gymnasium as gym
from stable_baselines3 import PPO
env = gym.make("JSBSimEnv-v0")          # 本项目自建环境
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_jsbsim")

# 3. 推理
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

> 组员学习路径：`源码\docs\guide\`（官方指南）→ `源码\stable_baselines3\common\policies.py`（网络结构）→ `stable_baselines3\ppo\ppo.py`（PPO 核心）。
