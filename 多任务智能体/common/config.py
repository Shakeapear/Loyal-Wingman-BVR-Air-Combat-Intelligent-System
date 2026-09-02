# -*- coding: utf-8 -*-
"""
common/config.py
================
网络结构与超参数配置（节点数：成本与性能权衡，依据开源项目官方文档默认值）

节点数依据（以文档为准）：
- stable-baselines3 文档（rl_zoo3 超参表）：简单连续控制任务默认 net_arch=[64,64]，
  复杂任务为 [256,256]；PPO 原始论文用 [400,300]。
- CleanRL ppo_continuous_action.py：默认 [64,64] tanh（MuJoCo 全任务统一）。
- tianshou MuJoCo 示例：hidden_sizes=[256,256] tanh。

本项目的取值原则：
- 观测维度小（12~17 维）、任务简单的（巡航保持）：[64,64]（最小成本）；
- 态势相关、需要非线性决策的（盘旋/爬升下降/追击）：[128,128]；
- 最复杂、奖励稀疏的（规避）：[256,256]（参照 tianshou 默认）。
"""
import yaml

# 每个任务的网络与算法超参（验证性运行用小步数，完整训练值见 total_timesteps_full）
TASK_CONFIGS = {
    "cruise_hold": {
        "algo": "PPO",
        "policy": "MlpPolicy",
        "net_arch": [64, 64],            # SB3/CleanRL 简单任务文档默认
        "activation": "tanh",
        "learning_rate": 3e-4,
        "total_timesteps_full": 1_000_000,
        "validation_timesteps": 2048,    # 仅验证性运行
        "note": "观测 12 维、奖励稠密、任务简单，取最小网络降低成本",
    },
    "turn": {
        "algo": "SAC",
        "policy": "MlpPolicy",
        "net_arch": [128, 128],          # 中等任务
        "activation": "relu",
        "learning_rate": 3e-4,
        "total_timesteps_full": 1_000_000,
        "validation_timesteps": 1024,
        "note": "盘旋需协调坡度/高度/速度三维，中等网络",
    },
    "climb_descent": {
        "algo": "TD3",
        "policy": "MlpPolicy",
        "net_arch": [128, 128],
        "activation": "relu",
        "learning_rate": 3e-4,
        "total_timesteps_full": 1_000_000,
        "validation_timesteps": 1024,
        "note": "能量管理任务，中等网络",
    },
    "pursuit": {
        "algo": "DQN",
        "policy": "MlpPolicy",
        "net_arch": [128, 128],
        "activation": "relu",
        "learning_rate": 1e-3,
        "total_timesteps_full": 500_000,
        "validation_timesteps": 1024,
        "note": "离散 48 动作 + 17 维观测，中等网络",
    },
    "evasion": {
        "algo": "PPO",
        "policy": "MlpPolicy",
        "net_arch": [256, 256],          # tianshou MuJoCo 文档默认
        "activation": "tanh",
        "learning_rate": 3e-4,
        "total_timesteps_full": 2_000_000,
        "validation_timesteps": 1024,
        "note": "奖励稀疏、态势感知要求高，取 tianshou 默认大网络",
    },
}


def save_configs(path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(TASK_CONFIGS, f, allow_unicode=True, sort_keys=False)


def load_configs(path=None):
    if path is None:
        return TASK_CONFIGS
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
