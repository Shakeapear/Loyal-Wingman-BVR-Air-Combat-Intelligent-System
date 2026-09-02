# -*- coding: utf-8 -*-
"""
agents/base_agent.py
====================
任务专精智能体基类：统一"环境构建 → 模型构建 → 训练 → 推理 → 保存"接口。

每个任务一个子类（不同任务使用不同 DRL 模型，见同目录各任务文件）：
    - cruise_hold_ppo.py    巡航保持  PPO
    - turn_sac.py           盘旋      SAC
    - climb_descent_td3.py  爬升下降  TD3
    - pursuit_dqn.py        接敌追击  DQN（离散动作）
    - evasion_ppo.py        威胁规避  PPO
"""
from pathlib import Path

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

from common.config import TASK_CONFIGS
from common.jsbsim_env import JSBSimFlightEnv
from common.tasks import TASKS, init_task_state

ALGO_REGISTRY = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DQN": DQN, "A2C": A2C}

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class TaskAgent:
    """任务专精智能体基类。"""

    task_id = None           # 子类必须指定，对应 TASKS 键

    def __init__(self):
        if self.task_id is None or self.task_id not in TASKS:
            raise ValueError(f"无效任务编号: {self.task_id}")
        self.task_cfg = dict(TASKS[self.task_id])
        self.train_cfg = dict(TASK_CONFIGS[self.task_id])
        self.algo_cls = ALGO_REGISTRY[self.train_cfg["algo"]]
        self.env = None
        self.model = None

    # ---------- 环境 ----------

    def build_env(self):
        self.env = JSBSimFlightEnv(self.task_cfg)
        return self.env

    def reset_env(self):
        obs, info = self.env.reset()
        init_task_state(self.env)
        return obs, info

    # ---------- 模型 ----------

    def build_model(self):
        """按 config.py 中的节点数构建模型（不训练）。"""
        import torch.nn as nn

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[self.train_cfg["activation"]]

        policy_kwargs = {
            "net_arch": self.train_cfg["net_arch"],
            "activation_fn": act_fn,
        }
        kwargs = dict(policy=self.train_cfg["policy"], env=self.build_env(),
                      policy_kwargs=policy_kwargs,
                      learning_rate=self.train_cfg["learning_rate"], verbose=1)
        if self.train_cfg["algo"] == "DQN":
            kwargs["learning_starts"] = 256      # 验证性运行：快速进入学习
            kwargs["train_freq"] = 4
        self.model = self.algo_cls(**kwargs)
        return self.model

    # ---------- 训练与推理 ----------

    def train(self, timesteps=None):
        if self.model is None:
            self.build_model()
        ts = timesteps or self.train_cfg.get("total_timesteps_full", 1_000_000)
        self.model.learn(total_timesteps=ts, progress_bar=True)
        return self.model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)[0]

    def rollout(self, n_steps=120, deterministic=True):
        """推理 rollout：返回轨迹 dict（与 trajectory_v1 字段对应）。"""
        obs, _ = self.reset_env()
        traj = {"t_s": [], "h_sl_m": [], "vtrue_mps": [], "phi_rad": [],
                "action_speed_cmd_mps": [], "action_roll_cmd_rad": [],
                "action_alt_cmd_m": [], "reward": []}
        for t in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)
            traj["t_s"].append(t)
            traj["h_sl_m"].append(info["h_sl_m"])
            traj["vtrue_mps"].append(info["vtrue_mps"])
            traj["action_speed_cmd_mps"].append(float(action[0]) if hasattr(action, "__len__") else float(action))
            traj["action_roll_cmd_rad"].append(float(action[1]) if hasattr(action, "__len__") and len(action) > 1 else 0.0)
            traj["action_alt_cmd_m"].append(float(action[2]) if hasattr(action, "__len__") and len(action) > 2 else 0.0)
            traj["reward"].append(float(reward))
            if terminated or truncated:
                break
        return traj

    # ---------- 存取 ----------

    def save(self, path=None):
        path = Path(path) if path else MODEL_DIR / f"{self.task_cfg['model_name']}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"[ok] 模型已保存: {path}")
        return path

    def load(self, path=None):
        path = Path(path) if path else MODEL_DIR / f"{self.task_cfg['model_name']}"
        self.model = self.algo_cls.load(str(path), env=self.build_env())
        return self.model
