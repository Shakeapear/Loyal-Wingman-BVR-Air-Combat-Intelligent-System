# -*- coding: utf-8 -*-
"""
common/tasks.py
===============
任务定义与奖励函数。每个任务 = 一个奖励函数 + 任务参数 + 观测/动作空间设定。

任务清单（任务专精，不同任务不同 DRL 模型，见 agents/）：
    1. cruise_hold    巡航保持    PPO  连续动作
    2. turn           盘旋机动    SAC  连续动作
    3. climb_descent  爬升/下降   TD3  连续动作
    4. pursuit        接敌追击    DQN  离散动作（16 航向 × 3 油门）
    5. evasion        威胁规避    PPO  连续动作（带虚拟威胁目标）
"""
import numpy as np


def _penalty_done(env):
    return -100.0 if env._get("position/h-agl-ft") * 0.3048 < 30.0 else 0.0


# ---------- 任务 1：巡航保持 ----------

def cruise_hold_reward(env):
    h = env._get("position/h-sl-ft") * 0.3048
    v = env._get("velocities/vtrue-fps") * 0.3048
    phi = env._get("attitude/phi-rad")
    h_err = abs(h - env.cfg["target_alt_m"])
    v_err = abs(v - env.cfg["target_speed_mps"])
    r = 1.0 - 0.5 * min(h_err / 300.0, 1.0) - 0.5 * min(v_err / 50.0, 1.0) - 0.2 * min(abs(phi) / 0.5, 1.0)
    return r + _penalty_done(env)


# ---------- 任务 2：盘旋机动 ----------

def turn_reward(env):
    phi = env._get("attitude/phi-rad")
    h = env._get("position/h-sl-ft") * 0.3048
    v = env._get("velocities/vtrue-fps") * 0.3048
    r = (1.0 - min(abs(phi - env.cfg["target_bank_rad"]) / 0.6, 1.0)
         - 0.2 * min(abs(h - env.cfg["target_alt_m"]) / 500.0, 1.0)
         - 0.3 * min(abs(v - env.cfg["target_speed_mps"]) / 60.0, 1.0))
    return r + _penalty_done(env)


# ---------- 任务 3：爬升/下降 ----------

def climb_descent_reward(env):
    h = env._get("position/h-sl-ft") * 0.3048
    err = abs(h - env.cfg["target_alt_m"])
    r = 1.0 - min(err / 400.0, 1.0)
    if err < 100.0:
        r += 0.5
    if err < 30.0:                    # 达到高度带，任务完成
        r += 5.0
    return r + _penalty_done(env)


def climb_descent_done(env):
    return abs(env._get("position/h-sl-ft") * 0.3048 - env.cfg["target_alt_m"]) < 30.0


# ---------- 任务 4：接敌追击（离散） ----------

def pursuit_reward(env):
    dist = float(np.hypot(env.target.x, env.target.y))
    dh = abs(env.target.h - env._get("position/h-sl-ft") * 0.3048)
    prev = env._last_pursuit_dist
    env._last_pursuit_dist = dist
    r = 0.8 * (prev - dist) / 100.0          # 接近率奖励
    r += 0.3 * (1.0 - min(dist / 6000.0, 1.0))  # 距离优势
    r -= 0.2 * min(dh / 1000.0, 1.0)             # 高度差惩罚
    return r + _penalty_done(env)


def pursuit_done(env):
    return float(np.hypot(env.target.x, env.target.y)) < 800.0


# ---------- 任务 5：威胁规避（连续，虚拟威胁逼近） ----------

def evasion_reward(env):
    dist = float(np.hypot(env.target.x, env.target.y))
    prev = env._last_evasion_dist
    env._last_evasion_dist = dist
    r = 0.3 * (dist - prev) / 100.0            # 拉开距离奖励
    r += 0.2 * min(dist / 4000.0, 1.0)         # 保持距离
    r -= 0.5 * env._get("accelerations/Nz") > 4.0  # 剧烈机动轻微惩罚
    return r + _penalty_done(env)


# ---------- 任务注册表 ----------

TASKS = {
    "cruise_hold": {
        "target_alt_m": 4572.0,      # 15000 ft
        "target_speed_mps": 228.6,   # 750 fps
        "max_steps": 120,
        "extra_obs": [],
        "reward_fn": cruise_hold_reward,
        "algo": "PPO", "model_name": "cruise_hold_ppo",
    },
    "turn": {
        "target_bank_rad": 0.5236,   # 30°
        "target_alt_m": 4572.0,
        "target_speed_mps": 228.6,
        "max_steps": 120,
        "extra_obs": [],
        "reward_fn": turn_reward,
        "algo": "SAC", "model_name": "turn_sac",
    },
    "climb_descent": {
        "target_alt_m": 5334.0,      # 17500 ft
        "max_steps": 150,
        "extra_obs": [],
        "reward_fn": climb_descent_reward,
        "done_fn": climb_descent_done,
        "algo": "TD3", "model_name": "climb_descent_td3",
    },
    "pursuit": {
        "discrete_action": True,
        "n_heading": 16, "n_throttle": 3, "n_discrete": 48,
        "use_target": True,
        "target_kwargs": {"x0": 5000.0, "y0": -3000.0, "v": 180.0,
                          "turn_rate": 0.05, "h0": 5000.0},
        "extra_obs": ["rel_north_m", "rel_east_m", "rel_alt_m", "dist_m", "bearing_rad"],
        "max_steps": 120,
        "reward_fn": pursuit_reward,
        "done_fn": pursuit_done,
        "algo": "DQN", "model_name": "pursuit_dqn",
    },
    "evasion": {
        "use_target": True,
        "target_kwargs": {"x0": -2000.0, "y0": 2500.0, "v": 260.0,
                          "turn_rate": -0.08, "h0": 4572.0},
        "extra_obs": ["rel_north_m", "rel_east_m", "rel_alt_m", "dist_m"],
        "max_steps": 120,
        "reward_fn": evasion_reward,
        "algo": "PPO", "model_name": "evasion_ppo",
    },
}


def init_task_state(env):
    """任务相关的 env 私有状态初始化（reset 后调用）。"""
    if env.target is not None:
        env._last_pursuit_dist = float(np.hypot(env.target.x, env.target.y))
        env._last_evasion_dist = env._last_pursuit_dist
