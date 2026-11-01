# -*- coding: utf-8 -*-
"""
agents/task_agents/mission_env.py
=================================
阶段三中期检查版任务级环境适配（单机近似，不含多机协同）。

在现有 1v1 `BVRCombatEnv` / `FlatActionBVRCombatEnv` 之上叠加：
1. 任务标签编码（沿用 28 维观测的 26/27 两维，2-bit 编码，待 obs_v2 扩 4 维）；
2. 任务特定奖励塑形（仅为中期检查可训练性，不代表最终任务指标）；
3. 扁平训练环境动作空间收窄为 ActionV1（武器维 0~1），物理/终止不变；
4. 固定尺度观测包装（策略内部使用，外部仍以原始 28 维观测为契约）。

任务映射（中期检查版）：
    strike   超视距打击  label=(1,0)  复用 BVR 基线，无额外塑形
    patrol   空域巡逻    label=(0,1)  单机拦截近似（跟踪/攻击区奖励）
    defense  战术防御    label=(1,1)  单机生存近似（ECM/告警/距离）
    ew       电子压制    label=(0,0)  单机干扰近似（ECM 与敌雷达状态）

多机巡逻/防御/HVA 保护按计划书阶段三 3.3 后续扩展，本文件不实现。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.policy_interface import ACTION_V1_HIGH, ACTION_V1_LOW
from common.bvr_combat_env import (BVRCombatEnv, FlatActionBVRCombatEnv, OBS_HIGH,
                                   OBS_LOW)
from common.errors import ARG_INVALID, ValidationError

MISSION_IDS = ("strike", "patrol", "defense", "ew")
MISSION_LABELS = {
    "strike": (1.0, 0.0),
    "patrol": (0.0, 1.0),
    "defense": (1.0, 1.0),
    "ew": (0.0, 0.0),
}
MISSION_NAMES_ZH = {
    "strike": "超视距打击",
    "patrol": "空域巡逻",
    "defense": "战术防御",
    "ew": "电子压制",
}

# 中期检查版任务环境差异（单机近似；非最终场景参数）
MISSION_ENV_CONFIG = {
    "strike": {"max_steps": 240, "enemy_fire_interval_s": 8.0},
    "patrol": {"max_steps": 300, "enemy_fire_interval_s": 10.0},
    "defense": {"max_steps": 300, "enemy_fire_interval_s": 5.0},
    "ew": {"max_steps": 300, "enemy_fire_interval_s": 6.0},
}

# 阶段三 P0/P1 可选场景（opt-in；默认 30~80 km / 240 步不变）
SCENE_CONFIGS = {
    "default": {},
    "bvr80-120": {"scene_dist_min_m": 80e3, "scene_dist_max_m": 120e3,
                  "max_steps": 600},
}


def scene_config(name: str) -> dict:
    """返回场景覆盖配置（未知场景抛 ARG_INVALID）。"""
    if name not in SCENE_CONFIGS:
        raise ValidationError(ARG_INVALID, "未知训练场景",
                              detail={"scene": name, "valid": list(SCENE_CONFIGS)})
    return dict(SCENE_CONFIGS[name])


def validate_mission(mission: str) -> str:
    if mission not in MISSION_IDS:
        raise ValidationError(ARG_INVALID, "未知任务级专家智能体",
                              detail={"mission": mission, "valid": list(MISSION_IDS)})
    return mission


class ScaledObservation(gym.ObservationWrapper):
    """固定尺度观测层：2(x-lo)/(hi-lo)-1 → [-1,1]（设计文档 §4.2 方案 A）。

    仅用于策略网络输入；对外 get_action 仍接收原始 28 维物理量观测。
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        self._low, self._high = low, np.maximum(high, low + 1e-6)
        self.observation_space = spaces.Box(-np.ones_like(low), np.ones_like(high),
                                            dtype=np.float32)

    def observation(self, observation):
        return normalize_obs(observation, self._low, self._high)


def normalize_obs(obs, low=None, high=None) -> np.ndarray:
    """原始观测 → [-1,1]；默认使用 BVR 环境 OBS_LOW/OBS_HIGH。"""
    low = OBS_LOW if low is None else np.asarray(low, dtype=np.float32)
    high = OBS_HIGH if high is None else np.asarray(high, dtype=np.float32)
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    span = np.maximum(high - low, 1e-6)
    return np.clip(2.0 * (arr - low) / span - 1.0, -1.0, 1.0).astype(np.float32)


def wrap_observation(env) -> ScaledObservation:
    return ScaledObservation(env)


class MissionMixin:
    """任务标签 + 任务奖励塑形（不改变底层物理/终止）。"""

    def __init__(self, mission: str = "strike", config: dict | None = None):
        self.mission = validate_mission(str(mission))
        cfg = dict(MISSION_ENV_CONFIG[self.mission])
        if config:
            cfg.update(config)
        # 任务级训练/部署/行为树默认使用本机 AP 辅助模式；显式 raw_fcs 可覆盖。
        cfg.setdefault("control_mode", "assisted")
        super().__init__(config=cfg)

    def _build_obs(self):
        obs = super()._build_obs()
        label = MISSION_LABELS[self.mission]
        obs[26], obs[27] = label
        return np.clip(obs, OBS_LOW, OBS_HIGH).astype(np.float32)

    def _mission_reward(self, info: dict, obs) -> float:
        mission = self.mission
        extra = 0.0
        radar_state = info.get("radar_state")
        enemy_radar = info.get("enemy_radar_state")
        tracked = radar_state == "TRACK"
        if mission == "strike":
            return 0.0
        if mission == "patrol":
            if tracked:
                extra += 0.05
            if info.get("in_zone"):
                extra += 0.20
            if info.get("rwr_alarm"):
                extra -= 0.05
        elif mission == "defense":
            extra += 0.02
            if info.get("ecm_on"):
                extra += 0.05
            if enemy_radar == "TRACK":
                extra -= 0.10
            if info.get("rwr_alarm"):
                extra -= 0.05
            if info.get("maws_alarm"):
                extra -= 0.10
        elif mission == "ew":
            extra += 0.02
            if info.get("ecm_on"):
                extra += 0.10
                if enemy_radar != "TRACK":
                    extra += 0.10
            if not tracked:
                extra -= 0.02
        return float(np.clip(extra, -0.5, 0.5))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        extra = self._mission_reward(info, obs)
        total = float(reward) + extra
        info["mission_id"] = self.mission
        info["mission_reward"] = extra
        info["reward_total"] = total
        info["reward"] = total
        return obs, total, terminated, truncated, info


class TaskMissionEnv(MissionMixin, BVRCombatEnv):
    """Dict 动作部署环境（行为树主通道：to_env_action → step）。"""


class FlatTaskMissionEnv(MissionMixin, FlatActionBVRCombatEnv):
    """Box(7) 训练环境；动作空间收窄为 ActionV1 合法域。"""

    def __init__(self, mission: str = "strike", config: dict | None = None):
        super().__init__(mission=mission, config=config)
        self.action_space = spaces.Box(low=ACTION_V1_LOW.copy(), high=ACTION_V1_HIGH.copy(),
                                       shape=(7,), dtype=np.float32)


def make_task_env(mission: str = "strike", config: dict | None = None,
                  flat: bool = True, seed: int | None = None):
    """任务环境工厂（中期检查版）。flat=True 训练通道，False 部署 Dict 通道。"""
    env = (FlatTaskMissionEnv if flat else TaskMissionEnv)(mission=mission, config=config)
    if seed is not None:
        env.reset(seed=seed)
    return env
