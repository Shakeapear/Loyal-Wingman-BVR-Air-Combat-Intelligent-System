# -*- coding: utf-8 -*-
"""
agents/hybrid_policy.py
=======================
阶段三 P1：混合动作 Actor-Critic 与组合分布（连续 4 维 + 武器 3 维 Bernoulli）。

设计依据：`多任务智能体/统一输出接口与DRL网络设计.md` §4.3/§4.4/§4.8。

动作（ActionV1，shape=(7,)）：
    [throttle, roll, pitch, yaw, fire, switch, ecm]
    - 连续 4 维：z ~ N(mu, sigma)，a_cont = tanh(z)；
      throttle = (a_0 + 1) / 2 ∈ [0,1]；roll/pitch/yaw = a_1..3 ∈ [-1,1]；
      log_prob 含 tanh 雅可比修正；
    - 武器 3 维：Bernoulli(logits)，logits += (mask_weapon - 1) * 1e8；
      switch 掩码恒 False（单弹种）；ecm 由网络学习（0/1 边沿）；
      fire 由 compute_action_mask 决定。

掩码来源：`agents.policy_interface.compute_action_mask`（与 `_try_fire_own` 同源）。
SB3 接入：`HybridActorCriticPolicy(ActorCriticPolicy)`；训练环境用
`MaskedObsWrapper` 把 7 维掩码拼到观测后（28→35），策略内部拆分。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

from agents.policy_interface import ACTION_V1_DIM, compute_action_mask

CONT_DIM = 4
WEAPON_DIM = 3
MASK_SCALE = 1e8
EPS = 1e-6


def split_obs_mask(obs, obs_dim: int):
    """拆分 `[base_obs, mask]`；纯 28 维输入时使用保守默认掩码。"""
    tensor = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] == obs_dim:
        mask = torch.zeros(tensor.shape[:-1] + (ACTION_V1_DIM,),
                           dtype=torch.bool, device=tensor.device)
        mask[..., :4] = True
        mask[..., 4] = False          # 无 info 时保守禁用 fire
        mask[..., 5] = False          # switch 恒 False
        mask[..., 6] = True           # ecm 恒可用
        return tensor, mask
    if tensor.shape[-1] < obs_dim + WEAPON_DIM:
        raise ValueError(f"观测维度 {tensor.shape[-1]} 不匹配 obs_dim={obs_dim}+7")
    return tensor[..., :obs_dim], tensor[..., obs_dim:] > 0.5


class HybridActorCritic(nn.Module):
    """共享主干 + 连续头 + 武器头 + Critic 头。"""

    def __init__(self, obs_dim: int, hidden: tuple = (256, 128)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        h0, h1 = int(hidden[0]), int(hidden[1])
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, h0), nn.Tanh(),
            nn.Linear(h0, h1), nn.Tanh())
        self.mu = nn.Linear(h1, CONT_DIM)
        self.log_std = nn.Parameter(torch.zeros(CONT_DIM))
        self.weapon_logits = nn.Linear(h1, WEAPON_DIM)
        self.critic = nn.Linear(h1, 1)

    def forward(self, obs, mask=None):
        h = self.trunk(obs)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        logits = self.weapon_logits(h)
        if mask is not None:
            weapon_mask = mask[..., 4:].to(dtype=logits.dtype, device=logits.device)
            logits = logits + (weapon_mask - 1.0) * MASK_SCALE
        # switch 当前恒 False（单弹种，接口预留）；ecm 保留网络输出，
        # 由策略学习是否在某个决策步发送一次 0→1 边沿（不是恒 1）。
        logits = torch.cat([
            logits[..., 0:1],
            torch.full_like(logits[..., 1:2], -MASK_SCALE),
            logits[..., 2:3],
        ], dim=-1)
        value = self.critic(h).squeeze(-1)
        return mu, std, logits, value


class HybridDistribution(Distribution):
    """连续 Gaussian(tanh) × 武器 Bernoulli 的因子化分布。

    `weapon_mask` 为 3 维布尔（fire/switch/ecm）；False 维的 logits 被压到
    -1e8，采样恒为 0、熵为 0、对参数梯度为 0。
    """

    def __init__(self, mu, std, weapon_logits, weapon_mask=None):
        super().__init__()
        self.mu = mu
        self.std = std
        self.weapon_logits = weapon_logits
        if weapon_mask is None:
            weapon_mask = torch.ones_like(weapon_logits, dtype=torch.bool)
        self.weapon_mask = weapon_mask.bool()
        self.distribution = [
            torch.distributions.Normal(mu, std),
            torch.distributions.Bernoulli(logits=weapon_logits),
        ]
        self.normal = self.distribution[0]
        self.bernoulli = self.distribution[1]

    # ------------------------------------------------------------------
    # 采样/确定动作
    # ------------------------------------------------------------------
    @staticmethod
    def _to_action(z, weapon):
        cont = torch.tanh(z)
        throttle = (cont[..., 0:1] + 1.0) / 2.0
        return torch.cat([throttle, cont[..., 1:], weapon], dim=-1)

    def sample(self):
        z = self.normal.sample()
        weapon = self.bernoulli.sample()
        return self._to_action(z, weapon)

    def mode(self):
        weapon = (self.bernoulli.probs > 0.5).to(self.weapon_logits.dtype)
        return self._to_action(self.mu, weapon)

    # ------------------------------------------------------------------
    # 对数概率/熵
    # ------------------------------------------------------------------
    def log_prob(self, action):
        action = action.to(dtype=self.mu.dtype, device=self.mu.device)
        cont = action[..., :CONT_DIM].clone()
        cont[..., 0] = cont[..., 0] * 2.0 - 1.0        # throttle -> [-1,1]
        cont = cont.clamp(-1.0 + EPS, 1.0 - EPS)
        z = torch.atanh(cont)
        log_prob = self.normal.log_prob(z).sum(dim=-1)
        log_prob = log_prob - torch.log(1.0 - cont ** 2 + EPS).sum(dim=-1)
        log_prob = log_prob + self.bernoulli.log_prob(
            action[..., CONT_DIM:]).sum(dim=-1)
        return log_prob

    def entropy(self):
        cont_ent = self.normal.entropy().sum(dim=-1)
        weapon_ent = self.bernoulli.entropy() * self.weapon_mask.to(
            dtype=self.weapon_logits.dtype)
        return cont_ent + weapon_ent.sum(dim=-1)

    # ------------------------------------------------------------------
    # SB3 风格辅助接口
    # ------------------------------------------------------------------
    @staticmethod
    def proba_distribution_net(latent_dim: int):
        return nn.Linear(latent_dim, CONT_DIM), nn.Parameter(torch.zeros(CONT_DIM))

    def proba_distribution(self, mu, std, weapon_logits, weapon_mask=None):
        self.mu, self.std, self.weapon_logits = mu, std, weapon_logits
        if weapon_mask is not None:
            self.weapon_mask = weapon_mask.bool()
        self.normal = torch.distributions.Normal(mu, std)
        self.bernoulli = torch.distributions.Bernoulli(logits=weapon_logits)
        self.distribution = [self.normal, self.bernoulli]
        return self

    def actions_from_params(self, mu, std, weapon_logits, weapon_mask=None,
                            deterministic: bool = False):
        self.proba_distribution(mu, std, weapon_logits, weapon_mask)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mu, std, weapon_logits, weapon_mask=None):
        actions = self.actions_from_params(mu, std, weapon_logits, weapon_mask)
        return actions, self.log_prob(actions)


class HybridActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy 的混合动作实现（其余 PPO/GAE/buffer 复用 SB3）。"""

    def _build(self, lr_schedule) -> None:
        obs_dim = int(np.prod(self.observation_space.shape)) - ACTION_V1_DIM
        if obs_dim <= 0:
            raise ValueError("HybridActorCriticPolicy 需要 obs 维度 = base_obs + 7 掩码")
        self.obs_dim = obs_dim
        self.hybrid = HybridActorCritic(obs_dim)
        if self.ortho_init:
            for module in self.hybrid.modules():
                if isinstance(module, nn.Linear):
                    self.init_weights(module, gain=np.sqrt(2))
        self.optimizer = self.optimizer_class(
            self.hybrid.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    # ------------------------------------------------------------------
    def _forward_hybrid(self, obs):
        base, mask = split_obs_mask(obs, self.obs_dim)
        mu, std, logits, value = self.hybrid(base, mask)
        dist = HybridDistribution(mu, std, logits, mask[..., 4:])
        return dist, value

    def get_distribution(self, obs) -> HybridDistribution:
        dist, _ = self._forward_hybrid(obs)
        return dist

    def forward(self, obs, deterministic: bool = False):
        dist, value = self._forward_hybrid(obs)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return (actions.reshape((-1, *self.action_space.shape)),
                value.reshape(-1, 1), log_prob)

    def evaluate_actions(self, obs, actions):
        dist, value = self._forward_hybrid(obs)
        log_prob = dist.log_prob(actions.reshape(-1, ACTION_V1_DIM))
        entropy = dist.entropy()
        return value.reshape(-1, 1), log_prob, entropy

    def predict_values(self, obs):
        _, value = self._forward_hybrid(obs)
        return value.reshape(-1, 1)

    def _get_action_dist_from_latent(self, latent_pi):
        # 保留基类接口；混合策略不使用 mlp_extractor latent
        raise NotImplementedError("HybridActorCriticPolicy 使用 get_distribution")


def masked_obs(obs, info, env_cfg=None) -> np.ndarray:
    """原始 28 维 obs + info → 35 维 `[obs, mask]`（推理侧）。"""
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    mask = compute_action_mask(obs_arr, info, env_cfg).astype(np.float32)
    return np.concatenate([obs_arr, mask], axis=0).astype(np.float32)


class MaskedObsWrapper(gym.ObservationWrapper):
    """把 ActionMaskV1 拼到观测末尾（28→35），供 SB3 训练/推理使用。

    - 基础环境观测契约（28 维）不变；仅策略输入变为 35 维；
    - 每步在包装层调用 `compute_action_mask(obs, info, env_cfg)`，训练与推理同源；
    - `observation_space = Box([base_low, 0...], [base_high, 1...])`。
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        self._base_dim = int(low.size)
        self.observation_space = spaces.Box(
            low=np.concatenate([low, np.zeros(ACTION_V1_DIM, dtype=np.float32)]),
            high=np.concatenate([high, np.ones(ACTION_V1_DIM, dtype=np.float32)]),
            dtype=np.float32)
        self.env_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
        self._last_info: dict = {}

    def _with_mask(self, obs) -> np.ndarray:
        base = np.asarray(obs, dtype=np.float32).reshape(self._base_dim)
        # 掩码阈值（距离/弹量等）必须使用尺度化前的物理观测；ScaledObservation
        # 会保存 _last_raw_obs。若上游没有该属性，才退回当前观测（info 完整时
        # compute_action_mask 仍优先使用 info，不受影响）。
        raw = getattr(self.env, "_last_raw_obs", None)
        mask_obs = raw if raw is not None else base
        mask = compute_action_mask(mask_obs, self._last_info, self.env_cfg)
        return np.concatenate([base, mask.astype(np.float32)], axis=0)

    def observation(self, observation):
        return self._with_mask(observation)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_info = dict(info or {})
        return self._with_mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = dict(info or {})
        return self._with_mask(obs), reward, terminated, truncated, info


__all__ = ["HybridActorCritic", "HybridDistribution", "HybridActorCriticPolicy",
           "MaskedObsWrapper", "split_obs_mask", "masked_obs", "MASK_SCALE"]
