# -*- coding: utf-8 -*-
"""
多任务智能体/tests/test_hybrid_policy.py（阶段三 P1）
======================================================
覆盖：混合动作网络/组合分布数学、掩码一致性、SB3 接入、短训练与保存加载。

运行：python -m pytest tests/test_hybrid_policy.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli, Normal

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.hybrid_policy import (HybridActorCritic, HybridActorCriticPolicy,
                                  HybridDistribution, MaskedObsWrapper,
                                  masked_obs, split_obs_mask)
from agents.policy_interface import (ACTION_V1_DIM, compute_action_mask)
from agents.ppo_agent import HybridPPOAgent, make_masked_env
from common.bvr_combat_env import BVRCombatEnv

torch.set_num_threads(1)


def _obs(batch=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 28, generator=gen)


def _mask(batch=8, fire=True):
    mask = torch.ones(batch, 7, dtype=torch.bool)
    mask[:, 4] = fire
    mask[:, 5] = False
    mask[:, 6] = True
    return mask


class TestHybridNetwork:
    def test_shapes_and_action_bounds(self):
        net = HybridActorCritic(28)
        mu, std, logits, value = net(_obs(64), _mask(64, fire=True))
        assert mu.shape == (64, 4) and std.shape == (64, 4)
        assert logits.shape == (64, 3) and value.shape == (64,)
        dist = HybridDistribution(mu, std, logits, _mask(64)[:, 4:])
        actions = dist.sample()
        assert actions.shape == (64, ACTION_V1_DIM)
        assert torch.all(actions[:, 0] >= 0) and torch.all(actions[:, 0] <= 1)
        assert torch.all(actions[:, 1:4].abs() <= 1)
        assert set(actions[:, 4:].unique().tolist()) <= {0.0, 1.0}
        assert torch.all(actions[:, 5] == 0)          # switch 恒 False
        assert torch.all(actions[:, 6] == 1)          # ecm 恒 True

    def test_log_prob_matches_analytic_formula(self):
        net = HybridActorCritic(28)
        mask = _mask(16, fire=True)
        mu, std, logits, _ = net(_obs(16), mask)
        dist = HybridDistribution(mu, std, logits, mask[:, 4:])
        action = dist.sample()

        cont = action[:, :4].clone()
        cont[:, 0] = cont[:, 0] * 2 - 1
        z = torch.atanh(cont.clamp(-1 + 1e-6, 1 - 1e-6))
        expected = (Normal(mu, std).log_prob(z).sum(-1)
                    - torch.log(1 - cont ** 2 + 1e-6).sum(-1)
                    + Bernoulli(logits=logits).log_prob(action[:, 4:]).sum(-1))
        assert torch.allclose(dist.log_prob(action), expected, atol=1e-5)

    def test_entropy_matches_formula_and_masked_is_zero(self):
        net = HybridActorCritic(28)
        mu, std, logits, _ = net(_obs(8), _mask(8, fire=True))
        mask = _mask(8, fire=True)
        dist = HybridDistribution(mu, std, logits, mask[:, 4:])
        expected = (Normal(mu, std).entropy().sum(-1)
                    + (Bernoulli(logits=logits).entropy()
                       * mask[:, 4:].float()).sum(-1))
        assert torch.allclose(dist.entropy(), expected, atol=1e-5)

        no_weapon = torch.zeros(8, 3, dtype=torch.bool)
        dist_nw = HybridDistribution(mu, std, logits, no_weapon)
        assert torch.allclose(dist_nw.entropy(),
                              Normal(mu, std).entropy().sum(-1), atol=1e-5)

    def test_illegal_fire_probability_and_gradient_zero(self):
        net = HybridActorCritic(28)
        mask = _mask(64, fire=False)
        mu, std, logits, _ = net(_obs(64, seed=1), mask)
        assert torch.all(logits[:, 0] <= -1e7)
        dist = HybridDistribution(mu, std, logits, mask[:, 4:])
        assert float(dist.bernoulli.probs[:, 0].max().detach()) < 1e-6
        samples = dist.sample()
        assert torch.all(samples[:, 4] == 0)

        net.zero_grad()
        loss = dist.log_prob(samples).sum()
        loss.backward()
        assert net.weapon_logits.weight.grad[0].abs().max() < 1e-6
        assert net.weapon_logits.bias.grad[0].abs() < 1e-6

    def test_deterministic_mode(self):
        net = HybridActorCritic(28)
        mask = _mask(8, fire=True)
        mu, std, logits, _ = net(_obs(8), mask)
        dist = HybridDistribution(mu, std, logits, mask[:, 4:])
        mode = dist.mode()
        assert torch.all(mode[:, 5] == 0) and torch.all(mode[:, 6] == 1)
        assert torch.equal(mode, dist.get_actions(deterministic=True))

    def test_split_obs_mask(self):
        obs = torch.randn(3, 35)
        obs[:, 28:] = torch.tensor([1, 1, 1, 1, 0, 0, 1], dtype=torch.float32)
        base, mask = split_obs_mask(obs, 28)
        assert base.shape == (3, 28) and mask.shape == (3, 7)
        assert torch.all(mask[:, :4]) and not mask[:, 4].any()
        base2, mask2 = split_obs_mask(torch.randn(3, 28), 28)
        assert base2.shape == (3, 28) and not mask2[:, 4].any()


class TestMaskParity:
    @staticmethod
    def _env():
        env = BVRCombatEnv(config={"compute_envelope": False, "max_steps": 2})
        obs, _ = env.reset(seed=3)
        return env, obs

    def _info(self, env):
        frame = env.get_viz_frame()
        return {"own_missiles_left": int(env.own_n_left),
                "radar_state": env.radar.state("enemy"),
                "dist_m": float(frame["dist_m"]),
                "r_min_own_m": float(env.r_min_own),
                "r_max_own_m": float(env.r_max_own)}

    def _compare(self, env, obs):
        info = self._info(env)
        mask = compute_action_mask(obs, info, env.cfg)
        fired = env._try_fire_own()
        assert bool(mask[4]) is bool(fired)
        return mask, fired

    def test_parity_no_missiles(self, monkeypatch):
        env, obs = self._env()
        try:
            monkeypatch.setattr(env.radar, "state", lambda _: "TRACK")
            monkeypatch.setattr(env, "_launch_missile", lambda *a, **k: None)
            env.own_n_left = 0
            self._compare(env, obs)
            assert not compute_action_mask(obs, self._info(env), env.cfg)[4]
        finally:
            env.close()

    def test_parity_not_tracked(self, monkeypatch):
        env, obs = self._env()
        try:
            monkeypatch.setattr(env.radar, "state", lambda _: "SEARCH")
            env.own_n_left = 2
            self._compare(env, obs)
        finally:
            env.close()

    def test_parity_out_of_envelope(self, monkeypatch):
        env, obs = self._env()
        try:
            monkeypatch.setattr(env.radar, "state", lambda _: "TRACK")
            monkeypatch.setattr(env, "_launch_missile", lambda *a, **k: None)
            env.own_n_left = 2
            dist = env.get_viz_frame()["dist_m"]
            env.r_min_own, env.r_max_own = dist + 1000.0, dist + 2000.0
            self._compare(env, obs)
            env.own_n_left = 2
            env.r_min_own, env.r_max_own = dist - 2000.0, dist - 1000.0
            self._compare(env, obs)
        finally:
            env.close()

    def test_parity_valid_shot(self, monkeypatch):
        env, obs = self._env()
        try:
            monkeypatch.setattr(env.radar, "state", lambda _: "TRACK")
            monkeypatch.setattr(env, "_launch_missile", lambda *a, **k: None)
            env.own_n_left = 2
            dist = env.get_viz_frame()["dist_m"]
            env.r_min_own, env.r_max_own = dist - 1000.0, dist + 1000.0
            mask, fired = self._compare(env, obs)
            assert mask[4] and fired
        finally:
            env.close()

    def test_masked_obs_and_wrapper(self):
        env = make_masked_env("strike", {"compute_envelope": False, "max_steps": 2})
        try:
            obs, info = env.reset(seed=1)
            assert obs.shape == (35,)
            assert np.allclose(obs[28:], compute_action_mask(obs[:28], info,
                                                             env.env_cfg).astype(np.float32))
            assert obs[33] == 0 and obs[34] == 1
        finally:
            env.close()


@pytest.fixture(scope="class")
def masked_env():
    env = make_masked_env("strike", {"compute_envelope": False, "max_steps": 20})
    yield env
    env.close()


class TestSB3Integration:
    def test_predict_and_evaluate_actions(self, masked_env):
        from stable_baselines3 import PPO
        model = PPO(HybridActorCriticPolicy, masked_env, n_steps=32, batch_size=32,
                    n_epochs=1, seed=0, verbose=0)
        obs = masked_env.reset(seed=0)[0]
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (7,)
        assert 0 <= action[0] <= 1 and np.all(np.abs(action[1:4]) <= 1)
        assert action[4] == 0 or obs[32] > 0.5      # fire 不越过掩码
        assert action[5] == 0 and action[6] == 1

        obs_t = torch.as_tensor(obs).float().unsqueeze(0)
        act_t = torch.as_tensor(action).float().unsqueeze(0)
        value, log_prob, entropy = model.policy.evaluate_actions(obs_t, act_t)
        assert value.shape == (1, 1) and log_prob.shape == (1,) and entropy.shape == (1,)
        assert torch.all(torch.isfinite(value)) and torch.all(torch.isfinite(log_prob))
        assert torch.all(torch.isfinite(entropy))

    def test_learn_save_load_deterministic(self, masked_env, tmp_path):
        from stable_baselines3 import PPO
        model = PPO(HybridActorCriticPolicy, masked_env, n_steps=64, batch_size=32,
                    n_epochs=1, seed=0, verbose=0)
        model.learn(total_timesteps=2048)
        values = [float(v) for v in model.logger.name_to_value.values()
                  if np.isscalar(v)]
        assert values and all(np.isfinite(v) for v in values)

        path = tmp_path / "hybrid"
        model.save(str(path))
        loaded = PPO.load(str(path) + ".zip", env=masked_env, device="cpu")
        obs = masked_env.reset(seed=7)[0]
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        assert np.allclose(a1, a2)
        assert a2.shape == (7,)

    def test_agent_get_action_policy_output(self, masked_env):
        from stable_baselines3 import PPO
        agent = HybridPPOAgent(task="strike", seed=0)
        agent.env = masked_env
        agent.train_env = masked_env
        agent.model = PPO(HybridActorCriticPolicy, masked_env, n_steps=16, batch_size=16,
                          n_epochs=1, seed=0, verbose=0)
        obs = masked_env.reset(seed=2)[0]
        out = agent.get_action(obs[:28], info={"own_missiles_left": 2,
                                               "radar_state": "SEARCH",
                                               "dist_m": 40000.0,
                                               "r_min_own_m": 1000.0,
                                               "r_max_own_m": 50000.0})
        assert out.action.shape == (7,) and out.source == "drl"
        assert out.action_version == "action_v1"
        assert out.masked is not None and out.masked.shape == (7,)
        assert out.action[4] == 0                     # 未 TRACK：fire 禁用
        assert out.action[5] == 0 and out.action[6] == 1
        out2 = agent.get_action(obs[:28], info={"own_missiles_left": 2,
                                                "radar_state": "TRACK",
                                                "dist_m": 40000.0,
                                                "r_min_own_m": 1000.0,
                                                "r_max_own_m": 50000.0},
                                deterministic=False)
        assert out2.action.shape == (7,)
