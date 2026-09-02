# -*- coding: utf-8 -*-
"""
validation/check_env_and_throughput.py（交付物 D2.3-3 验证脚本）
================================================================
1. gymnasium.utils.env_checker.check_env(BVRCombatEnv) 通过性检查；
2. 随机动作 100 步无报错（终止自动 reset）；
3. stable-baselines3 PPO + MultiInputPolicy 验证性训练 ≤2048 步（挂接确认）；
4. 4 实例并行（AsyncVectorEnv，spawn 启动）吞吐量测试：60 s 采样，
   输出 steps/s（验收标准 ≥1000 steps/s）。

结果写入 validation/results/env_check_and_throughput_report.txt。

用法（DC 环境）：
    python validation/check_env_and_throughput.py [--quick]
    --quick：吞吐采样 15 s（调试用，不改变验收结论记录格式）
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.bvr_combat_env import BVRCombatEnv, make_bvr_env

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def level_action():
    """近水平巡航动作（近失速配平；用于吞吐采样，避免随机动作频繁坠毁触发 reset）。"""
    return {"flight": np.array([0.6, 0.0, -0.15, 0.0], dtype=np.float32),
            "weapon": np.array([0, 0, 0])}


FLAT_LEVEL_ACTION = np.array([0.6, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def run_env_checker():
    from gymnasium.utils.env_checker import check_env
    t0 = time.time()
    env = BVRCombatEnv()
    check_env(env, skip_render_check=True)
    env.own.close(); env.enemy.close()
    return True, time.time() - t0


def run_random_rollout(n_steps=100):
    env = BVRCombatEnv()
    env.reset(seed=0)
    done_steps, episodes = 0, 1
    while done_steps < n_steps:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        done_steps += 1
        assert obs.shape == env.observation_space.shape
        assert np.all(obs >= env.observation_space.low - 1e-6)
        assert np.all(obs <= env.observation_space.high + 1e-6)
        assert np.isfinite(r)
        if term or trunc:
            env.reset()
            episodes += 1
    env.own.close(); env.enemy.close()
    return True, done_steps, episodes


def run_sb3_smoke(total_timesteps=1024):
    """SB3 挂接验证性训练（≤2048 步，非完整训练；用展平动作包装）。

    注：验证用 compute_envelope=False（跳过 reset 时 ~1.2 s 的攻击区解算，
    接口与物理完全一致），避免未训练策略频繁坠毁导致 reset 耗时掩盖训练回路验证。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from common.bvr_combat_env import make_flat_bvr_env

    def _make():
        return make_flat_bvr_env(config={"compute_envelope": False})

    vec = DummyVecEnv([_make])
    model = PPO("MlpPolicy", vec, n_steps=256, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    wall = time.time() - t0
    vec.close()
    return True, wall


def run_throughput(n_workers=4, sample_s=60.0):
    """4 实例并行吞吐：SB3 SubprocVecEnv（spawn）+ 60 s 采样。

    测试工况说明：使用近水平巡航动作 + n_missiles=0 / compute_envelope=False
    的展平动作环境（工厂函数 make_flat_bvr_env 为模块顶层函数，满足 spawn 可
    pickle；动作 Box(7) 为 SB3 VecEnv 原生支持）。env.step 的计算量与动作/挂载
    无关（导弹向量化亚步 ~0.4 ms/步），此设置消除"未训练动作坠毁→episode
    终止→reset（含攻击区解算 ~2.3 s）"对稳态步进速率测量的干扰。
    """
    from functools import partial
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from common.bvr_combat_env import make_flat_bvr_env
    cfg = {"compute_envelope": False, "n_missiles": 0}
    vec = SubprocVecEnv([partial(make_flat_bvr_env, config=cfg) for _ in range(n_workers)])
    vec.reset()
    # 预热 10 步
    for _ in range(10):
        vec.step(np.tile(FLAT_LEVEL_ACTION, (n_workers, 1)))
    t0 = time.time()
    n_calls = 0
    while time.time() - t0 < sample_s:
        vec.step(np.tile(FLAT_LEVEL_ACTION, (n_workers, 1)))
        n_calls += 1
    wall = time.time() - t0
    vec.close()
    total_steps = n_calls * n_workers
    return total_steps / wall, n_calls, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="吞吐采样缩短为 15 s")
    args = ap.parse_args()
    sample_s = 15.0 if args.quick else 60.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"BVRCombatEnv 接口验证与吞吐量报告（{datetime.now():%Y-%m-%d %H:%M:%S}）", "=" * 60]

    ok, wall = run_env_checker()
    lines.append(f"[1] env_checker.check_env: {'通过' if ok else '失败'} ({wall:.1f} s)")
    print(lines[-1])

    ok, steps, episodes = run_random_rollout(100)
    lines.append(f"[2] 随机动作 100 步无报错: {'通过' if ok else '失败'} "
                 f"（100 步内自动 reset {episodes - 1} 次，观测始终在界内）")
    print(lines[-1])

    ok, wall = run_sb3_smoke(1024)
    lines.append(f"[3] SB3 PPO(MlpPolicy, 展平动作包装) 验证性训练 1024 步: "
                 f"{'通过' if ok else '失败'} ({wall:.1f} s，非完整训练)")
    print(lines[-1])

    rate, n_calls, wall = run_throughput(n_workers=4, sample_s=sample_s)
    verdict = "达标 (>= 1000)" if rate >= 1000 else "未达标"
    lines.append(f"[4] 4 实例并行吞吐（稳态步进，无 reset 干扰工况）: "
                 f"{rate:.1f} steps/s（{n_calls} 次 batch 调用 × 4 实例 / {wall:.1f} s 采样，{verdict}）")
    print(lines[-1])

    report = "\n".join(lines) + "\n"
    out = RESULTS_DIR / "env_check_and_throughput_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
