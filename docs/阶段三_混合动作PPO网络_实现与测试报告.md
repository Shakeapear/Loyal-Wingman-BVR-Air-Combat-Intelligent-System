# 阶段三 P1 混合动作 PPO 网络 实现与测试报告

> 日期：2026-11-01　｜　范围：P1 网络与接口最小可用版（混合动作、掩码、SB3 接入、短训练）。
> 状态：**🟡 网络实现 + 2048 步冒烟通过；>60% 命中率与长训练未验收。**
> 设计依据：`多任务智能体/统一输出接口与DRL网络设计.md` §3.3/§4.3/§4.4/§4.8/§4.9。

## 1. 结论

- 新增 `HybridActorCritic` / `HybridDistribution` / `HybridActorCriticPolicy`：
  连续 4 维 Gaussian+tanh、武器 3 维 Bernoulli、动作掩码进入分布；
- 新增 `MaskedObsWrapper`：基础 28 维观测契约不变，策略输入 28+7=35 维；
- 新增 `agents/ppo_agent.py`：SB3 PPO 训练/保存/加载/推理入口，hybrid 为 opt-in；
- 新增 14 项单测，覆盖形状/界、log_prob/熵解析对拍、掩码梯度、`_try_fire_own` 对拍、
  SB3 predict/evaluate、2048 步 learn、save/load；
- 2048 步冒烟（2 envs、seed 0）通过：`ep_rew_mean=304.40`、`mean_step_reward=2.856`、
  `policy_gradient_loss=-9.69e-4`、`value_loss=8546.59`、`entropy_loss=-5.704`、
  `approx_kl=0.00398`，所有日志有限；保存后可加载并确定性复现动作。

## 2. 环境与依赖

| 项 | 值 |
|----|----|
| Python | 3.12（conda `DC`） |
| torch | 2.13.0+cpu |
| stable-baselines3 | 2.9.0 |
| gymnasium | 1.3.0 |
| 训练环境 | `FlatTaskMissionEnv`（strike，`control_mode=assisted`），ActionV1 `Box(7)` |

未升级/降级依赖，未使用 GPU。

## 3. 网络结构

`多任务智能体/agents/hybrid_policy.py::HybridActorCritic`

```
obs_base(28) → FC(28→256) + Tanh → FC(256→128) + Tanh
   ├─ 连续头 FC(128→4) → μ；log_std ∈ R^4（state-independent）
   │     z ~ N(μ, σ)；a_cont = tanh(z)
   │     throttle=(a0+1)/2，roll/pitch/yaw=a1..3
   ├─ 武器头 FC(128→3) → logits（fire/switch/ecm）
   │     logits += (mask_weapon - 1) * 1e8
   │     switch 固定 -1e8（恒 0）；ecm 固定 +1e8（恒 1）
   └─ Critic FC(128→1) → V(s)
```

组合分布：

```
π(a|s) = N(z; μ, σ) · Π_i Bernoulli(w_i; p_i, mask_i)
log π = Σ log N(z; μ, σ) − Σ log(1 − tanh²(z)) + Σ log Bernoulli(w; p, mask)
H = H_N(μ,σ) + Σ mask_i · H_B(p_i)
```

- 连续熵使用 `Normal(μ,σ).entropy()`；tanh 雅可比项为 `−log(1−a²+ε)`；
- 掩码为 False 的武器维 logits≈−1e8，采样恒 0、熵 0、对参数梯度为 0；
- `ecm` 动作恒 1；`switch` 恒 0（单弹种，接口预留）。

## 4. 动作掩码

掩码唯一来源：`agents.policy_interface.compute_action_mask(obs, info, env_cfg)`，与
`BVRCombatEnv._try_fire_own` 的发射条件对拍（单测）：

| 维度 | 条件 |
|------|------|
| 0–3 | 恒 True |
| fire | `own_missiles_left > 0` ∧ `radar_state == "TRACK"` ∧ `r_min_own ≤ dist ≤ r_max_own` |
| switch | 恒 False |
| ecm | 恒 True（动作恒定 1） |

训练/推理一致性：

- `MaskedObsWrapper` 在每步 `reset/step` 用当前 `info` 计算掩码并拼到观测末尾（28→35）；
- SB3 rollout 与 PPO update 都读取同一份 35 维观测，掩码随轨迹进入 buffer，
  因此 `evaluate_actions` 与采样使用的掩码严格一致；
- 推理 `HybridPPOAgent.get_action(obs, info=...)` 用同一函数计算掩码，拼成 35 维后
  `predict`，并在返回前再次 `apply_action_mask`；
- 28 维裸观测推理时默认保守禁用 fire（避免无 info 误发射）。

## 5. SB3 接入

`HybridActorCriticPolicy(ActorCriticPolicy)` 覆写：

| 方法 | 行为 |
|------|------|
| `_build` | 构建 `HybridActorCritic(obs_dim=obs_dim-7)` 与 optimizer；不使用 mlp_extractor |
| `forward` | 拆分 obs/mask → 组合分布采样 → `(action, value, log_prob)` |
| `get_distribution` | 返回 `HybridDistribution` |
| `evaluate_actions` | 返回 `(value, log_prob, entropy)`，供 PPO 裁剪/GAE 复用 |
| `predict_values` | 仅计算 V(s)，供 GAE 末步 bootstrap |
| `_predict` | 继承基类，经 `get_distribution` 采样/取 mode |

GAE、clip、buffer、logger 均复用 SB3；仅策略网络/分布为自定义实现。

## 6. 训练入口

`多任务智能体/agents/ppo_agent.py::HybridPPOAgent`

- 环境链：`FlatTaskMissionEnv → ScaledObservation → MaskedObsWrapper`；
- CLI 支持 `--task/--policy/--timesteps/--n-envs/--seed/--out-dir`；
- 可选：`--resume`、`--checkpoint-freq`、`--tensorboard/--tb-log`；
- 默认 `policy=hybrid`；`--policy mlp` 保留对照通道；
- 训练后自动加载模型做确定性评估，检查动作有限、连续维界、武器掩码；
- 保存 `models/hybrid_smoke/hybrid_ppo_final.zip` + `.spec.json` + `train_summary.json`（`models/` 不入库）。

## 7. 测试

新增 `多任务智能体/tests/test_hybrid_policy.py`：**14 passed**（约 12 s）。

| 覆盖项 | 断言 |
|--------|------|
| 形状/界 | throttle∈[0,1]，roll/pitch/yaw∈[-1,1]，武器∈{0,1}，switch=0，ecm=1 |
| log_prob 对拍 | 与手写 Normal+Bernoulli+tanh 雅可比公式一致（atol 1e-5） |
| 熵对拍 | 与手写公式一致；掩码维熵贡献=0 |
| 非法 fire | logits≤−1e7，p<1e-6，采样不选，参数梯度≈0 |
| 掩码对拍 | 无弹/未跟踪/低于 r_min/高于 r_max/合法发射 5 种边界与 `_try_fire_own` 一致 |
| SB3 | `predict` 形状/界；`evaluate_actions` 返回 value/log_prob/entropy |
| 训练/存取 | `learn(2048)` 不报错、loss 有限；save/load 后 deterministic 动作一致 |
| PolicyOutput | `source="drl"`、`masked` 正确、`action_version="action_v1"` |

三套全量回归：**169 passed / 0 failed**（多任务 94 + 行为树 52 + 可视化 23）。

## 8. 2048 步冒烟结果

命令（`多任务智能体/`）：

```powershell
& $PY -X utf8 -m agents.ppo_agent --task strike --policy hybrid `
    --timesteps 2048 --n-envs 2 --seed 0 --out-dir models/hybrid_smoke
```

| 项 | 值 |
|----|----|
| 时间 | 2026-11-01 |
| 总步数 | 2048（2 envs × n_steps 128 × 8 次更新） |
| wall time | 37.755 s |
| `rollout/ep_rew_mean` | 304.397 |
| `train/mean_step_reward` | 2.8559 |
| `train/policy_gradient_loss` | −0.000969 |
| `train/value_loss` | 8546.59 |
| `train/entropy_loss` | −5.7041 |
| `train/approx_kl` | 0.00398 |
| `all_finite` | True |
| 评估（16 步 deterministic） | 平均奖励 2.5088，动作有限、在界内 |
| 保存/加载 | `hybrid_ppo_final.zip` 可加载并 predict |
| 模型路径 | `多任务智能体/models/hybrid_smoke/hybrid_ppo_final.zip`（`.gitignore`，不入库） |
| 归档证据 | `docs/assets/stage3_p1_hybrid/train_summary.json`、`hybrid_ppo_final.spec.json`（不含模型权重） |

> 该结果仅证明“网络能训、动作合法、掩码正确、可复现”，不代表命中率或收敛质量；
> `value_loss` 在 2048 步下仍较大，属短训练预期。

## 9. 已知限制

1. **未做长训练/命中率验收**：P1 的 >60% 命中率需服务器长训练与固定测试集，本轮未执行。
2. **单任务冒烟**：仅 strike；patrol/defense/ew 未跑 hybrid 训练（接口同一）。
3. **观测扩展**：MaskedObsWrapper 将策略输入从 28 扩到 35；基础环境观测契约未变，
   但旧 MLP 模型不能直接复用 35 维输入（需重建/加 wrapper）。
4. **ecm/switch 为确定维**：ecm 恒 1、switch 恒 0，不参与策略梯度；后续多弹种/ECM 策略化时放开。
5. **未实现 LSTM/frame_stack**：部分可观测记忆能力留 P2。
6. **未接入 JSRL**：`guide_action/guide_steps` 推理透传已保留，课程衰减与训练循环未实现。
7. **训练吞吐/稳定性**：CPU + DummyVecEnv，2 envs 冒烟约 37.8 s；长训练需并行环境与监控。

## 10. 下一步

1. **P0 基线**：固定测试集 + MlpPolicy 纯 DRL 对照（<30% 命中率基线）。
2. **P1 长训练**：固定 20 组态势评估 hybrid 命中率，目标 >60%；补 reward/loss/引导曲线。
3. **P2**：`jsrl_trainer.py`（GuideInterface 课程衰减 + 内在奖励）、LSTM/frame_stack、多机环境。
4. 将 `HybridPPOAgent` 接入 `expert_registry` 的 opt-in 通道（当前可直接 `get_action`，未改默认注册表）。
