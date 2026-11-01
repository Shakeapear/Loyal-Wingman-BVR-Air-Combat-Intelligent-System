# 阶段三 P1 混合动作 PPO 网络 命令清单

> 环境：Windows PowerShell 7；conda `DC`（Python 3.12，torch 2.13.0+cpu，SB3 2.9.0，gymnasium 1.3.0）。
> 工作目录如无特别说明为仓库根 `C:\Users\36266\Desktop\大创 开发`。

## 0. 环境变量

```powershell
$PY = "C:\Users\36266\.conda\envs\DC\python.exe"
```

## 1. 单元测试

```powershell
# P1 新模块（14 项）
& $PY -m pytest "多任务智能体\tests\test_hybrid_policy.py" -q

# 多任务智能体全套
& $PY -m pytest "多任务智能体\tests" -q

# 三套全量（多任务 94 + 行为树 52 + 可视化 23 = 169 项）
& $PY -m pytest "多任务智能体\tests" "引导策略行为树\tests" "可视化工具\tests" -q
```

预期：`14 passed`、`169 passed / 0 failed`。

## 2. 2048 步冒烟训练（hybrid）

```powershell
Set-Location -LiteralPath "多任务智能体"
& $PY -X utf8 -m agents.ppo_agent --task strike --policy hybrid `
    --timesteps 2048 --n-envs 2 --seed 0 --out-dir models/hybrid_smoke
```

预期输出（2026-11-01 实测）：

```text
[完成] hybrid task=strike steps=2048 wall≈37.8s
  rollout/ep_rew_mean: ≈304.40
  train/policy_gradient_loss: ≈-9.69e-4
  train/value_loss: ≈8546.6
  train/entropy_loss: ≈-5.704
  train/approx_kl: ≈0.00398
[评估] finite=True in_bounds=True mean_reward≈2.509
[模型] models\hybrid_smoke\hybrid_ppo_final.zip
```

产物（`多任务智能体/models/` 已 `.gitignore`，不入库）：

- `hybrid_ppo_final.zip`
- `hybrid_ppo_final.spec.json`
- `train_summary.json`

## 3. 可选：TensorBoard / Checkpoint / Resume

```powershell
Set-Location -LiteralPath "多任务智能体"

# TensorBoard（日志目录不入库，确认 .gitignore 后使用）
& $PY -X utf8 -m agents.ppo_agent --task strike --policy hybrid `
    --timesteps 2048 --n-envs 2 --seed 0 --out-dir models/hybrid_smoke `
    --tensorboard --tb-log tensorboard/hybrid_smoke

# 每 1024 环境步保存 checkpoint
& $PY -X utf8 -m agents.ppo_agent --task strike --policy hybrid `
    --timesteps 2048 --n-envs 2 --seed 0 --out-dir models/hybrid_smoke_ckpt `
    --checkpoint-freq 1024

# 从已有模型继续训练
& $PY -X utf8 -m agents.ppo_agent --task strike --policy hybrid `
    --timesteps 1024 --n-envs 2 --seed 0 --out-dir models/hybrid_resume `
    --resume models/hybrid_smoke/hybrid_ppo_final.zip
```

说明：`--checkpoint-freq` 以环境步计；内部按 `freq // n_envs` 转为 SB3 回调步数。

## 4. 兼容 MlpPolicy 对照

```powershell
Set-Location -LiteralPath "多任务智能体"
& $PY -X utf8 -m agents.ppo_agent --task strike --policy mlp `
    --timesteps 2048 --n-envs 2 --seed 0 --out-dir models/mlp_smoke
```

说明：`agents/task_agents` 原有 MlpPolicy 训练路径未改；本命令为 35 维掩码观测的
MLP 对照，主要用于确认入口不回归。

## 5. 加载模型并推理

```python
import sys
sys.path.insert(0, r"多任务智能体")

from agents.ppo_agent import HybridPPOAgent

agent = HybridPPOAgent(task="strike", seed=0)
agent.load(r"多任务智能体/models/hybrid_smoke/hybrid_ppo_final.zip")
out = agent.get_action(
    obs,                      # 原始 28 维 BVR 观测
    info={"own_missiles_left": 2, "radar_state": "TRACK",
          "dist_m": 40000.0, "r_min_own_m": 1000.0, "r_max_own_m": 50000.0},
    deterministic=True,
)
print(out.action, out.masked, out.source, out.action_version)
```

## 6. 快速检查掩码

```python
import numpy as np
from agents.policy_interface import compute_action_mask

obs = np.zeros(28, dtype=np.float32)
info = {"own_missiles_left": 0, "radar_state": "SEARCH",
        "dist_m": 40000.0, "r_min_own_m": 1000.0, "r_max_own_m": 50000.0}
print(compute_action_mask(obs, info))   # [T T T T F F T]
```

## 7. 回归红线

- 不修改 `common/bvr_combat_env.py` 物理/奖励/终止逻辑；
- 不改 ActionV1 7 维契约；
- 不升级/降级 torch/SB3/gymnasium；
- 不提交 `多任务智能体/models/`、`tensorboard/`、缓存；
- 不 push；P1 状态保持 🟡，长训练/命中率另做。
