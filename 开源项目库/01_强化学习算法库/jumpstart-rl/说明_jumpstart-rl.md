# jumpstart-rl（JSRL 论文实现）

## 简介

Jump-Start Reinforcement Learning（JSRL，ICML 2023 论文 arXiv:2204.02372）的第三方标准实现：用"引导策略（guide-policy）"为 RL 提供起始状态课程，显著提升探索效率。**本项目计划书阶段二/三采用的正是 JSRL 框架**（行为树引导策略 → 引导阶段步数 h 逐步归零 → 探索策略独立学习）。

- 仓库：https://github.com/steventango/jumpstart-rl
- 许可证：MIT（`pip install jsrl` 可安装）
- 源码位置：`源码\src\jsrl\`（核心实现）、`源码\examples\`（curriculum/random 两种训练示例）
- 备注：2024-01 后无更新（作者已定版），作为**标准参考实现**使用，非活跃维护项

## 选择理由

1. 计划书明确采用 JSRL 两阶段训练（步骤 2.6 引导接口 / 阶段三），该库是唯一以 SB3 为基座的现成实现；
2. 提供 `guide_curriculum_steps` 衰减机制与 `guide_kwargs` 接口，可把我们的**行为树引导策略**直接挂入；
3. 代码量小（core 仅数个文件），易于读懂后迁移到我们自己的训练框架。

## 运行说明

```powershell
pip install jsrl   # 依赖 stable-baselines3 + gymnasium-robotics

# 官方示例（PointMaze）：
python examples/train_jsrl_curriculum.py
python examples/train_jsrl_random.py
```

核心用法（对照我们项目的接入点）：

```python
# 官方模式：guide-policy 先执行 h 步，再由 exploration-policy 接手
model = TD3("MlpPolicy", env, ...)   # exploration-policy
model.learn(
    total_timesteps=...,
    callback=None,
    jsrl_guide_policy=bt_guide_policy,          # ← 我们的行为树引导策略
    jsrl_guide_steps=initial_h,                  # ← 初始引导步数
    jsrl_guide_curriculum_steps=decay_step,      # ← 每多少步衰减一次
)
```

> 组员学习路径：`源码\src\jsrl\jsrl.py`（滚动引导逻辑）→ `examples\train_jsrl_curriculum.py`。
