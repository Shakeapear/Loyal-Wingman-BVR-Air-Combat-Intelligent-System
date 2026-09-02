# tianshou（清华 DRL 库，中文文档）

## 简介

Tianshou 是清华大学开源的高质量 PyTorch 强化学习库（10.9k★，MIT），特点：
- 模块化设计（Policy/Buffer/Trainer/Collector 分离），改算法=改几行；
- 内置 PPO、SAC、DQN、TD3、MADDPG 等 20+ 算法，**支持多智能体**；
- 中文/英文双语文档，学习曲线友好。

- 仓库：https://github.com/thu-ml/tianshou
- 许可证：MIT
- 源码位置：`源码\tianshou\`、`源码\examples\`、`源码\docs\`

## 选择理由

1. 组员（本科阶段）上手最容易的主流 DRL 库，中文文档齐全；
2. 多智能体支持（MADDPG 等）可用于阶段四协同实验的快速原型；
3. 与 SB3 形成"主用 SB3、备选/教学 tianshou"的互补布局。

## 运行说明

```powershell
pip install tianshou
```

最小 PPO 示例（详见 `源码\examples\mujoco\ppo.py`）：

```python
import gymnasium as gym, torch, tianshou as ts
env = gym.make("JSBSimEnv-v0")                     # 换成本项目环境
policy = ts.policy.ActorCritic(
    ts.utils.net.common.Net(env.observation_space.shape, env.action_space.shape,
                            hidden_sizes=[256, 256], activation=torch.nn.Tanh), ...)
ppo = ts.policy.PPOPolicy(actor=policy, critic=..., optim=torch.optim.Adam(...))
trainer = ts.trainer.onpolicy_trainer(ppo, ts.data.Collector(...), ...)
```

> 组员学习路径：`源码\docs\zh\`（中文文档）→ `源码\tianshou\policy\modelfree\ppo.py`。
