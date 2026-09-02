# gym-jsbsim（单机 RL 控制环境，已下载）

## 基本信息

| 项目 | 内容 |
|------|------|
| 仓库 | https://github.com/Gor-Ren/gym-jsbsim（原版，253 星；新版 Gymnasium 版见 https://github.com/JGalego/gymnasium-jsbsim） |
| 内容 | 基于 JSBSim 的固定翼飞机 Gym 环境，Python 3.7+，MIT 许可 |
| 安装 | `pip install gym git+https://github.com/sryu1/gym-jsbsim jsbsim` |

## 提供的任务（接口设计范例）

- **HeadingControlTask**：保持初始高度直飞（航向保持）；
- **TurnHeadingControlTask**：转到指定航向并保持高度；
- 动作空间：升降舵 [-1,1] + 副翼 [-1,1] + 油门 [0,1]（连续），部分任务可选离散；
- 观测：约 12 维飞行状态（俯仰/滚转/航向角、空速、位置、舵面等）；
- 奖励：奖励整形（Shaping）与标准（Standard）两档。

## 与本项目的关系

本项目计划书步骤 2.3 的 `JSBSimGymBridge` 可直接参考此仓库：
- `gym_jsbsim\tasks.py`：任务/奖励/终止条件定义方式；
- `gym_jsbsim\envs.py`：`step/reset` 与属性读写封装；
- `gym_jsbsim\agents\`：PID 基准控制器（可作为行为树底层动作实现参考）。

> 注意：本项目使用本机 JSBSim 1.2.4 安装包；该仓库自带 JSBSim 依赖调用方式可参考，但版本接口以我们的 1.2.4 为准。
