# 阶段三 JSRL 底座：专家智能体 DRL 实现说明

> 版本：v0.1（中期检查版）｜日期：2026-10-30｜范围：阶段三 DRL 底座，不含完整 JSRL 训练循环、多级协同与完整 2.6/2.7 战术树。
>
> 设计依据：`多任务智能体/统一输出接口与DRL网络设计.md` §3；接口约定兼容根目录 `接口规范.md` v1.0。

---

## 1. 本轮结论

本轮交付“可实例化、可小规模训练、可被行为树按任务名调用、输出合法 ActionV1”的 4 个任务级专家智能体，并把 5 个既有技能级智能体纳入统一注册表：

- **任务级（新增）**：`strike` / `patrol` / `defense` / `ew`，统一 `get_action(obs) -> PolicyOutput(ActionV1)`；
- **技能级（保留）**：`cruise_hold` / `turn` / `climb_descent` / `pursuit` / `evasion`，保持原 `TaskAgent.predict` 与 3 维/离散动作，不退化；
- **行为树**：新增独立 py_trees 任务选择通道，按黑板任务名选择 4 个任务级智能体，叶节点 `get_action → to_env_action → BVRCombatEnv.step(Dict)`；
- **JSRL 接入点**：`get_action(..., guide_action=..., guide_steps=...)` 已预留并实现透传/占位，**未实现**完整课程衰减、`jsrl_trainer.py` 与混合动作策略。

---

## 2. 专家智能体清单（以仓库定义为准）

| 层级 | task_id | 状态 | 观测 | 原/新动作 | 行为树/调用方式 |
|---|---|---|---|---|---|
| 技能级 | `cruise_hold` 巡航保持 | 已有，未重构 | 12 维 `Box` | `Box(3)` AP 指令 | `TaskAgent.predict`；`to_ap_command` 兼容 |
| 技能级 | `turn` 盘旋 | 已有，未重构 | 12 维 `Box` | `Box(3)` AP 指令 | 同上 |
| 技能级 | `climb_descent` 爬降 | 已有，未重构 | 12 维 `Box` | `Box(3)` AP 指令 | 同上 |
| 技能级 | `pursuit` 接敌追击 | 已有，未重构 | 17 维 `Box` | `Discrete(48)` → AP | `predict` 返回离散索引；`to_ap_command` 仅用于 ActionV1 兼容端 |
| 技能级 | `evasion` 威胁规避 | 已有，未重构 | 16 维 `Box` | `Box(3)` AP 指令 | 同 `cruise_hold` |
| 任务级 | `strike` 超视距打击 | **新增** | 28 维 BVR `Box` | `ActionV1 (7,)` | 行为树主通道 `get_action → to_env_action` |
| 任务级 | `patrol` 空域巡逻 | **新增（中期检查版）** | 28 维 BVR `Box` | `ActionV1 (7,)` | 同上 |
| 任务级 | `defense` 战术防御 | **新增（中期检查版）** | 28 维 BVR `Box` | `ActionV1 (7,)` | 同上 |
| 任务级 | `ew` 电子压制 | **新增（中期检查版）** | 28 维 BVR `Box` | `ActionV1 (7,)` | 同上 |

> `patrol/defense/ew` 本轮为**单机近似**：复用 1v1 `BVRCombatEnv` 的观测/物理/雷达/导弹链路，仅叠加任务标签与任务奖励塑形；多机航线、HVA、多方向来袭等按计划书 3.3 后续扩展，未在本轮声称完成。

---

## 3. 总体架构

```mermaid
flowchart TB
    BT[py_trees 任务选择树<br/>Selector(memory=False)]
    C1[MissionIs<br/>任务名/中文别名]
    L1[ExpertAgentControl<br/>按任务名创建智能体]
    REG[ExpertAgentRegistry<br/>4 任务级 + 5 技能级]
    AG[TaskDRLExpertAgent<br/>get_action]
    PO[PolicyOutput<br/>ActionV1 + mask + source]
    AD[action_adapter<br/>to_env_action]
    ENV[BVRCombatEnv / TaskMissionEnv<br/>Dict flight+weapon]
    TR[训练通道<br/>FlatTaskMissionEnv + 观测尺度层 + SB3 PPO]

    BT --> C1 --> L1
    L1 --> REG
    REG --> AG
    AG --> PO --> AD --> ENV
    AG -.guide_action/guide_steps.-> JSRL[JSRL 预留接口]
    TR --> AG
```

两条动作通道：

1. **主通道（任务级）**：`ActionV1` 7 维控制向量，训练用 `FlatTaskMissionEnv`（`Box(7)`），部署用 `TaskMissionEnv`（`Dict{flight,weapon}`）；
2. **兼容通道（技能级）**：`command_v1` 3 维自驾仪设定点 `[v_cmd_mps, roll_cmd_rad, alt_cmd_m]`，由 `to_ap_command`/`command_v1` 从 ActionV1 近似生成，仅用于旧模型/冷启动，不新增学习者。

---

## 4. ActionV1 / ActionMaskV1 / PolicyProtocol

### 4.1 ActionV1（冻结版本 `action_v1`）

文件：`多任务智能体/agents/policy_interface.py`

| 索引 | 字段 | 范围 | 语义 |
|---:|---|---|---|
| 0 | `throttle` | [0, 1] | 油门指令 |
| 1 | `roll` | [-1, 1] | 滚转/副翼 |
| 2 | `pitch` | [-1, 1] | 俯仰/升降舵 |
| 3 | `yaw` | [-1, 1] | 偏航/方向舵 |
| 4 | `fire` | {0,1} | 当前步发射请求（边沿） |
| 5 | `switch` | {0,1} | 切换武器槽（接口预留） |
| 6 | `ecm` | {0,1} | ECM 开关翻转（边沿） |

- 数据形态：`np.ndarray, shape=(7,), dtype=float32`；
- `clip_action_v1`：裁剪连续维、武器维按 `>0.5` 二值化；
- `action_spec(task_id)`：JSON 可序列化声明（版本/维度/范围/边沿维/单位/掩码规则），随模型保存为 `*.spec.json`。

### 4.2 ActionMaskV1

`compute_action_mask(obs, info, env_cfg)` 与 `BVRCombatEnv._try_fire_own` 的发射条件对齐：

- 0~3 恒 `True`；
- `fire`：`own_missiles_left > 0` ∧ `radar_state == "TRACK"` ∧ `r_min_own_m ≤ dist_m ≤ r_max_own_m`；
- `switch`：恒 `False`（当前单弹种）；
- `ecm`：恒 `True`；
- 信息缺失时保守禁用 `fire`。

推理时 `get_action` 内部计算掩码并清零非法维，`PolicyOutput.masked` 随结果返回。

### 4.3 PolicyProtocol / PolicyOutput

```python
out = agent.get_action(obs, info=info, deterministic=True,
                       guide_action=bt_action_v1, guide_steps=current_h)
# out.action: (7,) float32 ActionV1
# out.source: "drl" | "guide" | "fallback" | "safety"
# out.masked: (7,) bool
# out.action_version: "action_v1"
```

- `guide_steps > 0` 且 `guide_action` 非空 → `source="guide"`，动作透传后仍经裁剪/掩码（安全优先）；
- 本轮**不实现**引导步数衰减、引导样本回填与内在奖励，`jsrl_trainer.py` 留待后续；
- 模型异常时兜底 `SAFE_ACTION_V1`（巡航油门+平飞），行为树不中断。

---

## 5. 注册表

文件：`多任务智能体/agents/expert_registry.py`

- `ExpertAgentRegistry`：注册、别名解析、按 tier 查询、按任务名实例化；
- `build_default_registry()` 注册 9 项：
  - 任务级 4：`strike/patrol/defense/ew`，`tier="task"`，`action_channel="action_v1"`；
  - 技能级 5：`cruise_hold/turn/climb_descent/pursuit/evasion`，`tier="skill"`，`action_channel="command_v1"`；
- 中文别名：`超视距打击→strike`、`空域巡逻→patrol`、`战术防御→defense`、`电子压制→ew`；技能级同样支持中文别名；
- 常用 API：`create(name)`、`get(name)`、`resolve(name)`、`task_ids()`、`skill_ids()`、`list(tier)`。

示例：

```python
from agents.expert_registry import DEFAULT_REGISTRY
agent = DEFAULT_REGISTRY.create("超视距打击")
out = agent.get_action(obs, info=info)
```

---

## 6. 任务级智能体实现

### 6.1 文件与职责

| 文件 | 职责 |
|---|---|
| `agents/task_agents/base_task_agent.py` | 公共底座：环境/PPO/推理/训练/保存加载/rollout |
| `agents/task_agents/mission_env.py` | 单机任务环境适配、任务标签、奖励塑形、观测尺度层 |
| `agents/task_agents/strike_agent.py` | 超视距打击基线 |
| `agents/task_agents/patrol_agent.py` | 空域巡逻（中期检查版） |
| `agents/task_agents/defense_agent.py` | 战术防御（中期检查版） |
| `agents/task_agents/ew_agent.py` | 电子压制（中期检查版） |

### 6.2 环境映射（中期检查版）

`MissionMixin` 叠加在现有环境上，不修改 `common/bvr_combat_env.py`：

| 任务 | 环境类 | 任务标签（obs[26], obs[27]） | 塑形要点 |
|---|---|---|---|
| `strike` | `FlatTaskMissionEnv`（训练）/`TaskMissionEnv`（部署） | (1, 0) | 无额外塑形，复用 BVR 原奖励 |
| `patrol` | 同上 | (0, 1) | 雷达跟踪 +0.05、进入攻击区 +0.20、RWR 告警 -0.05 |
| `defense` | 同上 | (1, 1) | 每步 +0.02、ECM 开 +0.05、敌雷达跟踪 -0.10、RWR/MAWS 告警惩罚 |
| `ew` | 同上 | (0, 0) | ECM 开 +0.10，且敌雷达未跟踪再 +0.10；自身未跟踪 -0.02 |

> 标签是 28 维观测的 2-bit 临时编码，不是设计文档 §4.2 的 4 维 one-hot `obs_v2`；后续扩展时替换，不影响 ActionV1/PolicyProtocol。

### 6.3 训练配置

- 算法：SB3 PPO + `MlpPolicy`；
- 网络：`[128, 128]`，tanh；
- `n_steps=256`、`batch_size=64`、`n_epochs=4`、`lr=3e-4`、`γ=0.99`、`λ=0.95`、`clip=0.2`、`ent_coef=0.01`；
- 观测：策略内部固定尺度层 `2(x-lo)/(hi-lo)-1`，外部 `get_action` 仍接收原始 28 维观测；
- 完整训练目标步数 `1e7` 仅登记在配置中，本轮**未执行**；验收只跑 512 步/智能体。

### 6.4 保存/加载

- `save(path)`：SB3 `.zip` + 同名 `*.spec.json`（含 ActionV1 声明、任务、观测、算法、路径）；
- `load(path)`：重建相同观测/动作空间的环境后加载，确定性推理逐元素一致（验证见测试报告）；
- 默认模型目录 `多任务智能体/models/`（被 `.gitignore` 排除，可随时再生）。

---

## 7. 行为树调用

文件：`多任务智能体/agents/bt_expert.py`

- `MissionIs`：按黑板 `mission` 与任务名/中文别名匹配；
- `ExpertAgentControl`：创建注册表中的任务级智能体，循环调用 `get_action`，用 `to_env_action` 转 Dict 动作下发；
- `build_expert_tree`：构建 Selector → 各任务 Sequence（`memory=False`，便于任务切换时重新判定）；
- `run_missions`：按任务名顺序驱动树，记录每步 ActionV1/奖励/态势；
- 现有 `引导策略行为树/llm_bt` schema **未改动**；本轮新增独立 py_trees 调度通道，后续可把叶节点并入其动作目录。

最小用法：

```python
from agents.bt_expert import build_expert_tree, run_missions
root, bb = build_expert_tree(tasks=("strike", "patrol"), max_steps=24)
results = run_missions(root, bb, ["超视距打击", "空域巡逻"])
```

---

## 8. 技能级兼容

文件：`多任务智能体/agents/action_adapter.py`

- `to_env_action(a7)`：ActionV1 → `Dict{flight(4), weapon(3)}`；
- `to_flat_action(a7)`：ActionV1 → `Box(7)`；
- `to_ap_command(a7, obs)` / `command_v1`：ActionV1 → 3 维 AP 设定点：
  - `roll` 直通并裁剪 ±0.90；
  - `throttle → v + 200·(throttle-0.55)`，裁剪 180~280 m/s；
  - `pitch → alt + 50·pitch`（垂速/高度环逆映射近似），裁剪 3000~8000 m；
  - `yaw` 忽略；
- 技能级智能体仍按原 `predict` 调用，至少验证 `cruise_hold` 可实例化/推理，未接入行为树主通道。

---

## 9. 已知限制（明确不作为本轮验收项）

1. **无完整 JSRL**：只有 `guide_action/guide_steps` 透传占位，无课程衰减、引导样本策略、内在奖励、`jsrl_trainer.py`。
2. **无多机协同**：`patrol/defense/ew` 是单机近似；无 4 导航点航线、HVA、僚机、多方向来袭。
3. **无混合动作策略**：训练仍用 SB3 `Box(7)` 高斯策略，武器维在推理时阈值化+掩码；设计文档 P1 的 `HybridActorCritic` 未实现。
4. **无 LSTM/frame_stack**：雷达间歇丢失下的记忆能力未实现。
5. **任务标签/场景未达最终口径**：28 维观测、2-bit 任务标签、30~80 km/240 步默认场景；80~120 km/600 步参数化与 `obs_v2` 未完成。
6. **无性能达标声明**：512 步训练只验证“可训练、损失有限、接口正确”，不验证命中率/拦截率/HVA 存活率/EW 探测概率。
7. **模型未入库**：`models/` 被 `.gitignore` 排除；演示使用验证脚本产出的极小模型或随机初始化模型。
8. **演示时长极短**：行为树演示每任务 12 步，仅证明调度/接口闭环，不代表战术效果。

---

## 10. 关键文件

```
多任务智能体/agents/
├── policy_interface.py        ActionV1/ActionMaskV1/PolicyProtocol/PolicyOutput/action_spec
├── action_adapter.py          to_env_action/to_flat_action/to_ap_command/command_v1
├── expert_registry.py         9 项专家注册表 + 中文别名
├── bt_expert.py               py_trees 任务选择树 + 专家叶节点
└── task_agents/
    ├── base_task_agent.py     任务级公共底座
    ├── mission_env.py         单机任务环境/标签/塑形/观测尺度
    ├── strike_agent.py
    ├── patrol_agent.py
    ├── defense_agent.py
    └── ew_agent.py
多任务智能体/validation/
├── run_expert_validation.py   4 智能体小规模训练/保存加载验证
└── run_expert_bt_demo.py      行为树最小演示（4 任务）
多任务智能体/tests/test_expert_agents.py
docs/阶段三_专家智能体DRL_*.md
```
