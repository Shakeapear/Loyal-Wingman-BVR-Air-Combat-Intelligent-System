# 多任务专精 DRL 智能体：统一输出接口与网络设计

> **阶段三预研文档**（对应《组会安排_2026-09-19》行动项"JSRL 训练框架预研：环境并行/接口复盘点"）
> 版本 v0.1（设计草案）｜日期 2026-09-19｜范围：**只定义接口规范与网络设计**，完整训练按阶段三执行。
>
> 依据：
> - 《大创项目实施计划书》步骤 3.1（JSRL 训练框架）、3.2（BVR 打击智能体）、3.3（巡逻/防御/电子压制 + 统一接口）；
> - 《研究计划与实施方案》§3.1.3（PPO 网络/超参）、§3.2.2（分任务训练方案）；
> - 现网代码（截至 2026-09-19）：`多任务智能体/common/`、`agents/`、`引导策略行为树/llm_bt/`。

---

## 0. 结论摘要

1. **现状**：项目里存在**两套动作语义、四种输出形态**（技能级 3 维自驾仪指令 / 追击 48 离散 / 交战级 Dict 4+3 / 扁平 7 维），彼此没有统一的动作规范与推理协议；计划书 D3.3 最低完成标准要求的 `get_action(obs) → action` 尚未定义。
2. **建议**：以**交战级 7 维控制向量 `ActionV1`** 作为所有任务级 DRL 智能体的统一输出（与 `FlatActionBVRCombatEnv` 完全对齐，BVR 环境零适配）；用 `PolicyProtocol`（`get_action` + `ActionSpec`）统一推理入口；技能级 3 维 AP 指令降为兼容通道 `command_v1`，只用于冷启动/旧模型适配。
3. **网络**：PPO Actor-Critic，共享主干 FC(256→128)，连续 4 维高斯头（tanh 压缩）+ 武器 3 维 Bernoulli 头（**动作掩码**），可选 LSTM(128) 处理雷达间歇丢失；参数量约 4.1 万（+LSTM 约 17.3 万），CPU 可训。
4. **落地顺序**：P0 用 SB3 内置 `MlpPolicy + FlatActionBVRCombatEnv` 先跑通打击基线 → P1 自定义混合动作策略（掩码）→ P2 LSTM + JSRL 引导接口。巡逻/防御/电子压制需先扩展多机环境（见 §4.1）。

---

## 1. 术语与层级

| 层级 | 含义 | 现有产物 | 阶段三角色 |
|------|------|---------|-----------|
| **技能级** | 单机基本机动技能（巡航/盘旋/爬降/追击/规避） | `common/jsbsim_env.py` + `agents/*.py`（5 个） | 已完成的历史交付；**不进入阶段三训练主线** |
| **任务级（本文重点）** | 4 种作战任务智能体：超视距打击 / 空域巡逻 / 战术防御 / 电子压制 | 无（阶段三交付 D3.2/D3.3） | 训练主体，统一输出接口的适用对象 |
| **行为树侧** | 引导策略行为树（2.5/2.6）+ 三层调度行为树（3.4） | `引导策略行为树/llm_bt/executor.py` | 既是 JSRL 的引导策略，也是任务级智能体的调度方 |

> 注：`项目状态与进度总览.md` §4 已记录"技能级 5 智能体"与计划书"任务级 4 智能体"的术语差异；本文的"多任务专精 DRL 智能体"特指**任务级**。

---

## 2. 当前输出接口盘点（事实，截至 2026-09-19）

### 2.1 总览

| # | 层级/载体 | 输出表示 | 语义与触发 | 决策频率 | 代码出处 |
|---|-----------|---------|-----------|---------|---------|
| 1 | 技能级连续（巡航/盘旋/爬降/规避） | `Box(3)` | `[v_cmd m/s, roll_cmd rad, alt_cmd m]` → 自驾仪设定点 | 1 Hz（内环 120 Hz） | `common/jsbsim_env.py:33-34,126-135` |
| 2 | 技能级离散（追击） | `Discrete(48)` | 16 航向 × 3 油门 → 解码回同一 3 维 AP 指令 | 1 Hz | `common/jsbsim_env.py:137-149` |
| 3 | 智能体封装 | `TaskAgent.predict()` 原样返回 env 动作 | 无掩码、无元数据、无统一 spec | — | `agents/base_agent.py:82-83` |
| 4 | 交战级（训练环境） | `Dict{flight Box(4), weapon MultiDiscrete([2,2,2])}` | 油门+滚/俯/偏三轴；发射/切弹/ECM **边沿触发** | 1 Hz（内环 60 Hz） | `common/bvr_combat_env.py:294-299,431-445` |
| 5 | 交战级（SB3 兼容） | `Box(7)` | 前 4 维连续 + 后 3 维按 `>0` 阈值化为 0/1 | 1 Hz | `common/bvr_combat_env.py:842-862` |
| 6 | 行为树执行侧 | `tick_once() -> Dict{flight, weapon}` | 动作节点把指令写入黑板 `action` 后整步下发 | 1 Hz | `引导策略行为树/llm_bt/executor.py:53-79,301-306` |
| 7 | 录制/可视化侧 | `get_viz_frame()` 全量态势帧 | **不含任何动作字段** | 每步 | `common/bvr_combat_env.py:717-795`；`可视化工具/visualization/recorder.py:23-36` |

### 2.2 技能级接口细节

- 动作空间：
  - 连续 `Box([180, -0.90, 3000], [280, 0.90, 8000])`，单位 m/s、rad、m（`jsbsim_env.py:33-34`）；
  - 追击任务 `Discrete(48)`：`heading_bin*3 + throttle_bin`，航向档 `-π~π` 等分 16 档，油门档 200/230/260 m/s；航向误差经 P 控制（增益 0.8）转滚转指令，高度维持当前高度（`jsbsim_env.py:137-149`）。
- 下发路径：`fcs/ap-speed-setpoint-fps` / `fcs/ap-roll-setpoint-rad` / `fcs/ap-alt-setpoint-ft`，由 F-104 自驾仪 `autopilot-hold.xml` 以 120 Hz 内环执行（`jsbsim_bridge.py:184-190`）。
- 智能体侧：
  - `TaskAgent.build_env/reset_env/build_model/train/predict/rollout/save/load`（`base_agent.py`）；
  - `rollout()` 按 trajectory_v1 字段记录 `action_speed_cmd_mps / action_roll_cmd_rad / action_alt_cmd_m`（`base_agent.py:88-99`）；
  - 模型名 `{algo}_{task}.zip`，网络 `net_arch` 按任务分档（`common/config.py:21-72`）。

### 2.3 交战级接口细节（阶段三主训练环境）

- 动作（`bvr_combat_env.py:294-299`）：
  - `flight = [throttle 0~1, roll -1~1, pitch -1~1, yaw -1~1]` → `step_manual()` 直接驱动
    `fcs/throttle-cmd-pilot`、`fcs/aileron-cmd-norm`、`fcs/elevator-cmd-norm`、`fcs/rudder-cmd-norm`
    （F-104 人工通道，非自驾仪；`jsbsim_bridge.py:171-180`）；
  - `weapon = [fire, switch, ECM]`，**边沿触发**：`fire=1` 尝试发射、`switch=1` 翻转武器槽（当前为接口预留，`own_selected` 未参与武器差异）、`ECM=1` 翻转干扰开关（`bvr_combat_env.py:437-444`）。
- 发射条件（掩码设计的事实依据，`_try_fire_own`，`bvr_combat_env.py:558-575`）：
  1) `own_n_left > 0`；2) 本机雷达状态 = `TRACK`；3) `r_min_own_m ≤ dist ≤ r_max_own_m`（reset 时解算的静态攻击区）。
- 观测：28 维 `Box`（`OBS_FIELDS_BVR`），含本机 12 维 + 武器/ECM 2 维 + 目标 7 维（未探测置 0）+ 威胁 4 维 + **任务标签 one-hot 2 维（26/27，当前恒 `[1,0]`=BVR 截击）**。
- 扁平化封装：`FlatActionBVRCombatEnv`（Box(7)，后 3 维 `>0` 阈值），使 SB3 PPO/SAC 可直连；`make_flat_bvr_env` 为 SubprocVecEnv 顶层工厂。

### 2.4 行为树执行侧（2.5/2.6）

- 叶子动作目录 7 个、条件 6 个（`llm_bt/schema.py:43-61`）；动作节点不直接产生 7 维动作，而是通过指令回路写成 `flight` 中间量（滚转经坡度内环、俯仰经垂速环、油门按速差微调，`executor.py:89-106`），武器为边沿位；
- `tick_once` 返回 Dict 动作，`run_tree_sim.py:91-101` 对 `flight` 做形状/有限性检查（异常计 `logic_errors`）并裁剪后送入环境；
- 结论：**行为树已经天然输出"交战级 Dict 动作"**，与任务级 DRL 的输出目标一致，缺的只是"统一规范 + 智能体推理协议"。

### 2.5 现状问题清单（推进训练前必须解决）

| # | 问题 | 影响 | 证据 |
|---|------|------|------|
| G1 | 两套动作语义（技能级 AP 指令 vs 交战级控制向量），无统一规范 | 任务级智能体无法套用同一训练/评估/调度代码 | §2.2 vs §2.3 |
| G2 | 无 `get_action(obs) → action` 协议（计划书 D3.3 最低完成标准） | 3/4 种智能体无法"全部封装为统一接口" | 计划书 3.3 |
| G3 | 混合动作空间（4 连续 + 3 离散）SB3 原生不支持，现用阈值化近似 | 武器动作无梯度/无概率、边沿语义未建模 | `bvr_combat_env.py:59-61,842-862` |
| G4 | 无动作掩码（无弹禁射、无跟踪禁射、冷却期） | 学出无效发射、白耗样本；安全违规统计失真 | `_try_fire_own` 静默返回 False |
| G5 | 任务标签仅 2 维，四任务无法区分 | 巡逻/防御/电子压制无法复用同一观测规范 | `OBS_FIELDS_BVR` 26/27 |
| G6 | 观测/配置面向 1v1（距离 30~80 km、240 步硬编码） | 打击任务计划要求 80~120 km、600 步；巡逻/防御需多机 | 计划书 3.2；`bvr_combat_env.py:356-363` |
| G7 | 动作不随帧记录（`get_viz_frame()`/CSV 无动作列） | 训练复盘、黑板"DRL 智能体 IO"模块无数据 | `recorder.py:23-36` |
| G8 | 部分可观测（雷达间歇丢失）无反挂记忆 | 现有 MLP 策略在丢失目标时无历史信息 | 计划书 3.1 增强特性 |

---

## 3. 统一输出接口设计

### 3.1 设计原则

1. **单一声明式规范**：一个动作向量 + 一份机器可读 `ActionSpec`（维度/范围/触发语义/单位），不再靠约定俗成；
2. **与现有环境零摩擦**：规范与 `FlatActionBVRCombatEnv` 完全一致，BVR 环境身份适配；BT 的 Dict 动作有确定转换；
3. **SB3 友好**：原生 `Box(7)` 可直连内置算法（P0 基线）；自定义混合策略在此规范上升级（P1）；
4. **可序列化**：动作、掩码、来源（DRL/引导/兜底）可写 JSON，直接支持行为树黑板与 CSV 录制；
5. **向后兼容**：技能级 AP 指令以 `command_v1` 兼容通道保留，不删除既有 5 个模型与接口。

### 3.2 `ActionV1` 规范（7 维，冻结版本 1）

所有任务级智能体的**唯一输出**：

| 索引 | 名称 | 类型 | 范围 | 语义 | 下游映射 |
|:---:|------|------|------|------|---------|
| 0 | `throttle` | 连续 | [0, 1] | 油门指令（0 慢车，1 加力） | `fcs/throttle-cmd-pilot` |
| 1 | `roll` | 连续 | [-1, 1] | 滚转（副翼）指令 | `fcs/aileron-cmd-norm` |
| 2 | `pitch` | 连续 | [-1, 1] | 俯仰（升降舵）指令 | `fcs/elevator-cmd-norm` |
| 3 | `yaw` | 连续 | [-1, 1] | 偏航（方向舵）指令 | `fcs/rudder-cmd-norm` |
| 4 | `fire` | 二值（边沿） | {0,1} | 发射请求（本步有效） | `weapon[0]` |
| 5 | `switch` | 二值（边沿） | {0,1} | 切换武器槽请求 | `weapon[1]` |
| 6 | `ecm` | 二值（边沿） | {0,1} | ECM 开关翻转请求 | `weapon[2]` |

- 数据形态：`np.ndarray, dtype=float32, shape=(7,)`，即 `ActionV1`；
- 边沿语义：第 4~6 维仅在**当前决策步**为正（>0.5）时生效，不保持；连续维 1 Hz 零阶保持到下一步；
- 单位：全部归一化，无物理单位（避免 v/alt 量纲进入策略）；物理裁剪由环境侧负责（现环境已 `clip`）。

> **为什么不是 3 维 AP 指令**：任务级智能体需要与行为树/JSRL 共享同一动作语义（BT 输出的就是 flight+weapon），且 4 维控制 + 武器可以表达能量机动/规避/OODA 全动作集；3 维 AP 指令表达能力受限且与武器无关，仅适合技能级。

### 3.3 `ActionMaskV1`（有效性掩码）

布尔向量 shape=(7,)，`True`=可选；训练时掩码写进策略分布（见 §4.4），推理时由同一函数计算：

| 维度 | 允许条件（与 `_try_fire_own` 严格一致） | 数据来源 |
|------|------------------------------------------|---------|
| 0~3 | 恒 True | — |
| 4 fire | `n_missiles > 0` ∧ `radar_state=="TRACK"` ∧ `r_min_own_m ≤ dist ≤ r_max_own_m`（+ 可选冷却） | `info["own_missiles_left"]`, `info["radar_state"]`, `info["r_min_own_m/r_max_own_m"]`, `info["dist_m"]` |
| 5 switch | 当前恒 False（单弹种，接口预留）；实现多弹种后放开 | `info` |
| 6 ecm | 恒 True（可选：限制翻转频率防抖动） | `info["ecm_on"]` |

实现位置：`agents/policy_interface.py::compute_action_mask(obs, info, env_cfg) -> np.ndarray[bool, 7]`（与环境同源，禁止各自重写）。

### 3.4 推理协议 `PolicyProtocol`（计划书 D3.3"统一接口"）

```python
# agents/policy_interface.py（设计草案，阶段三 D3.1/D3.3 落地）
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np

ACTION_V1_VERSION = "action_v1"
ACTION_V1_DIM = 7
ACTION_V1_LOW  = np.array([0., -1., -1., -1., 0., 0., 0.], dtype=np.float32)
ACTION_V1_HIGH = np.array([1.,  1.,  1.,  1., 1., 1., 1.], dtype=np.float32)

@dataclass
class PolicyOutput:
    """一次推理的完整结果（动作 + 学习/追溯元数据）。"""
    action: np.ndarray                 # (7,) float32，ActionV1
    source: str = "drl"                # "drl" | "guide" | "fallback" | "safety"
    log_prob: float | None = None      # 训练用；推理可 None
    value: float | None = None
    hidden_state: Any = None           # LSTM 状态（P2），黑板上随策略保存
    masked: np.ndarray | None = None   # (7,) bool，本步 ActionMaskV1
    action_version: str = ACTION_V1_VERSION

class PolicyProtocol(Protocol):
    """所有任务级智能体（含 JSRL 探索策略）统一入口。"""
    task_id: str                                       # strike/patrol/defense/ew
    def action_spec(self) -> dict: ...                 # JSON 可序列化（见下）
    def reset(self) -> None: ...                       # episode 开始清 LSTM/内部状态
    def get_action(self, obs, *, info=None, deterministic=True,
                   mask=None, guide_action=None, guide_steps=0) -> PolicyOutput: ...

def action_spec(task_id: str) -> dict:
    return {
        "version": ACTION_V1_VERSION, "task_id": task_id, "dim": ACTION_V1_DIM,
        "fields": ["throttle", "roll", "pitch", "yaw", "fire", "switch", "ecm"],
        "low": ACTION_V1_LOW.tolist(), "high": ACTION_V1_HIGH.tolist(),
        "edge_dims": [4, 5, 6], "frame": "ENU/SI", "decision_hz": 1.0,
    }
```

约定：
- `get_action` **永不行边沿保持**——调用方每步重新调用；
- `guide_action/guide_steps` 为 JSRL 预留：`guide_steps > 0` 且提供 `guide_action` 时返回 `source="guide"`（见 §4.7）；
- 掩码既可由调用方传入，也可在策略内部按 `compute_action_mask` 计算（训练时强制内部计算）。

### 3.5 适配层（ActionV1 ↔ 现有环境）

| 目标环境 | 适配函数 | 规则 |
|---------|---------|------|
| `BVRCombatEnv` | `to_env_action(a7) -> {"flight": a7[:4], "weapon": (a7[4:7] > 0.5).astype(int)}` | 身份级，无信息损失 |
| `FlatActionBVRCombatEnv` | 恒等 | `Box(7)` 完全一致 |
| `JSBSimFlightEnv`（技能级，兼容） | `to_ap_command(a7, obs) -> Box(3)`：`roll` 直通；`throttle` → 速度指令 PI（`v_cmd = clip(v + Kp·(throttle-0.55), 180, 280)`）；`pitch` → 垂速→高度指令环（复用 `executor.py:98-101` 增益）；`yaw` 忽略 | 仅冷启动/旧模型兼容，训练不用 |

纯逻辑函数放 `agents/action_adapter.py`，单元测试覆盖：形状/范围/阈值、边沿位、慢车/加力边界、与 `FlatActionBVRCombatEnv.step` 的行为一致性。

### 3.6 与行为树、黑板、录制的集成

1. **行为树（L3 调度，阶段四）**：每个任务智能体封装为叶节点 `ReadSituation → get_action → SafetyGuard(L2) → 下发 ActionV1`；`PolicyOutput.source` 写入黑板决策链（计划书 3.5 的"DRL 智能体 IO"模块）。
2. **JSRL 引导（阶段三 3.1）**：引导策略（2.6 行为树）输出经 `to_env_action` 转 ActionV1；`get_action(..., guide_action=..., guide_steps=h)` 决定本步由谁出动作。
3. **录制（建议增补，待与可视化工具协调）**：
   - 环境侧：`step()` 保存 `last_action_v1`，写入 `info["action_v1"]`（7 值）与 `info["action_source"]`；
   - `get_viz_frame()` 增加 `"action"`（7 值 + source），`EpisodeRecorder.CSV_FIELDS` 增加 `act_throttle…act_ecm, act_source` 8 列（旧 CSV 读取保持兼容：缺列按空处理）；
   - 效果：训练/对抗数据可复盘"**状态→动作→奖励**"完整链，满足阶段三/五的追溯要求。

### 3.7 两种通道与版本策略

- `control_v1`（= ActionV1）：**任务级训练/部署的唯一目标通道**；
- `command_v1`（3 维 AP 指令）：仅技能级兼容与冷启动，接口对应 `JSBSimFlightEnv`，**不新增学习者**；
- 版本：`action_v1` 冻结；任何维度/语义变更必须升版本（`action_v2`），并在 `PolicyOutput.action_version` 与黑板日志中标注；`ActionSpec` 随模型文件一起归档（D3.2-3 推理接口说明的组成部分）。

---

## 4. DRL 神经网络设计

### 4.1 任务矩阵与环境就绪度

| 任务 | 计划成功判据 | 训练环境现状 | 需要的环境扩展 |
|------|-------------|-------------|---------------|
| 超视距打击 strike | 命中率 > 60%（纯 PPO baseline < 30%） | `BVRCombatEnv` 1v1 已有；**初始距离 30~80 km、240 步硬编码**（计划要求 80~120 km、600 步） | 场景参数化：`reset` 距离/高度/态势区间、`max_steps` 入 config（G6） |
| 空域巡逻 patrol | 渗透拦截率 > 70% | 无 | 多机扩展：4 导航点航线、随机方位入侵者（可复用敌方脚本策略）、航线偏差项 |
| 战术防御 defense | HVA 存活率 > 80% | 无 | 多机扩展：HVA（可复用 JSBSim 实例）、僚机、多方向来袭 |
| 电子压制 ew（可选） | 友军被探测概率降低 > 30% | 部分：ECM 效果已有（敌雷达有效距离 ×0.5，`ecm_radar_range_factor`） | 探测概率统计口径、多目标干扰、掩护对象状态观测 |

> 环境扩展统一落 `common/bvr_multi_env.py`（或 `bvr_combat_env.py` 的 `config` 扩展），**保持 §3 的 ActionV1/PolicyProtocol 不变**；先打 strike（现有环境即可），再扩多机。

### 4.2 输入设计

**基础观测**：沿用 28 维 `OBS_FIELDS_BVR`（有限 Box，环境内已裁剪），阶段三增补为 `obs_v2`：

| 块 | 维度 | 内容 | 说明 |
|----|:---:|------|------|
| 本机状态 | 12 | 现有 0~11 | 不变 |
| 武器/ECM | 2 | 现有 12~13 | 不变 |
| 目标（雷达） | 7 | 现有 14~20 | 未探测全 0 + 标志位，保留 |
| 威胁（RWR/MAWS） | 4 | 现有 22~25 | 不变 |
| **任务标签** | **4** | strike/patrol/defense/ew one-hot（替换现有 2 维） | G5：四任务共享网络的前提 |
| **导航/任务** | 6 | 航线偏差(2)、当前目标点相对向量(3)、返航余油标志(1) | 巡逻/防御用；strike 置 0 |
| **协同/保护对象** | 6~12 | 僚机与 HVA 相对状态（各 3~6 维） | 多机扩展后启用；strike 置 0 |

**归一化**（三选一，建议 A）：
- A. **固定尺度层**：用 `OBS_LOW/HIGH` 做 `2(x-lo)/(hi-lo)-1`，作为策略网络第一层（不可训）；`.pt` 自包含、跨机器可复现。注意 0 哨兵字段（未探测目标）缩放后为常数，由 `tgt_detected` 标志消除歧义；
- B. `VecNormalize` 运行均值/方差：省事但统计量需随模型归档，部署易错；
- C. 不做归一化：仅当观测全部量级相近时可行，本项目不满足。

**部分可观测**：优先 `frame_stack=4`（把最近 4 步观测拼成 4×N 输入，简单稳）；若验证显示雷达丢失期决策仍差，再上 LSTM（§4.5）。

### 4.3 网络总体结构（默认 PPO 方案，对齐计划书 §3.1.3）

```
obs_v2 (N≈40)
   │ 固定尺度层（A，可选）
   ├─ 共享主干: FC256 + Tanh → FC128 + Tanh          （计划书：共享 FC 256→128）
   │      │
   │      ├─ Actor-连续头: FC(128→4) → μ ；log_std 为 4 个独立可学习参数
   │      │       → 高斯采样 → tanh 压缩/映射 → [throttle, roll, pitch, yaw]
   │      ├─ Actor-武器头: FC(128→3) logits → 掩码 → Bernoulli{fire, switch, ecm}
   │      └─ Critic 头: FC(128→1) → V(s)
   └─ (P2 可选) 主干前插入 LSTM(128, 1 层)，policy 侧每步维护 hidden_state
```

参数量（N=28 估算）：

| 部件 | 参数 |
|------|------|
| 主干 FC(28→256→128) | 40,320 |
| 连续头（4 均值 + 4 log_std） | 520 |
| 武器头（3 logits） | 387 |
| Critic 头 | 129 |
| **合计** | **≈ 41.4k** |
| 追加 LSTM(128) | +131,584（≈ 17.3 万） |

结论：网络极小，**训练瓶颈在 JSBSim 仿真而非 GPU**；RTX 4060 级别绰绰有余，CPU 训练也可行（与计划书算力预算一致，余量留给并行环境）。

### 4.4 混合动作头与概率分布（G3/G4 的解法）

将动作拆成两组条件独立的分布并做**因子化**：

```
π_θ(a|s) = N(μ_φ(s), σ_φ)            # 连续 4 维（state-independent log_std）
         × Π_{i∈{fire,switch,ecm}} Bernoulli(p_i(s) | mask_i)
log π_θ(a|s) = Σ_{连续维} log N(z; μ, σ) − Σ log(1 − tanh²(z))   # tanh 压缩的雅可比修正
             + Σ_{武器维} log Bernoulli(a_i; p_i, mask_i)
熵 = 高斯熵 + 各 Bernoulli 熵之和（掩码维熵=0）
```

- **tanh 压缩映射**：连续头输出 `z`，策略动作为 `a_cont = tanh(z)`；`throttle = (a_0 + 1)/2`，其余滚/俯/偏直接用 `a_1..a_3`。PPO 更新使用带雅可比修正的对数概率（SB3 `SquashedDiagGaussian` 同款处理）；
- **掩码**：`logits_i += (mask_i − 1) · 1e8`（等价于把非法动作概率压到 0）；训练与推理必须用同一掩码函数（§3.3）；
- **边沿动作在轨迹中的表示**：BD 采样后按 `>0.5` 转 0/1（推理时 `p>0.5`），转移存入 buffer 的为该二值动作，`log_prob` 用分布计算——**这是 G3 的关键：让"发射/开关"进入概率模型，而不是阈值截断**。

### 4.5 LSTM（P2，处理雷达间歇丢失）

- 结构：主干后接单层 `LSTM(128 → 128)`，policy 与 value 共享 LSTM 输出；
- 状态管理：`PolicyOutput.hidden_state` 随 episode 保存/复位；训练时按 episode 边界截断 BPTT（SB3 `RecurrentPPO` 思路）；
- 优先级低于 P0/P1：先用 frame_stack 验证"记忆是否真的带来收益"，避免过早复杂化。

### 4.6 超参与训练配置（计划书 3.1/3.2 直接映射）

| 项 | 取值 | SB3/实现映射 | 依据 |
|----|------|-------------|------|
| 算法 | PPO（clip + GAE + 熵正则） | P0 `PPO`；P1 自定义 `HybridActorCriticPolicy` | 计划书 3.1 |
| 并行环境 | 8 | `SubprocVecEnv([make_flat_bvr_env]*8)` | 计划书 3.2 |
| 每环境 rollout | 256 步（8×256 = 2048/更新） | `n_steps=256` | 计划书 3.2 |
| batch | 64 | `batch_size=64` | 计划书 3.1 |
| epochs | 10 | `n_epochs=10` | 研究计划 §3.1.3 |
| lr actor / critic | 3e-4 / 1e-3 | P0 用 `learning_rate=3e-4`；P1 分头设置 | 计划书 3.1 |
| γ / λ_GAE / ε_clip / β_entropy | 0.99 / 0.95 / 0.2 / 0.01 | `gamma/gae_lambda/clip_range/ent_coef` | 计划书 3.1 |
| episode 长度 | 600 步（10 min @1Hz） | 环境 config `max_steps=600` | 计划书 3.2 |
| 总步数 | 1×10^7（打击；其余任务同量级） | `total_timesteps=1e7` | 计划书 3.2 |
| 监控 | TensorBoard：reward、actor/critic loss、guide 衰减曲线 | `tensorboard_log` | 计划书 3.1 |
| 评估 | 每 500 episode 固定测试集 | 回调 + `EvalCallback` 思路 | 计划书 3.1 |
| checkpoint | 每 1000 episode | `CheckpointCallback` | 计划书 3.1 |

工程提示：
- 8 实例下实测吞吐约 1611 steps/s（4 实例，`README_环境.md` §8）→ 1e7 步纯仿真约 1~2 h；每次 reset 含攻击区解算（约 1.2~3 s），600 步 episode 摊薄后可接受；
- 训练环境建议 `render_mode=None`（零开销），评估/演示用 `可视化工具/`；
- 训练场景参数化后，固定测试集 = 预生成 seed 列表（如 20 个 seed × 4 种初始态势），保证各任务评估口径一致。

### 4.7 JSRL 引导衔接（与 2.6 `guide_interface.py` / 阶段三 3.1 对齐）

课程式引导步数衰减（研究计划 §3.1.2）：

| 阶段 | Episode | 引导步数 h | 说明 |
|------|---------|-----------|------|
| 冷启动 | 0–500 | 全 episode | 行为树完全接管，DRL 观察 |
| 引导期 | 500–2000 | → 0.7·H_max | 逐步交棒 |
| 过渡期 | 2000–4000 | → 0.3·H_max | DRL 主导后半段 |
| 微调期 | 4000–6000 | → 0.1·H_max | 少量引导 |
| 独立期 | 6000–10000 | 0 | 完全自主 |

接口约定（`get_action` 已预留）：

```python
out = policy.get_action(obs, info=info, guide_action=bt_action_v1,
                        guide_steps=current_h, deterministic=False)
# episode 内 step < current_h 且 bt_action_v1 非空 → source="guide"，否则 source="drl"
```

- 引导期转移的处理方式（用引导动作送入 buffer / 仅 DRL 动作入 buffer / 折扣回填）属于 3.1 算法实现细节，但**接口边界固定**：引导策略只提供 `ActionV1`，不直接写环境；
- 内在奖励（行为树子目标完成 → 额外奖励）由 `jsrl_trainer.py` 汇总，建议子目标信号取 `llm_bt` 节点状态（`lock_target`/`fire_missile`/`evade_missile` 的 SUCCESS 边沿）；
- `update_guide_steps(metric)`：按"探索策略平均奖励 > 引导策略 × N%"或固定间隔触发，衰减调度器状态随 checkpoint 保存。

### 4.8 参考实现骨架（P1，设计草案）

```python
# agents/hybrid_policy.py（设计草案；阶段三按此落地并做单测）
import torch, torch.nn as nn, numpy as np

class HybridActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 128)):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]), nn.Tanh())
        self.mu = nn.Linear(hidden[1], 4)
        self.log_std = nn.Parameter(torch.zeros(4))     # state-independent
        self.weapon_logits = nn.Linear(hidden[1], 3)    # fire, switch, ecm
        self.critic = nn.Linear(hidden[1], 1)

    def forward(self, obs, mask=None):
        h = self.trunk(obs)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        logits = self.weapon_logits(h)
        if mask is not None:
            logits = logits + (mask[:, 4:] - 1.0) * 1e8   # True→0，False→−1e8
        return mu, std, logits, self.critic(h).squeeze(-1)

def sample_action(mu, std, logits, deterministic=False, generator=None):
    if deterministic:
        z = mu
        w = (logits > 0).float()
    else:
        z = mu + std * torch.randn_like(mu)
        w = torch.bernoulli(torch.sigmoid(logits))
    cont = torch.tanh(z)
    throttle = (cont[..., 0] + 1.0) / 2.0
    return torch.cat([throttle.unsqueeze(-1), cont[..., 1:], w], dim=-1)

def log_prob(mu, std, logits, action):
    # action: (B,7)；反 tanh 压缩 + 雅可比修正；武器维 Bernoulli（掩码维恒真不贡献）
    a = action.clone()
    a[..., 0] = a[..., 0] * 2.0 - 1.0                 # throttle -> [-1,1]
    z = torch.atanh(a[..., :4].clamp(-1 + 1e-6, 1 - 1e-6))
    normal = torch.distributions.Normal(mu, std)
    lp_cont = normal.log_prob(z).sum(-1) - torch.log(1 - a[..., :4] ** 2 + 1e-6).sum(-1)
    bern = torch.distributions.Bernoulli(logits=logits)
    lp_w = bern.log_prob(action[..., 4:]).sum(-1)
    return lp_cont + lp_w
```

SB3 集成方式：继承 `ActorCriticPolicy`，覆写 `_build`（装配上述模块）、`forward`、`evaluate_actions`、`get_distribution`（返回组合分布对象），其余（GAE/裁剪/buffer）复用 SB3 实现。**必测**：① 采样动作全在 ActionV1 界内；② 掩码维 `log_prob` 对应项对参数梯度为 0；③ `log_prob(sample)` 与分布解析式计算一致（数值对拍）；④ `save/load` 后同 obs 同动作。

### 4.9 实施路线（P0 → P1 → P2）

| 阶段 | 内容 | 产出（映射计划书交付物） | 判定 |
|------|------|------------------------|------|
| **P0 基线**（先行，最快） | `FlatActionBVRCombatEnv` + SB3 PPO `MlpPolicy`；场景参数化（80~120 km/600 步）；固定测试集与评估回调 | 打击基线模型 + 学习曲线 | 能训、复现实验口径；建立纯 DRL baseline（计划书要求 <30% 命中率对照） |
| **P1 统一接口+混合策略** | `policy_interface.py`（ActionV1/掩码/`get_action`）+ `action_adapter.py` + `hybrid_policy.py` + `ppo_agent.py` | D3.1-1 `ppo_agent.py`、推理接口说明（D3.2-3）雏形 | 接口单测通过；打击任务命中率 > 60%（达标线） |
| **P2 JSRL+LSTM+多任务** | `jsrl_trainer.py`（引导衰减 + 内在奖励）→ `strike/patrol/defense/ew` 四智能体封装；多机环境扩展 | D3.1-2、D3.2-1、D3.3-1/2/3/4 | 计划书 3.3 验收：巡逻 >70%、防御 >80%、EW >30%，四者同一 `get_action` |

建议文件规划（阶段三新建，均放 `多任务智能体/`）：

```
agents/
├── policy_interface.py    # ActionV1 / ActionMaskV1 / PolicyProtocol / PolicyOutput / action_spec
├── action_adapter.py      # to_env_action / to_ap_command（兼容通道）
├── hybrid_policy.py       # HybridActorCritic + 组合分布 + 掩码
├── ppo_agent.py           # D3.1-1（训练入口，P0 可直接用 SB3 PPO，P1 换混合策略）
├── jsrl_trainer.py        # D3.1-2（guide_steps 衰减 + 内在奖励 + 评估/存档）
└── task_agents/           # strike_agent.py / patrol_agent.py / defense_agent.py / ew_agent.py
common/bvr_multi_env.py    # 多机/巡逻/防御场景扩展（保持 ActionV1 与观测规范）
configs/train_config.yaml  # D3.1-3
```

### 4.10 验收口径（与计划书 3.2/3.3 一致）

| 指标 | 阈值 | 统计方法 |
|------|------|---------|
| 打击命中率 | > 60%（baseline < 30%） | 固定测试集 20 组态势 × 3 次，取均值 |
| 巡逻拦截率 | > 70% | 随机方位入侵 20 局 |
| 防御 HVA 存活率 | > 80% | 多方向来袭 20 局 |
| EW 探测概率降低 | > 30% | 干扰开/关对照统计敌雷达 TRACK 时长占比 |
| JSRL 收敛加速 | 3~5× | 达到同等命中率所需 episode 比 |
| 统一接口 | 4 智能体全部 `get_action` 可调用 | 接口一致性单元测试 + 行为树叶节点联调 |

---

## 5. 风险与待确认

| # | 风险/待决 | 影响 | 建议 |
|---|----------|------|------|
| R1 | 混合动作策略（P1）为自研组件，SB3 版本升级可能破坏兼容 | 训练代码维护成本 | 固定 SB3 版本（现 DC 环境）并在 `agents/` 单测中锁定行为；核心数学逻辑与 SB3 解耦（分布/掩码为纯函数） |
| R2 | 多机环境（巡逻/防御）尚无实现 | 3.3 时间线 | 10 月预研期先做"敌方脚本策略复制为多实体"的最小多机原型，验证 ActionV1 不变 |
| R3 | 动作掩码与 `_try_fire_own` 逻辑漂移 | 训练/评估口径不一致 | 掩码函数单测直接对拍 `_try_fire_own` 的边界条件（无弹/未跟踪/超包线） |
| R4 | 录制 CSV 增列影响 `可视化工具` 回放兼容 | 工具链回归 | 增列为可选字段，旧 CSV 缺列按空处理；改动需跑可视化工具单测 |
| R5 | 任务标签扩 4 维改变观测维度 | 旧模型（若存在）不兼容 | 阶段三模型从零训练，无历史包袱；`obs_v2` 以版本号标注 |
| R6 | 电子压制任务缺乏"探测概率"评估基准 | EW 验收 | 用固定测试集统计敌雷达 TRACK/SEARCH 时长占比作为代理指标（与计划书口径对齐后固化） |

---

## 附录 A：接口映射对照

| 场景 | 输入载体 | 动作形态 | 转换 |
|------|---------|---------|------|
| 任务级 DRL（新） | `obs_v2` | `ActionV1 (7,)` | 恒等 |
| `BVRCombatEnv.step` | obs 28 | `Dict{flight(4), weapon(3)}` | `to_env_action` |
| `FlatActionBVRCombatEnv.step` | obs 28 | `Box(7)` | 恒等（训练通道） |
| 行为树 `tick_once` | obs 28 + info | `Dict{flight(4), weapon(3)}` | `to_env_action`（BT→ActionV1）|
| 技能级 `JSBSimFlightEnv.step` | obs 12~17 | `Box(3)`/`Discrete(48)` | `to_ap_command`（兼容通道）|
| 录制/可视化 | `get_viz_frame()` | `action`（建议增补） | 直接落 `ActionV1` + `source` |

## 附录 B：现有关键事实索引（实现时查阅）

- 环境动作/观测/奖励/终止：`多任务智能体/common/README_环境.md` §3-§5；
- 发射条件：`common/bvr_combat_env.py:558-575`；ECM 效果：`common/bvr_combat_env.py:472-473`、`README_环境.md` §5.3；
- 行为树动作目录/条件：`引导策略行为树/llm_bt/schema.py:43-61`；指令回路增益：`llm_bt/executor.py:89-106`；
- 训练接入示例（4 实例）：`README_环境.md` §10；吞吐实测：§8；
- 计划书网络/超参：步骤 3.1；打击训练场景与奖励：步骤 3.2；统一接口要求：步骤 3.3。

## 附录 C：变更记录

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-09-19 | v0.1 | 初版：现状盘点（含 G1~G8 缺口）、ActionV1/PolicyProtocol 规范、混合动作 PPO 网络设计与 P0/P1/P2 路线 |
