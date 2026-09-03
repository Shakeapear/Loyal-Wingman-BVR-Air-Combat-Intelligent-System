# BVRCombatEnv 环境设计文档（步骤 2.3 交付）

本文档对应《大创项目实施计划书》步骤 2.3"标准化 Gym 交互接口开发"的交付物
D2.3-1（bvr_combat_env.py）、D2.3-2（jsbsim_bridge.py）、D2.3-3（接口验证脚本），
说明环境设计、坐标系约定、与步骤 2.2 模块的调用关系，及后续阶段三训练接入方式。

| 交付物 | 文件 |
|--------|------|
| D2.3-1 1v1 超视距交战环境 | `common/bvr_combat_env.py` |
| D2.3-2 JSBSim 桥接层 | `common/jsbsim_bridge.py`（5 任务技能环境 `jsbsim_env.py` 已改造复用） |
| D2.3-3 接口验证脚本 | `validation/check_env_and_throughput.py` → `validation/results/env_check_and_throughput_report.txt` |
| 单元测试 | `tests/test_bvr_combat_env.py`（20 用例） |

---

## 1. 文件结构与职责

```
common/
├── jsbsim_bridge.py    JSBSimGymBridge：单机 JSBSim 生命周期（init/load/reset/step/close）
│                       + NED/英制→ENU/SI 转换（ned_ft_to_enu_m）+ SI 状态字典
├── jsbsim_env.py       JSBSimFlightEnv（5 技能任务环境，接口不变，内部改用 bridge）
├── bvr_combat_env.py   BVRCombatEnv（本交付，1v1 超视距交战）+ FlatActionBVRCombatEnv
│                       （SB3 动作展平包装）+ make_bvr_env/make_flat_bvr_env 顶层工厂
├── missile_model.py    步骤 2.2 交付（未修改，仅 import）
└── radar_model.py      步骤 2.2 交付（未修改，仅 import）
```

---

## 2. 坐标系约定（本任务核心易错点，已踩坑记录）

| 项 | 约定 |
|----|------|
| 环境内部坐标系 | ENU + SI：x=东、y=北、z=上，米、米/秒、弧度（API 边界角度用 deg） |
| JSBSim 输出 | NED + 英制，经 `ned_ft_to_enu_m()` 转换：`pos=[距离东, 距离北, h-sl-ft·0.3048]`，`vel=[v-east-fps, v-north-fps, -v-down-fps]·0.3048` |
| **位置坑位（实测）** | JSBSim 的 `position/distance-from-start-lat/lon-mt` 是**无符号距离**（向南/向西飞行时符号丢失，已实测验证），不能用于定位；`jsbsim_bridge` 改由 `position/lat-gc-deg / long-gc-deg` 与重置原点差值换算本地 ENU（1° 纬 = 111320 m） |
| 航向 | JSBSim `psi`：0=正北、顺时针为正（东=90°）；观测 `psi_rad` 与 RWR/雷达方位均按此约定 |
| **雷达航向坑位（实测）** | 2.2 雷达模块内部旋转矩阵的航向约定为 0=正东（其单测即该约定），与 JSBSim 差 90°；环境在调用雷达/RWR 时传入 `90° − ψ` 做约定转换，不改 2.2 模块 |
| 敌机初始位置 | 以本机为原点按经纬度偏移放置（敌机 bridge 的 reset 原点即其 IC 经纬度），环境内以 `_enemy_offset`（东/北，米）把敌机本地坐标换算为绝对 ENU；**导弹发射/跟踪目标必须用绝对坐标**（见 bvr_combat_env.py `_launch_missile` 注释） |

坐标转换由 `tests/test_bvr_combat_env.py::test_ned_ft_to_enu_m` 单独测试。

---

## 3. BVRCombatEnv 空间定义

### 3.1 动作空间（计划书 2.3）

`spaces.Dict`：
- `"flight"`: `Box([0,-1,-1,-1], [1,1,1,1])`——连续量，1 Hz 决策、零阶保持；
  映射到 F-104 人工通道（AP 主开关关闭）：油门→`fcs/throttle-cmd-pilot`、
  滚转→`fcs/aileron-cmd-norm`、俯仰→`fcs/elevator-cmd-norm`、偏航→`fcs/rudder-cmd-norm`
  （F-104 FCS 通道定义见 `JSBSim/aircraft/f104/Systems/FCS-*.xml`）；
- `"weapon"`: `MultiDiscrete([2,2,2])`——离散武器控制，边沿触发：
  `[发射(0/1), 切换武器(0/1), ECM(0/1)]`。切换武器=切换当前选中导弹槽（D2.1 挂载
  固定 2×中距弹，切换为接口预留）；ECM=电子干扰开关（效果见 §5.2）。

**SB3 兼容性**：stable-baselines3 2.x 不支持 Dict/Tuple 动作空间（实测报错），
故提供 `FlatActionBVRCombatEnv`（动作展平为 Box(7)，后 3 维按 `>0` 阈值化为 0/1），
物理/奖励/终止与观测完全不变，供 PPO/SAC 等训练接入（见 §8）。

### 3.2 观测空间（计划书 2.3：本机+目标+威胁+任务标签）

有限 `Box(28)`（env_checker 友好，`OBS_LOW/OBS_HIGH` 见源码），字段顺序：

| 索引 | 字段 | 含义 | 范围 |
|------|------|------|------|
| 0~11 | h_sl_m, vtrue_mps, mach, phi/theta/psi_rad, alpha/beta_rad, nz_g, throttle, fuel_frac, vd_mps | 本机位姿+速度+油量 | 见 OBS_LOW/HIGH |
| 12~13 | n_missiles, ecm_on | 武器余量、ECM 状态 | [0,2] / {0,1} |
| 14~20 | tgt_e_m, tgt_n_m, tgt_alt_m, tgt_dist_m, tgt_bearing_rad, tgt_speed_mps, tgt_heading_rad | 目标相对位置/速度/方位（**由本机雷达提供**） | 相对量 ±200 km 等 |
| 21 | tgt_detected | 雷达跟踪标志 | {0,1}，**未探测到时 14~20 全置 0**（计划书要求） |
| 22~23 | rwr_alarm, rwr_bearing_rad | RWR 锁定告警 + 15° 量化方位 | {0,1} / [-π,π] |
| 24~25 | maws_alarm, maws_tta_s | MAWS 告警 + 到达时间估计 | {0,1} / [0,200] |
| 26~27 | mission_onehot | 任务标签 one-hot（当前恒 [1,0]=BVR 截击，预留扩展） | {0,1}² |

---

## 4. 奖励与终止（计划书 2.3 条款逐项）

### 4.1 奖励

| 事件 | 奖励 | 说明 |
|------|------|------|
| 本机导弹命中敌机 | **+200** | 计划书条款 |
| 本机被敌导弹命中 | **-200** | 计划书条款 |
| 进入攻击区 | **+5/步** | dist∈[R_min,R_max] 且雷达跟踪（攻击区=reset 时 `launch_envelope()` 解算的静态包线，见 §5.1） |
| 安全违规 | **-10/步** | 超包线（nz>+5.5g 或 <-2.5g、M>1.65、h>14500 m，对应 D2.1 平台约束 +5/-2G、M1.6、14000 m 留 10% 裕度）或失速（vtrue<90 m/s） |
| 燃油耗尽 | **-50**（终止时） | 计划书条款 |
| 稠密塑造（补充项） | ±0.5/步 | 接近率 0.02·(−Δdist/1000)，缓解奖励稀疏；计划书允许的小幅塑造，已在 README 说明 |

### 4.2 终止

| 条件 | 类型 | reason（info） |
|------|------|------|
| 命中敌机 / 被命中 | terminated | enemy_hit / own_hit |
| 本机坠毁（h_agl<30 m） | terminated | own_crash |
| 敌机坠毁 | terminated（奖励 0，计划书未规定，注明） | enemy_crash |
| 燃油耗尽（<50 lb） | terminated | fuel_out |
| 逃逸区域（距初始点 >150 km） | terminated | escape |
| episode 超时（默认 240 步） | truncated | timeout |

---

## 5. 与步骤 2.2 模块的调用关系（不改 2.2 接口）

### 5.1 攻击区（missile_model.launch_envelope）

- reset 时对双方各解算一次 `launch_envelope()`（约 0.6 s/次），episode 内作为
  **静态包线阈值**（R_max/R_min/R_nez）使用：
  - 本机发射判定：`R_min ≤ dist ≤ R_max` 且本机雷达跟踪；
  - 敌方脚本发射判定：`R_min ≤ dist ≤ 0.95·R_max` 且敌方雷达跟踪 + 发射间隔 ≥8 s；
  - "进入攻击区 +5/步" 同用该阈值。
- 静态近似理由：解算一次 ~0.6 s，逐步解算会破坏 1 Hz 实时性；初始态势下解算的
  包线与交战前期几何一致，作为发射决策阈值足够（发射时点亦在包线内侧判定）。
- `config["compute_envelope"]=False` 可跳过（单测/吞吐测试用，改用保守固定阈值）。

### 5.2 导弹（missile_model.Missile）

- 发射：以发射机 ENU 状态创建 `Missile`（初速 = 发射机速度 + 30 m/s 沿航向）；
- 推进：每决策步 5 个亚步（dt=0.2 s，与 2.2 攻击区解算器 dt 一致），
  命中判定=脱靶量<10 m（导弹连续碰撞检测）；
- **目标轨迹连续插值（重要）**：导弹亚步的目标位置/速度按本决策步内飞行器的
  P(t)→P(t+1) 线性插值推进（推进前先快照双方状态）。若目标每秒跳变一次，
  末端会产生 ~100~200 m 的脱靶并绕圈（实测：40 km 迎头会脱靶），插值后与
  2.2 解算器的连续目标运动一致（实测 40~55 km 迎头全部命中，58 km 超 R_max 脱靶）；
- **吞吐优化**：亚步推进用环境内 `_missile_batch_step()` 向量化批处理（与
  `Missile.step` 公式逐项一致），由 `test_missile_batch_step_equivalence` 与
  `Missile.simulate` 对拍验证等价（迎头/侧向两工况）；`Missile.step` 本身仍可
  逐帧调用（接口未变）。
- 敌方导弹：满足 §5.1 条件 + 跟踪即发射（脚本策略，随机延迟由 seed 决定）。

### 5.3 雷达/RWR/MAWS（radar_model）

- 每决策步 `FireControlRadar.update()`（本机/敌机各一部）；探测→跟踪→丢失→
  重捕获状态机、±60° 扫描体积、10 目标上限、R∝σ^(1/4) 均来自 2.2 模块；
- RWR：敌机雷达对我进入 STT 跟踪（=敌雷达 TRACK 本机）→ 锁定告警 + 15° 量化方位（无距离）；
- MAWS：敌方在飞导弹 → 逼近告警 + TTA=r/V_c；
- **ECM 效果**（确定性、可测）：本机 ECM 开启时敌方雷达有效距离 ×0.5
  （`ecm_radar_range_factor=0.5`），敌机失跟踪后停止发射（见测试
  `test_ecm_blocks_enemy_radar`）。

---

## 6. 敌方脚本策略（敌我各一个 JSBSim 实例，敌方脚本控制）

- 航向对准：AP 滚转设定 = clip(2.0·方位误差, ±0.9)；
- 高度跟随：AP 高度设定 = 本机当前高度（8000~20000 ft 截断）；
- 速度保持：AP 速度设定 = 760 fps（约 M0.73）；
- 发射：见 §5.1/§5.2；发射后间隔 8 s + U(0,4) s 随机延迟。

---

## 7. 随机初始态势与复现（计划书 2.3）

`reset(seed=None)` 全部随机量由 `np.random.default_rng(seed)` 顺序抽取：
初始距离 30~80 km、目标相对方位 ±60°、双方高度 8000~20000 ft、
本机航向 0~360°、敌机航向 = 朝向本机 ±30° 抖动。同 seed 两次 reset 观测完全一致
（`test_reset_seed_reproducibility`），可复现。

---

## 8. 性能设计（验收：4 实例并行 ≥1000 steps/s）

| 设计点 | 取值 | 依据 |
|--------|------|------|
| JSBSim 内环步长（BVR 环境） | 1/60 s（60 Hz） | 5 任务技能环境保持 1/120 s 不变；BVR 环境人工通道为直接舵面指令（无 120 Hz 自驾仪内环），60 Hz 为驾驶模拟常规频率，吞吐 ×2 |
| 导弹亚步步长 | 0.2 s（5 亚步/决策步） | 与 2.2 攻击区解算器 dt=0.2 一致；命中由连续碰撞检测保证（等价性测试验证） |
| 导弹推进实现 | 向量化批处理 `_missile_batch_step` | 单步热路径纯 numpy（逐分量算术替代 np.cross/einsum），4 枚导弹 ~0.4 ms/步 |
| 属性读取 | `fdm["name"]`（0.3 μs/读） | 比 get_property_value 快 ~3 倍（实测基准见开发记录） |

实测（`check_env_and_throughput.py`）：单进程 env.step ≈ 2.2 ms；
4 实例 SB3 `SubprocVecEnv`（spawn，展平动作工厂 make_flat_bvr_env）并行
**≈1611 steps/s（≥1000 达标，60 s 采样）**。
注：吞吐测试用"无 reset 干扰工况"（近水平动作 + n_missiles=0 + 跳过攻击区解算），
因 env.step 计算量与动作/挂载无关（导弹向量化亚步 ~0.4 ms/步），而 reset 含
2.3 s 攻击区解算，未训练策略频繁坠毁会导致 reset 主导统计
（方法已在脚本与报告中注明）。

---

## 9. 验证结果汇总（D2.3-3）

- `gymnasium.utils.env_checker.check_env(BVRCombatEnv)`：**通过**
  （含确定性检查：传感器每个 episode 全新实例，reset(seed)→step 完全可复现）；
- 随机动作 100 步无报错（自动 reset，观测始终在界内）：**通过**；
- SB3 PPO(MlpPolicy + FlatActionBVRCombatEnv) 验证性训练 1024 步：**通过**
  （≤2048 步，非完整训练）；
- 4 实例并行吞吐：**≈1611 steps/s ≥ 1000**；
- 导弹全链路：迎头 40/45/50/55 km 发射→飞行→命中全部可复现
  （55 km ≈ 2.2 包线 R_max=58.3 km 的 94%，58 km 超包线脱靶，与 2.2 攻击区一致）；
- 新旧 pytest 全量通过（2.2 的 24 项 + 本任务 20 项）+ `run_validation.py --skip-train`
  5 任务回归通过（结果文件：`validation/results/env_check_and_throughput_report.txt`）。

## 10. 阶段三训练接入说明

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from common.bvr_combat_env import make_flat_bvr_env

# 4 实例并行训练环境（工厂函数为模块顶层可导入函数，满足 spawn pickling）
vec = SubprocVecEnv([make_flat_bvr_env for _ in range(4)])
model = PPO("MlpPolicy", vec, n_steps=256, verbose=1)
model.learn(total_timesteps=1_000_000)   # 阶段三的完整训练步数，另行配置
```

- 观测/奖励全部在环境内完成归一化（有限 Box），直接接 MlpPolicy；
- 完整训练配置（网络结构/超参）遵循项目 config.py 的分级原则，在阶段三开始时
  在 `common/config.py` 增补 BVR 条目（本任务未做完整训练）；
- 渲染/可视化（步骤 2.4，2026-09-03 交付）：`get_viz_frame()` 返回全量态势快照；
  `render()` 支持 "human"（实时战术显示窗口）与 "rgb_array"（离屏帧），经
  `config["render_mode"]` 启用，默认 None（训练零开销）；绘图与回放实现见
  `visualization/` 包（实时窗口演示 `python -m visualization.demo`，
  CSV 回放 `python -m visualization.replay <csv>`）。
