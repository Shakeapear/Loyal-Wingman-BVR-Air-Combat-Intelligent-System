# 战场要素建模：导弹与雷达（步骤 2.2 交付文档）

本文档对应《大创项目实施计划书》步骤 2.2"战场要素建模——导弹与雷达"的交付物
D2.2-1 ~ D2.2-4，说明模型公式、参数来源、与开源参考实现的差异，并给出后续
`BVRCombatEnv`（步骤 2.3）的接口用法示例。

| 交付物 | 文件 |
|--------|------|
| D2.2-1 导弹模型 | `common/missile_model.py` |
| D2.2-2 雷达模型 | `common/radar_model.py` |
| D2.2-3 攻击区曲线图 | `validation/results/D2.2-3_*.png`（出图脚本 `validation/make_launch_envelope_charts.py`） |
| D2.2-4 单元测试 | `tests/test_missile_model.py`、`tests/test_radar_model.py` |

所有代码为纯 numpy 实现（无新依赖），可直接
`from common.missile_model import Missile, launch_envelope` 导入。

---

## 1. 导弹模型（missile_model.py）

### 1.1 3DOF 质点动力学

惯性直角坐标系（ENU，z 向上，SI 单位）下：

- 速度矢量积分：`v̇ = a_ax·û + a_lat + g`，其中
  - `û = v/|v|` 为速度方向单位矢量；
  - 轴向加速度 `a_ax = (T(t) − D(t)) / m(t)`；
  - `a_lat` 为侧向过载（制导指令，限幅 `|a_lat| ≤ n_max·g`，默认 n_max=30g）；
  - `g = (0, 0, −9.80665) m/s²`。
- 阻力（与 CloseAirCombat 一致的形式）：
  `D = ½·c_D·S_ref·ρ(h)·|v|²`，`S_ref = π(d/2)²`（与速度正交的截面积），
  `c_D` 取常数 0.6（细长体导弹跨/超声速典型 0.4~0.8，常数近似）。
- 空气密度按国际标准大气（ISA）：
  `h < 11 km: ρ = ρ0·(T/T0)^(g/(R·L)−1)`，`T = T0 + L·h`；
  `h ≥ 11 km: 等温层指数衰减`。ρ0=1.225 kg/m³，T0=288.15 K，L=−0.0065 K/m。
- 质量：`m(t) = m0 − ∫ṁ dt`，质量流量由 `T = g·Isp·ṁ` 反推（两段常值）。

### 1.2 双推力固体火箭发动机

```
             ┌ thr_boost (25 kN)    t ∈ [0, 3 s)    助推段（公开描述 2~4 s）
T(t) =       ├ thr_sustain (9 kN)   t ∈ [3 s, 11 s) 续航段（公开描述 6~10 s）
             └ 0                    t ≥ 11 s        被动段（滑翔）
```

总冲 ≈ 25kN×3s + 9kN×8s = 147 kN·s；Isp=250 s（固体推进剂典型 240~260 s），
燃料消耗 ≈ 60 kg，m0=160 kg → 末质量 100 kg（与 AIM-120 公开重量 ~157~161.5 kg 一致）。

### 1.3 制导律（比例导引 + 中制导交接 + 末端增广）

- **比例导引（TPN，矢量形式）**：`a_cmd = N·(Ω_LOS × V_m) + g·ẑ`
  其中视线角速度矢量 `Ω_LOS = e × V_rel / r`（e 为弹目视线单位矢量，r 为弹目距离，
  V_rel = V_t − V_m），N=4（文献典型 3~4）。式中 `+g·ẑ` 为重力补偿项，
  对应 CloseAirCombat 公式 `n_z = (Kv/g)·ε̇ + cosθ` 中的 cosθ 项——缺少该项时
  导弹在长距离平飞中会产生数公里的稳态下沉。
- **中制导交接（纯追踪段）**：目标位于后半球（速度矢量与视线夹角 > 90°）时，
  TPN 无指令（正后方为退化几何），先按纯追踪将速度矢量拉向视线：
  `a_cmd = K_pp·|V_m|·(e − cosθ·û)`，K_pp=3；转入前半球后自动切回 TPN。
  对应真实导弹的 INS/数据链中制导 + 末端主动导引头截获流程。
- **末端制导（APN）**：剩余飞行时间 `t_go = r/V_c < 3 s` 时叠加目标加速度的
  视线法向分量：`a_cmd = a_PN + (N/2)·(a_t − (a_t·e)e)`，提升对机动目标的末端精度。
- **过载限幅**：指令侧向过载（含重力补偿）幅值限幅到 `n_max·g`（默认 30g，
  AIM-120C/PL-12 公开机动能力 30~40G 取保守值），限幅前/后分别记录于
  `max_lat_cmd_g` / `max_lat_g`。

### 1.4 命中判定

- 杀伤半径 `kill_radius = 10 m`（高爆破片战斗部 + 近炸引信的仿真惯例）；
- **连续碰撞检测**：每步按弹目相对位移线段（考虑本步相对速度）到目标的最小距离
  判定，防止大积分步长"跨过"杀伤球导致的假脱靶；
- `miss_distance` 记录全程最小弹目距离（脱靶量）。

### 1.5 攻击区（LAE）与不可逃逸区（NEZ）快速解算

`launch_envelope(launcher_h, launcher_speed, target_h, target_speed, bearing_deg, ...)`：

- 输入：发射机/目标高度、速度、目标相对方位角（相对发射机航向，0°=正前方，
  ±180°=正后方；目标默认朝向发射机飞行）；返回 `r_max / r_min / r_nez`（m）。
- 方法：以本模型 3DOF 仿真为基准，对发射距离在 [1 km, 180 km] 上做
  **批量探测（几何级数扫描）+ 区间二分细化**（32 路向量化并行，单工况 <1 s）：
  - `R_max`：对直飞目标命中区间的**右边界**（最大发射距离）；
  - `R_min`：对直飞目标命中区间的**左边界**，并取 `max(·, 引信保险距离 1 km)`；
  - `R_nez`：对按最大过载（默认 9g，战斗机典型值）**水平盘旋规避**的目标的命中
    区间右边界，左右两个规避方向各算一次取更小值（保守）。
- NEZ ⊂ LAE 由定义保证（R_nez 上限取 R_max）。

---

## 2. 雷达模型（radar_model.py）

### 2.1 火控雷达 FireControlRadar

- **雷达方程**：`R_max(σ) = R_max_ref·(σ/σ_ref)^(1/4)`，
  R_max_ref = 80 km @ σ_ref = 5 m²（取自计划书 2.1 节"雷达探测距离 ≤80 km
  （对 RCS=5 m² 目标）"的约束）。例：战斗机 3 m² → 70.4 km，大型机 10 m² → 95.1 km。
- **扫描体积**：方位/俯仰 ±60°（相对机头视轴），体积外不可探测；
- **最大跟踪数**：10（超出时按距离优先，近者保留，其余转 LOST）；
- **状态机**（每目标）：
  `TRACK（跟踪）→ LOST（转出体积/超跟踪保持距离，记忆保留）→ 连续
  REACQUIRE_SCANS(=3) 次更新内满足探测条件 → 重捕获 TRACK`；
  跟踪保持距离取探测距离 ×1.1（滞后，避免边界抖动）；
- 默认确定性模型（`p_detect=1.0`），可设 <1 引入探测概率（带随机种子）。

### 2.2 RWR（无源雷达告警接收机）

- 上报灵敏度范围内的敌方雷达辐射源；方位仅粗测：**15° 量化**（精度约 ±10°）；
- **距离不可测**（无源定位）：`range=None`；
- 敌雷达对我进入单目标跟踪（STT 锁定）时输出**锁定告警** `alarm=True`。

### 2.3 MAWS（导弹逼近告警，紫外/红外告警近似）

- 探测距离 8 km（紫外告警典型 3~10 km 量级取中值）、近似全向；
- 仅对径向接近（接近率 > 50 m/s）的导弹告警；
- 估计到达时间 `TTA = r / V_c`。

---

## 3. 参数来源表

### 3.1 导弹参数（AIM-120C / PL-12 量级）

| 参数 | 取值 | 来源/依据 |
|------|------|-----------|
| 发射质量 m0 | 160 kg | AIM-120A/B 约 157 kg、C-5 约 161.5 kg（公开资料） |
| 弹径 d | 0.178 m | AIM-120 公开 7 英寸 |
| 阻力系数 c_D | 0.6（常数） | 细长体导弹跨/超声速典型 0.4~0.8，常数近似（CloseAirCombat 取 0.1，本模型取更保守的 0.6） |
| 助推段推力/时长 | 25 kN / 3 s | 双推力固体火箭"助推 2~4 s"公开描述；推力按总冲匹配估算（精确值未公开） |
| 续航段推力/时长 | 9 kN / 8 s | "续航 6~10 s"公开描述；推力按总冲匹配估算 |
| 比冲 Isp | 250 s | 固体推进剂典型 240~260 s |
| 侧向可用过载 | 30g | AIM-120C/PL-12 公开机动能力 30~40G（凤凰网等中文公开报道：闪电-10A 最大过载 38g），取保守 30 |
| 最大速度 | ~M3.5~4 | 本模型仿真结果 1090~1130 m/s（6 km 高度），与公开 M4 同量级 |
| 最大发射距离 | 仿真 50~70 km | 公开 AIM-120C 约 80~100+ km 为最优条件（高空、超音速发射、迎头）；本模型在 F-104 平台（M0.8~1.1、+5G 约束）与简化动力学下给出 50~70 km，量级合理 |
| 比例导引系数 N | 4 | 文献典型 3~4（CloseAirCombat K=3，BVRGym N=2） |
| 杀伤半径 | 10 m | 高爆破片战斗部 + 近炸引信仿真惯例 |
| 引信保险距离 | 1 km | 公开约数百米~1 km |

### 3.2 雷达参数

| 参数 | 取值 | 来源/依据 |
|------|------|-----------|
| 参考探测距离 | 80 km @ RCS=5 m² | 计划书 2.1 节雷达约束（老式雷达） |
| RCS 典型值 | 战斗机 3~5 m²，大型机 10 m² | 计划书 2.2 节检索惯例 |
| 扫描范围 | 方位/俯仰 ±60° | 计划书 2.2 节要求 |
| 最大跟踪数 | 10 | 计划书 2.2 节要求 |
| RWR 方位量化 | 15° | 典型 RWR 精度 ±10° 量级 |
| MAWS 探测距离 | 8 km | 紫外告警典型 3~10 km 取中值 |

---

## 4. 与开源参考实现的差异

| 项目 | CloseAirCombat（docs/missile_engine.md） | BVRGym（jsb_gym） | 本项目 missile_model.py |
|------|------|------|------|
| 动力学 | 弹道系 3DOF 质点（ṫ/θ̇/φ̇ 方程） | JSBSim 导弹 FDM + PID 自动驾驶仪 | ENU 直角坐标 3DOF 质点（等价于弹道系 3DOF） |
| 导引律 | PN，球坐标视线角速率形式，K=3，含重力补偿（cosθ 项） | TPN 矢量形式，N=2，高度/航向分离控制 | TPN 矢量形式，N=4（可选 3），重力补偿 + 后半球纯追踪交接 + 末端 APN |
| 过载限制 | 无显式限幅 | PID 控制间接实现 | 显式 30g 限幅（记录限幅前/后值） |
| 发动机 | 单推力 Isp·ṁ 模型 | JSBSim FDM | 双推力固体火箭（助推+续航） |
| 阻力 | c_D=0.1、动态等效面积 S(Δθ,Δφ) | JSBSim FDM | c_D=0.6、常数 S_ref=π(d/2)² |
| 命中判定 | — | 距离 < effective_radius=300 m | 杀伤半径 10 m + 连续碰撞检测 |
| 攻击区/NEZ | 无 | 无 | **新增**：批量探测+二分快速解算 R_max/R_min/R_nez |
| 依赖 | JSBSim 子模块 | JSBSim/pymap3d/PID | 仅 numpy |

要点：
- 本模型**不依赖 JSBSim**，可作为步骤 2.3 BVRCombatEnv 中导弹/雷达的底层；
- 命中判据比 BVRGym 的 300 m 杀伤半径严格（10 m），更贴近近炸引信物理；
- 攻击区（LAE）/NEZ 解算为本项目新增能力（两个参考项目均未提供）。

---

## 5. 攻击区曲线结论（D2.2-3）

出图工况：发射机与目标同速同高，目标始终朝向发射机（迎头进入）。

| 验收要点 | 结果 |
|----------|------|
| R_max 随高度升高而增大 | 迎头 R_max：3048 m→52.2 km、4572 m→60.4 km、6096 m→70.3 km ✓ |
| 侧向方位角下 R_max 明显减小 | 4572 m：迎头 60.4 km → 正侧方 45.1 km（−25%）→ 正后方 42.0 km ✓ |
| NEZ ⊂ LAE | 全部方位角 R_min < R_nez < R_max ✓（迎头 NEZ≈33 km ≈ 0.55×R_max） |
| 曲线形态 | 前半球心形（cardioid），随 |方位角| 增大单调下降，左右对称；与公开 LAE 图形态一致 |

注：NEZ 曲线存在 ±2~3 km 的二分边界抖动（规避目标命中区对距离敏感），图中做了
3 点平滑（原值见出图日志/缓存 npz）。

---

## 6. BVRCombatEnv 接口用法示例（步骤 2.3 预留）

```python
import numpy as np
from common.missile_model import Missile, launch_envelope
from common.radar_model import FireControlRadar, RWR, MAWS

# ---- 火控解算：是否满足发射条件（LAE 判据）----
env = launch_envelope(launcher_h=4572.0, launcher_speed=280.0,
                      target_h=4572.0, target_speed=280.0, bearing_deg=20.0)
r_now = 40000.0
if env.r_min <= r_now <= env.r_max:
    print("目标位于攻击区内，可发射；NEZ=%.1f km" % (env.r_nez / 1000.0))

# ---- 发射后逐帧推进导弹（每环境 step 调用）----
missile = Missile(pos=np.array([0.0, 0.0, 4572.0]), vel=np.array([280.0, 0.0, 0.0]))
missile.step(dt=0.2, target_pos=target_pos, target_vel=target_vel, target_acc=None)
if missile.hit:
    print("命中！脱靶量 %.1f m" % missile.miss_distance)

# ---- 机载火控雷达：探测/跟踪 ----
radar = FireControlRadar()                      # 80 km @ 5 m²，±60°，最多 10 目标
radar.update(targets=[{"id": "t1", "pos": t1_pos, "rcs": 5.0}],
             own_pos=own_pos, own_heading_deg=own_heading_deg)
state = radar.state("t1")                       # 'TRACK' / 'LOST' / None

# ---- RWR / MAWS 告警 ----
rwr = RWR()
info = rwr.update(emitters=[{"id": "e1", "pos": enemy_pos, "mode": "lock",
                             "lock_target": "self"}], own_pos=own_pos)
if info["alarm"]:
    print("锁定告警！方位约 %d°" % info["threats"][0]["bearing_deg"])

maws = MAWS()
w = maws.update(missiles=[{"id": "m1", "pos": m_pos, "vel": m_vel}],
                own_pos=own_pos, own_vel=own_vel)
if w["alarm"]:
    print("导弹逼近！TTA=%.1f s" % w["warnings"][0]["tta"])
```

---

## 7. 测试与验证（D2.2-4）

DC 环境运行 `python -m pytest 多任务智能体/tests -v`：**24 个用例全部通过**。
- 导弹（14 例）：迎头/尾追/侧向三种态势命中（含解析量级校验：命中时间、末速、
  纯迎头侧向过载 ≈1g）、超射程脱靶、大离轴角高速横穿脱靶、30g 过载限幅、
  攻击区趋势（高度↑→R_max↑、|方位角|↑→R_max↓、左右对称、NEZ⊂LAE）、
  批量解算与逐帧仿真交叉验证；
- 雷达（10 例）：RCS×16→R_max×2（R∝σ^(1/4)）、探测→跟踪→丢失→重捕获状态机、
  ±60° 扫描体积、10 目标上限（最近优先）、RWR 锁定告警/15° 量化/无距离、
  MAWS 告警与 TTA（解析对照 r/V_c）。
