# 可视化工具（项目级支撑工具，步骤 2.4 交付物 D2.4-1）

**定位：** 本目录独立于 `多任务智能体/` 智能体库——可视化不属于智能体库内容，
也不参与任何训练计算；它是项目的**训练自我调整 + 成果展示工具**：

| 用途 | 使用场景 | 本工具提供 |
|------|---------|-----------|
| **① 训练自我调整** | 训练/评测过程中观察智能体行为、诊断策略（为何不发射 / 为何失稳 / 奖励是否合理），据此调参、调整奖励函数、验证战术逻辑 | 实时 2D 战术窗口、逐帧 CSV 录制、任意 episode 的 GIF/MP4 回放、TacView 3D 复盘 |
| **② 成果展示** | 组会、阶段汇报、中期/结题验收的可视化材料 | 2D 战术态势图与动画、TacView 3D 回放（ACMI）、案例 GIF/MP4 |

> 训练路径零开销：默认 `render_mode=None` 时环境不做任何渲染调用，需要观察时再启用。

## 与 多任务智能体/ 的关系（单向数据接口）

```
多任务智能体/common/bvr_combat_env.py             可视化工具/visualization/
┌────────────────────────────────────┐          ┌──────────────────────────────┐
│ BVRCombatEnv.get_viz_frame()       │ ──帧字典─▶│ 2D 战术显示（实时/离屏）      │
│   唯一数据接口（ENU+SI，可 JSON）   │          │ CSV 录制与回放                │
│                                    │          │ GIF/MP4 渲染                  │
│ BVRCombatEnv.render(render_mode)   │ ◀惰性导入─│ ACMI 导出（TacView 3D）       │
└────────────────────────────────────┘  (Gym)   └──────────────────────────────┘
```

- 智能体库只产出数据（`get_viz_frame()` 帧字典，自包含绘制所需全部信息，含
  攻击区扫描角 `radar_az_limit_deg`）；本工具只呈现数据，两者通过帧格式解耦；
- `env.render()` 按 Gym 约定在首次调用时把本工具加入 `sys.path` 并惰性导入
  （未启用 render_mode 时零开销，不影响训练吞吐）；
- 本工具中 `tactical_core` / `offscreen` / `tactical_display` / `recorder` / `acmi` /
  `replay` / `csv2acmi` 均不依赖智能体库，可离线处理任意录制 CSV；
  仅 `demo.py` 驱动环境（唯一跨库依赖，用于成果展示与工具自检）。

## 目录结构

```
可视化工具\
├── README.md                ← 本文件
├── TacView观看指引.md        ← 组员 3D 回放上手指引
├── visualization\           ← Python 包（命令：python -m visualization.xxx）
│   ├── tactical_core.py     ← 绘图内核（后端无关：俯视态势 + 态势面板）
│   ├── tactical_display.py  ← 实时战术显示窗口（TkAgg，1 Hz 决策节奏刷新）
│   ├── offscreen.py         ← 离屏渲染：frame_to_rgb / render_episode（GIF/MP4）
│   ├── recorder.py          ← EpisodeRecorder：逐帧 CSV 记录/加载
│   ├── acmi.py              ← TacView ACMI 2.2 导出（自研实现）
│   ├── replay.py            ← 回放 CLI（CSV → GIF/MP4/ACMI）
│   ├── csv2acmi.py          ← 批量 CSV → ACMI CLI
│   └── demo.py              ← 演示脚本（脚本策略驱动环境，产出 CSV/GIF/ACMI）
├── tests\
│   └── test_visualization.py ← 20 用例（数据接口/录制/渲染/ACMI）
└── results\                 ← 运行产物（脚本自动创建，不入库）
```

## 快速开始（DC 环境，在 `可视化工具/` 目录下）

```powershell
# 实时窗口演示（脚本策略：接敌→双发齐射→F-pole 偏转→MAWS 规避）
python -m visualization.demo --seed 8

# 离线录制：CSV + GIF + ACMI（无窗口）
python -m visualization.demo --seed 8 --no-live --acmi

# 回放任一录制 CSV（GIF；--out results/demo.mp4 则输出 MP4，需 imageio-ffmpeg）
python -m visualization.replay results/demo_ep_seed8.csv --fps 10

# 任意录制 CSV → ACMI（TacView 3D 回放，见 TacView观看指引.md）
python -m visualization.csv2acmi results/demo_ep_seed8.csv
```

产物输出到 `可视化工具/results/`（`demo_ep_<tag>.csv/.gif/.acmi`）。

## 训练/评测中接入（自我调整）

```python
# 在 多任务智能体/ 目录下运行训练/评测脚本时：
from common.bvr_combat_env import BVRCombatEnv

env = BVRCombatEnv(config={"render_mode": "human"})   # 实时窗口；默认 None 零开销
obs, info = env.reset(seed=0)
for _ in range(240):
    obs, r, term, trunc, info = env.step(policy(obs, info))
    env.render()                                      # 1 Hz 刷新战术显示
    if term or trunc:
        break
env.close()
```

- `render_mode="human"`：实时战术显示窗口（仅单实例演示；**不要**在 SubprocVecEnv
  子进程中启用）；
- `render_mode="rgb_array"`：离屏帧 HxWx3 uint8（无显示设备可用，供评测脚本录制）；
- 训练期录制：任意脚本内用 `EpisodeRecorder` 逐帧采集（`rec.capture_env(env)`）→
  `rec.save_csv(path)`，事后用 `visualization.replay` / `csv2acmi` 离线回放对比。

## 单元测试

```powershell
# 在 可视化工具/ 目录下
python -m pytest tests/test_visualization.py -v      # 20 用例
```

覆盖：get_viz_frame 快照模式（键集/类型/几何一致性/JSON 可序列化）、事件标志、
render("rgb_array") 离屏帧、CSV 往返、GIF 渲染、绘图内核冒烟、ACMI 文件头与增强
（Coalition/Title、LockedTarget 建立与失锁、命中帧 Health=0 与 Kill、脱靶与末帧
冲刷 Miss）。

## TacView 3D 回放（acmi.py 导出 + csv2acmi 转换器）

`acmi.py` 导出 ACMI 2.2 文本（规范 https://www.tacview.net/documentation/acmi/en/ ），
TacView 免费版直接回放。已实现的增强：红蓝 `Coalition`、雷达 TRACK
`LockedTarget` 锁定连线（失锁空值移除 + Mode/Range）、命中帧 `Health=0`
爆炸、每弹 `Event=Timeout` 射击日志（Outcome:Kill/Miss，末帧在飞弹冲刷
记 Miss）、首帧初始距离 Bookmark、`Title` 带种子。

组员上手（详见 `TacView观看指引.md`）：TacView 已正式安装到本机，任意
`.acmi` 双击/拖入即看；CSV 录制文件经 `visualization.csv2acmi` 转成
同名 `.acmi`（支持一次多文件批量转换）。

## 演示脚本策略说明（demo.py）

仅为演示服务（非训练产物）。要点（均为实测踩坑记录）：

- **姿态闭环必需**：F-104 人工通道零输入不能保持高度（8 s 掉约 200 m）；
  升降舵符号与常规相反（正指令→低头）；滚转必须经坡度内环，否则 1 Hz
  零阶保持下失稳（phi ±180° 振荡→倒飞俯冲）。
- **F-pole 偏转是胜负手**：双方同时发射时我弹命中总晚 1~2 s（探针实测），
  发射后 crank 60° 背向偏转+下降可拖延敌弹拦截链。
- 10 种子验证：6 胜（enemy_hit）4 超时存活，0 坠毁 0 被命中。

## 依赖与许可

- 运行依赖仅 matplotlib + numpy（DC 环境已装）；GIF 用 PillowWriter（Pillow 自带）；
  MP4 需 `pip install imageio-ffmpeg`（自带 ffmpeg，无需系统安装）。
- ACMI 导出为自研实现（格式为 TacView 公开文本协议），仅借鉴思路，
  未拷贝 GPL 的 BVRGym TacviewLogger 代码；TacView 软件本身免费版可回放，
  军用单位使用需按官网要求另行授权（本模块不依赖 TacView 亦可完整工作）。
- 可选方案B（FlightGear 3D）未实施：JSBSim 原生 `FGOutputFG`/`data_output/flightgear.xml`
  就绪，需要时另立任务接入。
