# visualization/ —— BVR 空战可视化模块（步骤 2.4 交付物 D2.4-1）

2D 战术显示 + 回放：**俯视轨迹图**（双方位置+历史虚线）、**导弹轨迹动画**、
**攻击区半透明扇形**（R_max/R_nez/R_min，机头 ±60° 扫描扇区）、**态势面板**
（高度/速度/航向/油量/武器/敌我距离/RWR/MAWS 告警）、**CSV 记录与 GIF 回放**、
可选 **TacView ACMI 导出**。

## 架构与接口（全局统一）

```
BVRCombatEnv.get_viz_frame()   ← 唯一态势数据源（纯 Python dict，ENU+SI，可 JSON）
        │
        ├── TacticalDisplay      实时窗口（TkAgg，plt.ion，1 Hz 刷新）
        ├── offscreen            离屏渲染（FigureCanvasAgg，不碰全局后端）
        │      ├── frame_to_rgb      env.render("rgb_array")
        │      └── render_episode    帧序列 → GIF（PillowWriter，零依赖）/MP4
        ├── EpisodeRecorder      逐帧 CSV（导弹/事件列为 JSON 字符串）
        └── export_acmi          ACMI 2.2 文本（TacView 免费版 3D 回放）
```

环境侧启用方式：`BVRCombatEnv(config={"render_mode": "human" | "rgb_array"})`
后每步 `env.render()`；默认 `None`（训练/吞吐路径零开销）。实时渲染仅用于
单实例演示，**不要**在 SubprocVecEnv 子进程中启用。

## 快速开始（DC 环境，在 `多任务智能体/` 目录下）

```powershell
# 实时窗口演示（脚本策略：接敌→双发齐射→F-pole 偏转→MAWS 规避）
python -m visualization.demo --seed 8

# 离线录制：CSV + GIF + ACMI（无窗口）
python -m visualization.demo --seed 8 --no-live --acmi

# CSV 回放渲染（GIF/MP4）
python -m visualization.replay visualization/results/demo_ep_seed8.csv --fps 10
python -m visualization.replay xxx.csv --out xxx.mp4        # MP4 需 imageio-ffmpeg

# 任意 CSV 录制文件 → ACMI（TacView 3D 回放，见 TacView观看指引.md）
python -m visualization.csv2acmi visualization/results/demo_ep_seed8.csv
```

产物输出到 `visualization/results/`（`demo_ep_<tag>.csv/.gif/.acmi`）。

## TacView 3D 回放（export_acmi 增强 + csv2acmi 转换器）

`acmi.py` 导出 ACMI 2.2 文本（规范 https://www.tacview.net/documentation/acmi/en/ ），
TacView 免费版直接回放。已实现的增强：红蓝 `Coalition`、雷达 TRACK
`LockedTarget` 锁定连线（失锁空值移除 + Mode/Range）、命中帧 `Health=0`
爆炸、每弹 `Event=Timeout` 射击日志（Outcome:Kill/Miss，末帧在飞弹冲刷
记 Miss）、首帧初始距离 Bookmark、`Title` 带种子。

组员上手（详见 `TacView观看指引.md`）：免安装便携版在 `工具\Tacview\`
（`Tacview64.exe` 直接运行；官网安装包在项目根 `Tacview195Setup.exe`）。
使用中产生的任意 `.acmi` 双击/拖入即看；CSV 录制文件经
`visualization.csv2acmi` 转成同名 `.acmi`（支持一次多文件批量转换）。

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

## 单元测试

```powershell
python -m pytest 多任务智能体/tests/test_visualization.py -v   # 19 用例
```

覆盖：get_viz_frame 模式（键集/类型/几何一致性/JSON 可序列化）、事件标志、
render("rgb_array")、CSV 往返、GIF 渲染、绘图内核冒烟、ACMI 文件头与增强
（Coalition/Title、LockedTarget 建立与失锁、命中帧 Health=0 与 Kill、
脱靶与末帧冲刷 Miss）。
