# TacView 观看指引 —— BVR 空战 3D 回放（组员版）

用 TacView 免费版回放仿真导出的 ACMI 2.2 文本，获得专业级 3D 空战复盘：
红蓝阵营、雷达锁定连线、导弹轨迹、爆炸、射击日志。本指引覆盖：**启动、
任意录制文件如何可视化、界面操作**。

## 1. 启动 TacView

TacView 已正式安装到本机（官网下载），`.acmi` 文件自动关联，
**双击任何 .acmi 即开**；也可以从开始菜单启动 TacView，再
`File → Open` 或把文件拖入窗口。

标题栏显示"演示版（21天剩余）"属正常：试用期含高级功能，到期后自动转为
免费版，**回放 ACMI 文本文件不受影响**。

## 2. 任意文件可视化（使用中会产生多份文件时）

数据来源只有两种，分别一条路：

### 2a. 已有 .acmi 文件 → 直接看
- 安装后**双击**即开；或把文件**拖入** TacView 窗口；
  或 TacView 内 `File → Open`。
- 多份文件对比：每份独立打开一个 TacView 实例即可（程序支持多开）。

### 2b. CSV 录制文件（EpisodeRecorder 产物）→ 先转 .acmi
训练/评测积累的 `*.csv` 录制文件，用转换器生成同名 `.acmi`
（输出在 CSV 同目录，随后按 2a 打开）：

```powershell
# 在 多任务智能体/ 目录下，DC 环境
python -m visualization.csv2acmi visualization/results/demo_ep_seed8.csv
python -m visualization.csv2acmi a.csv b.csv c.csv      # 批量：一次多个
```

地理原点默认 lat0=30/lon0=120（与环境一致），非默认场景加
`--lat0/--lon0`；`--title` 可自定义标题。

### 2c. 代码里直接导出（写训练/评测脚本时）
```python
from visualization.acmi import export_acmi
export_acmi(frames, "run_042.acmi", title="PPO eval ep042")
```
`frames` 来自 `get_viz_frame()` 逐帧列表或 `EpisodeRecorder.frames`。

## 3. 界面看什么

| 画面元素 | 含义 |
| --- | --- |
| 蓝色标签 `Agent` | 我方 F-104（Coalition=Allies） |
| 红色标签 `ScriptAI` | 敌机（Coalition=Enemies） |
| 机间细线 | **雷达锁定连线**：一方雷达 TRACK 时指向对方，失锁即消失 |
| 小点+拖迹 | 在飞空空导弹（蓝=我方，红=敌方） |
| 火球/残骸 | 命中瞬间：被命中方 Health=0 + Destroyed 事件 |
| 顶部横幅 | 事件提示（发射 Message、Bookmark 初始距离、结局） |

## 4. 视角与时间轴操作

- **旋转**：左键拖动；**缩放**：滚轮；**平移**：右键拖动；
- **居中目标**：双击飞机标签，或对象列表选中后按 `F`；
- **播放/暂停**：空格；**逐帧**：←/→；**变速**：时间轴旁倍速按钮；
- **书签**：时间轴黄色刻度（首帧"Initial range xx km"、结局
  "Episode ends: ..."），单击跳转；
- **事件列表**：侧栏事件页签，双击任一事件镜头自动对准相关物体。

## 5. 射击日志（Shot Log）

每枚导弹消失时导出器写入一条 Timeout 事件，TacView 汇总为射击日志：

```
0,Event=Timeout|SourceId:100|AmmoType:AAM|AmmoCount:1|TargetId:200|Outcome:Kill
```

事件页签中筛选 `Timeout` 即可看到每次发射的结果（Kill/Miss）、发射方与
目标。示例 seed 8 共 4 条：我方 2 发（1 Kill + 1 Miss）、敌方 2 发（均 Miss）。

## 6. 常见问题

- **双击 .acmi 没反应** → 未安装 TacView；从开始菜单启动后
  `File → Open`，或把文件拖入窗口。
- **飞机在画面里很小** → 双击标签居中或滚轮放大；BVR 初始距离 30~80 km。
- **转换器报错"没有帧数据"** → 确认传入的是 EpisodeRecorder 保存的
  CSV（表头含 `own_missiles_json` 等列）。

演示截图可用 `python -m visualization.demo` 重新生成（产物在
`visualization/results/`）。
