# JSBSim 使用教程

**适用版本：** JSBSim 1.2.4（本项目使用 Windows 安装包，位于 `大创 尝试\JSBSim\`）
**读者：** 项目组成员（唐煜皓、李欣洋、焦阳、徐文强）

---

## 1. JSBSim 是什么

JSBSim 是一个开源（LGPL）的六自由度飞行动力学仿真库/程序。本项目用它作为底层飞行动力学引擎：
- 提供 60 型飞行器模型（本项目选用 **F-104** 主选、**F-4N** 备选，模拟老旧平台）；
- 求解刚体六自由度运动方程 + 气动/发动机/起落架/飞控系统模型；
- 可通过 XML 脚本与配置文件全流程控制，输出 CSV 飞行数据；
- 提供 C++ 库（`JSBSim.lib`）与 Python 绑定（`pip install jsbsim`，与本安装包等价），供 Gym 环境桥接调用（阶段二步骤 2.3）。

## 2. 安装与配置

### 2.1 已安装情况（本机）

- 安装目录：`大创 尝试\JSBSim\`（即下文所有命令的 `--root` 指向的目录）；
- 免编译：直接使用 `JSBSim.exe`，依赖的 `msvcp140.dll` 已随包提供；
- 无需注册环境变量：所有运行都在项目目录内通过命令行完成；
- Python 环境：Python 3.14.2（用 `py` 命令启动），已装 pandas/numpy/matplotlib。

### 2.2 组员新电脑安装步骤

1. 把整个 `JSBSim\` 文件夹（约几百 MB）复制到目标电脑；
2. 双击 `JSBSim.exe --version` 验证可运行（Win10/11 自带所需运行时；若报缺 DLL，安装 VC++ 2015-2022 运行库）；
3. 安装 Python 3.10+ 并执行 `pip install pandas numpy matplotlib`（仅数据处理需要；也可选装 `pip install jsbsim` 获得 Python 绑定，用于后续 Gym 桥接）；
4. 确认 `py --version` 可用。

### 2.3 目录结构（本项目安装包）

```
JSBSim\
├── JSBSim.exe          ← 主程序（命令行仿真器）
├── lib\JSBSim.lib      ← C++ 静态库（后续 Gym 桥接用）
├── include\            ← C++ 头文件
├── matlab\             ← MATLAB 接口
├── aircraft\           ← 60 型飞行器模型（每型一个目录）
│   ├── f104\           ← F-104C：f104.xml + Systems\（15 个系统文件）
│   │   ├── reset_cruise.xml          ← 【本项目新增】巡航初始化文件
│   │   └── Systems\autopilot-hold.xml ← 【本项目新增】姿态/高度/速度保持自动驾驶仪
│   ├── F4N\            ← F-4N：F4N.xml + Engines\
│   └── f16\            ← F-16 参考模型（f16.xml + reset00.xml）
├── engine\             ← 90 个发动机定义（J79-GE-11A、F100-PW-229 等）
├── systems\            ← 30 个通用系统组件（传感器/自动驾驶仪模板等）
├── scripts\            ← 63 个仿真脚本（f16_test.xml 等）
│   ├── f104_cruise.xml     ← 【本项目新增】F-104 巡航脚本
│   ├── f104_maneuver.xml   ← 【本项目新增】F-104 机动脚本
│   └── ap_debug.xml        ← 【本项目新增】自动驾驶仪调试脚本
├── data_output\        ← 10 个数据输出指令模板（position.xml、fcs.xml 等）
└── unins000.exe        ← 卸载程序（勿删其他文件）
```

## 3. 基本操作流程（命令行）

标准调用格式：

```
JSBSim.exe [脚本名] [输出文件名] <选项>
```

### 3.1 常用选项一览

| 选项 | 作用 | 示例 |
|------|------|------|
| `--root=<路径>` | 指定 JSBSim 根目录（aircraft/engine/scripts 所在目录） | `--root=.\JSBSim` |
| `--aircraft=<机型>` | 指定机型目录名 | `--aircraft=f104` |
| `--script=<脚本>` | 指定运行脚本 | `--script=scripts\f104_cruise.xml` |
| `--initfile=<初始化文件>` | 指定初始状态文件（相对机型目录） | `--initfile=reset_cruise` |
| `--end=<秒>` | 指定仿真结束时间 | `--end=10` |
| `--simulation-rate=<值>` | 积分步长：<1 为 dt 秒，>1 为频率 Hz | `--simulation-rate=120` |
| `--property=<名=值>` | 启动时设定任意属性 | `--property=ic/alpha-deg=3` |
| `--outputlogfile=<文件>` | 覆盖脚本中的输出文件名 | `--outputlogfile=test.csv` |
| `--logdirectivefile=<文件>` | 加载额外的输出指令文件（可多次） | `--logdirectivefile=data_output\position.xml` |
| `--realtime` | 按真实时间运行（演示用） | — |
| `--catalog[=机型]` | 打印该机型的全部属性清单（查属性名利器） | `--catalog=f104` |
| `--version` / `--help` | 版本 / 帮助 | — |

> 注意：`=` 前后不能有空格。属性名必须用 `--catalog` 或本教程 §6 验证过。

### 3.2 三种运行方式

**方式 A：脚本运行（推荐，全自动）**
```
& ".\JSBSim\JSBSim.exe" --root=".\JSBSim" --script="scripts\f104_cruise.xml"
```

**方式 B：命令行组合运行（快速试验）**
```
& ".\JSBSim\JSBSim.exe" --root=".\JSBSim" --aircraft=f104 --initfile=reset_cruise --end=10 --simulation-rate=120
```

**方式 C：交互式运行（手工学习用）**
不带任何参数启动后按提示输入，适合教学演示，不适合批量采集。

## 4. 仿真脚本（run script）说明

脚本文件是仿真运行的"总导演"，结构如下（完整示例见 `scripts\f104_cruise.xml`）：

```xml
<?xml version="1.0"?>
<runscript name="脚本名">
  <description>……</description>
  <use aircraft="f104" initialize="reset_cruise"/>   <!-- 机型 + 初始状态 -->
  <output name="输出.csv" type="CSV" rate="10">      <!-- 数据记录指令 -->
    <property> position/h-sl-ft </property>
    <property> velocities/vtrue-fps </property>
    ...
  </output>
  <run start="0.0" end="60.0" dt="0.00833333">       <!-- 仿真区间 + 步长 -->
    <event name="事件名">                             <!-- 事件：条件触发、执行动作 -->
      <notify/>                                       <!-- 触发时打印日志 -->
      <condition> simulation/sim-time-sec >= 10 </condition>
      <set name="fcs/ap-roll-setpoint-rad" value="0.5236"/>
    </event>
    ...
  </run>
</runscript>
```

### 4.1 `<use>`：机型与初始状态
- `aircraft` = `aircraft\` 下的目录名；
- `initialize` = 该机型目录下的初始化文件名（不含 .xml）。

### 4.2 `<run>`：仿真参数
- `start`/`end`：仿真时间区间（秒）；
- `dt`：积分步长（秒）。**本项目统一 dt = 1/120 = 0.00833333 s**（固定步长 RK4/Euler 积分，保证确定性）。数值精度与速度平衡，勿随意改大（>1/60 可能出现积分不稳定）。

### 4.3 `<event>`：事件系统（本项目主要控制手段）
- `<condition>` 内是逻辑表达式，支持 `==`、`>=`、`<=`、`lt`、`gt`、`and`、`or`；
- `<set name="属性名" value="值"/>` 在触发瞬间写入属性；
- 可选 `action="FG_RAMP" tc="秒"` 让参数按时间斜坡变化（模拟柔和操纵）；
- 可设置 `persistent="true"` 让事件反复触发；
- `<notify/>` 会在控制台打印触发时刻（验证事件时序用）。

### 4.4 `<output>`：数据输出指令（详见 §7）

## 5. 配置文件说明

| 文件类型 | 位置 | 作用 | 本项目改动 |
|---------|------|------|-----------|
| 机型主配置 `f104.xml` | `aircraft\f104\` | 质量/气动/发动机/系统声明 | 已注册新系统 autopilot-hold |
| 初始化文件 `reset_*.xml` | 机型目录内 | 初始位置、速度、姿态 | 新增 `reset_cruise.xml` |
| 系统文件 `Systems\*.xml` | 机型目录内 | 飞控/起落架/自动驾驶仪等系统 | 新增 `autopilot-hold.xml`；修改 `FCS-pitch/roll.xml`、`radar.xml` |
| 发动机 `engine\J79-GE-11A.xml` | `engine\` | 推力曲线/油耗/加力特性 | 无 |
| 输出指令模板 | `data_output\` | 预定义的输出属性组合 | 无 |

### 5.1 初始化文件（本项目 `reset_cruise.xml`）

```xml
<initialize name="reset_cruise">
  <ubody unit="FT/SEC">   898.5  </ubody>   <!-- 体轴 x 速度（898.5 fps ≈ M0.85 @15000ft） -->
  <vbody unit="FT/SEC">     0.0  </vbody>
  <wbody unit="FT/SEC">     0.0  </wbody>
  <latitude unit="DEG">    30.0  </latitude>
  <longitude unit="DEG">  120.0  </longitude>
  <phi unit="DEG">          0.0  </phi>
  <theta unit="DEG">        0.0  </theta>
  <psi unit="DEG">          0.0  </psi>
  <altitude unit="FT">  15000.0  </altitude>
  <gamma unit="DEG">        0.0  </gamma>
</initialize>
```
常用字段：`ubody/vbody/wbody`（体轴速度）、`alpha`/`beta`（迎角/侧滑角，与速度二选一）、`altitude`、`latitude`/`longitude`、`phi/theta/psi`（滚转/俯仰/偏航角，注意 JSBSim 的 `phi` 就是滚转角）、`gamma`（航迹角）。**JSBSim 内部全用英制（ft、fps、slug、lb），初始化与配置均为英制。**

### 5.2 系统文件（本项目新增 autopilot-hold.xml 摘要）

本系统给 F-104 增加了三个保持通道，通过脚本属性控制：

| 属性 | 含义 | 取值 |
|------|------|------|
| `fcs/ap-master-on` | 自驾仪总开关 | 0 关 / 1 开 |
| `fcs/ap-alt-hold-on` | 高度保持开关 | 0 关 / 1 开 |
| `fcs/ap-alt-setpoint-ft` | 目标高度 | ft |
| `fcs/ap-speed-setpoint-fps` | 目标真空速 | fps |
| `fcs/ap-roll-setpoint-rad` | 目标滚转角 | rad（正=右滚） |
| `fcs/ap-pitch-setpoint-rad` | 目标俯仰角（高度保持关闭时生效） | rad |
| `fcs/throttle-cmd-pilot` | 人工油门指令（与自驾仪速度通道求和） | 0~1 |

> 该系统已为后续 Gym 环境铺路：RL 动作可映射为（油门、滚转角指令、俯仰角指令），由自驾仪执行内环姿态控制——这正是阶段三 JSRL 引导策略需要的内层控制基础。

## 6. 常用参数设置（属性速查）

> 属性名务必用 `--catalog=f104` 核对。本项目已核对的关键属性：

| 属性 | 含义 | 单位 |
|------|------|------|
| `position/h-sl-ft` / `h-agl-ft` | 海拔高度 / 离地高度 | ft |
| `position/lat-geod-deg` / `long-gc-deg` | 纬度 / 经度 | deg |
| `position/distance-from-start-lat-mt` | 距起点北向/东向位移 | m |
| `velocities/vtrue-fps` / `mach` | 真空速 / 马赫 | fps / - |
| `velocities/u/v/w-fps` | 体轴三轴速度 | fps |
| `velocities/v-north/east/down-fps` | 北东地速度 | fps |
| `velocities/p/q/r-rad_sec` | 三轴角速度 | rad/s |
| `attitude/phi/theta/psi-rad` | 欧拉角 | rad |
| `aero/alpha-rad` / `beta-rad` | 迎角 / 侧滑角 | rad |
| `accelerations/Nx/Ny/Nz` | 过载（体轴，单位 g） | g |
| `fcs/throttle-cmd-norm` | 油门（0~1，>1 为加力区） | - |
| `fcs/elevator/aileron/rudder-cmd-norm` | 舵面指令（-1~1） | - |
| `fcs/ap-*` | 本项目自动驾驶仪属性（§5.2） | - |
| `propulsion/engine/thrust-lbs` | 发动机推力 | lb |
| `propulsion/total-fuel-lbs` | 总燃油 | lb |
| `propulsion/tank[n]/contents-lbs` | 各油箱油量 | lb |
| `propulsion/cutoff_cmd` | 供油开关（0=供油，1=切断） | - |
| `propulsion/starter_cmd` | 起动机 | 0/1 |
| `propulsion/engine/n1` / `n2` | 发动机转速（可写） | % |
| `propulsion/engine/set-running` | 一键置为运行状态 | 0/1 |
| `gear/gear-cmd-norm` | 起落架指令（0=收起，1=放下） | - |
| `fcs/speedbrake-cmd-norm` | 减速板 | 0~1 |
| `fcs/flap-cmd-norm` | 襟翼 | 0~1 |
| `inertia/mass-slugs` | 质量 | slug |
| `atmosphere/T-R` / `rho-slugs_ft3` | 大气静温 / 密度 | R / slug/ft³ |
| `simulation/sim-time-sec` | 仿真时间 | s |

**本项目脚本的标准起飞状态设置（四个事件缺一不可，照抄即可）：**
```xml
<!-- 1. 发动机一键运行（J79 涡喷必须切断供油挡才能出推力） -->
<set name="propulsion/engine[0]/set-running" value="1"/>
<set name="propulsion/cutoff_cmd" value="0"/>
<set name="propulsion/engine[0]/n2" value="100"/>
<set name="propulsion/engine[0]/n1" value="100"/>
<!-- 2. 收轮（否则巨大阻力，飞机会坠） -->
<set name="gear/gear-cmd-norm" value="0"/>
<set name="fcs/speedbrake-cmd-norm" value="0"/>
<set name="fcs/flap-cmd-norm" value="0"/>
<!-- 3. 设置采集燃油配置 -->
<set name="propulsion/tank[0]/contents-lbs" value="3500"/>
<set name="propulsion/tank[1]/contents-lbs" value="600"/>
<set name="propulsion/tank[2]/contents-lbs" value="600"/>
<!-- 4. 接通自驾仪 -->
<set name="fcs/ap-master-on" value="1"/>
<set name="fcs/ap-alt-hold-on" value="1"/>
<set name="fcs/ap-alt-setpoint-ft" value="15000"/>
<set name="fcs/ap-speed-setpoint-fps" value="750"/>
```

## 7. 导出飞行数据的方法

### 7.1 方法一：脚本内嵌 `<output>`（推荐）

```xml
<output name="f104_cruise_raw.csv" type="CSV" rate="10">
  <property> simulation/sim-time-sec </property>
  <property> position/h-sl-ft </property>
  ...
</output>
```
- `rate` = 输出频率（Hz），必须能整除积分频率（本项目 120 Hz，可取 1/10/20/30/40/60/120）；
- 输出文件名相对 JSBSim 根目录；
- 生成文件第一列为 `Time`（仿真秒），随后每列一个属性；本版本（1.2.4）列名带 `/fdm/jsbsim/` 前缀。

### 7.2 方法二：命令行 `--logdirectivefile`

复用 `data_output\` 下的模板（如 `position.xml`），与脚本内 `<output>` 可同时生效：
```
JSBSim.exe --root=.\JSBSim --script=scripts\f104_cruise.xml --logdirectivefile=data_output\position.xml
```

### 7.3 方法三：`--outputlogfile` 覆盖输出文件名（批量采集时改文件名用）

### 7.4 格式转换与重采样（本项目工具）

原始 CSV 为英制+前缀列名，不适合直接进训练代码。使用本项目工具：
```
py JSBSim_培训与验证资料\tools\convert_jsbsim_csv.py JSBSim\f104_cruise_raw.csv 样例数据\f104_cruise_si.csv      # 转 SI
py JSBSim_培训与验证资料\tools\convert_jsbsim_csv.py JSBSim\f104_cruise_raw.csv 样例数据\f104_cruise_si_1hz.csv 1  # 转 SI + 1Hz 重采样
```
输出文件同时保留 SI 列（`h_sl_m`、`vtrue_mps`…）与英制原值列（`position_h-sl-ft`…），列名对应关系见《03 字段字典》。

## 8. 完整操作示例（五分钟上手）

```powershell
# 1. 验证安装
& ".\JSBSim\JSBSim.exe" --version

# 2. 运行 F-104 巡航 120 秒（约 0.3 秒完成）
& ".\JSBSim\JSBSim.exe" --root=".\JSBSim" --script="scripts\f104_cruise.xml"

# 3. 运行 F-104 机动 60 秒（盘旋+爬升）
& ".\JSBSim\JSBSim.exe" --root=".\JSBSim" --script="scripts\f104_maneuver.xml"

# 4. 转换数据
py "JSBSim_培训与验证资料\tools\convert_jsbsim_csv.py" "JSBSim\f104_cruise_raw.csv" "JSBSim_培训与验证资料\样例数据\f104_cruise_si.csv"

# 5. 查看数据
py -c "import pandas as pd; print(pd.read_csv(r'JSBSim_培训与验证资料\样例数据\f104_cruise_si.csv').head())"

# 6. 查某机型全部属性名
& ".\JSBSim\JSBSim.exe" --root=".\JSBSim" --aircraft=f104 --catalog=f104
```

## 9. 常见问题排查（本项目实战经验）

### 9.1 飞机无推力 / 一直掉高度
- **原因 1：发动机没起动。** J79 涡喷的 `cutoff_cmd` 默认为 1（断油），且 N2 需要上转。最快方案：`set-running=1` + `cutoff_cmd=0` + 直接写 `n2=100`（§6 配方）。
- **原因 2：起落架没收起。** F-104 默认轮子放下，附加阻力系数 0.06，基本飞不动。必须 `gear-cmd-norm=0`。
- **原因 3：超重低能。** 满油 20010 lb 时 15000 ft 巡航需要频繁加力。数据采集统一用 4700 lb 燃油配置。

### 9.2 输出 CSV 里某些列为空或报 "No property by the name ... has been defined"
- 属性名写错（本版本严格区分 `v-north-fps`（对）与 `vn-fps`（错）等）；
- 用 `--catalog=f104` 核对后再写。

### 9.3 FATAL ERROR：Property xxx does not exist
- 模型文件缺陷（如 f104 的 `radar.xml` 引用不存在的 `systems/radar/range`）。修复方法：在该系统文件顶部加 `<property value="80.0">systems/radar/range</property>` 声明即可。修复后应记录到项目文档（本项目已修复并记录）。

### 9.4 飞机坠地 / GEAR_CONTACT 刷屏
- 仿真到地平面，通常是无推力+无操纵的滑降。按 9.1 排查；或提高初始高度/接通自驾仪。

### 9.5 自动驾驶仪输出不动（积分不起作用）
- JSBSim 的 `<pid>` 组件中 `<trigger>` 非零时**冻结积分器**（只保留 P 项）。本项目 AP 因此不用 trigger，改用输入端 switch 把误差置零来防积分饱和（见 `autopilot-hold.xml` 顶部注释）。

### 9.6 想调试系统内部信号
- 把中间量（如 `fcs/ap-elevator-cmd`、`fcs/ap-pitch-input`）加进 `<output>`，参照 `scripts\ap_debug.xml`。

## 10. 进阶：Python 绑定（阶段二步骤 2.3 桥接预告）

JSBSim 提供官方 Python 包（`pip install jsbsim`），API 与本安装包一致，核心接口：
```python
import jsbsim
fdm = jsbsim.FGFDMExec(root_dir="JSBSim")
fdm.load_model("f104")
fdm.set_property_value("ic/h-sl-ft", 15000)   # 或加载 reset 文件
fdm.run_ic()
fdm["fcs/throttle-cmd-norm"] = 0.8            # 写属性
fdm.run()                                      # 步进一次 dt
print(fdm["position/h-sl-ft"])                 # 读属性
```
阶段二步骤 2.3 的 `JSBSimGymBridge` 即基于此封装 `step/reset`。也可以 subprocess 调用 `JSBSim.exe` + CSV 交换数据（简单但慢），或 ctypes 链接 `JSBSim.lib`（复杂但最快）。三选一在步骤 2.3 定稿，本验证已确认三者均可行。
