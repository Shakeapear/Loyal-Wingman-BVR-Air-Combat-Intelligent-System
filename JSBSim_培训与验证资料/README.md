# JSBSim 培训与验证资料包

本目录是项目 JSBSim 仿真工作的完整技术资料，供全组使用。五份文档 + 数据样例 + 工具脚本，按需查阅。

## 文件索引

| 文件/目录 | 内容 | 主要读者 |
|-----------|------|---------|
| `01_JSBSim技术验证方案.md` | 验证目标、8 项验证步骤、测试条件、预期结果、通过标准，附首轮实测结果 | 指导教师、李欣洋 |
| `02_JSBSim使用教程.md` | 安装配置、命令行操作、脚本/配置文件说明、常用参数、数据导出方法、排障手册 | 唐煜皓（主）、全体 |
| `03_飞行数据样例与字段字典.md` | 6 个样例文件说明、47 列字段含义与单位、采样率与格式、场景时间线 | 李欣洋、焦阳 |
| `04_数据采集规范与模板.md` | 派发采集任务的必给信息清单、标准配置、命名/目录规范、质量检查、场景卡与记录表模板 | 徐文强（主）、全体 |
| `05_训练数据采集与样例详解.md` | 训练轨迹格式（trajectory_v1）逐部分讲解、采集流程、采集清单与储备计划 | 全体（采集执行人必读） |
| `样例数据\` | 巡航/机动 SI 样例 + **训练轨迹样例 f104_demo_ep0001.csv**（180 步 6 阶段机动） | 全体 |
| `tools\convert_jsbsim_csv.py` | 原始 CSV → SI/1Hz 标准格式转换工具 | 全体 |
| `tools\make_trajectory.py` | 原始 CSV → 训练轨迹文件（trajectory_v1）生成工具 | 全体 |

## 五分钟验证（复现全部成果）

```powershell
# 1. 引擎自检
& ".\JSBSim\JSBSim.exe" --version

# 2. 运行标准场景（各 <1 秒完成）
& ".\JSBSim\JSBSim.exe" --root=".\JSBSim" --script="scripts\f104_cruise.xml"
& ".\JSBSim\JSBSim.exe" --root=".\JSBSim" --script="scripts\f104_training_demo.xml"

# 3. 生成训练轨迹文件（智能体训练数据）
py "JSBSim_培训与验证资料\tools\make_trajectory.py" "JSBSim\f104_training_demo_raw.csv" "JSBSim_培训与验证资料\样例数据\f104_demo_ep0001.csv" --episode f104_demo_ep0001 --scenario S3 --task flight_skill_demo --collector 唐煜皓

# 4. 检查（元数据自动跳过）
py -c "import pandas as pd; d=pd.read_csv(r'JSBSim_培训与验证资料\样例数据\f104_demo_ep0001.csv', comment='#'); print(d.groupby('phase_label').size())"
```

## 本项目对 JSBSim 的改动清单（重要，勿丢失）

| 文件 | 改动 | 原因 |
|------|------|------|
| `JSBSim\aircraft\f104\reset_cruise.xml` | 新增巡航初始化（15000 ft / M0.85） | 数据采集标准初始状态 |
| `JSBSim\aircraft\f104\Systems\autopilot-hold.xml` | 新增姿态/高度/速度保持自驾仪 | 数据采集 + 后续 Gym 底层控制 |
| `JSBSim\aircraft\f104\Systems\FCS-pitch.xml` | 俯仰回路加入自驾仪增量 | 配合 autopilot-hold |
| `JSBSim\aircraft\f104\Systems\FCS-roll.xml` | 滚转回路加入自驾仪增量 | 配合 autopilot-hold |
| `JSBSim\aircraft\f104\Systems\radar.xml` | 补齐缺失属性 `systems/radar/range` | 修复模型加载崩溃 |
| `JSBSim\aircraft\f104\f104.xml` | 注册 autopilot-hold 系统 | 配合 autopilot-hold |
| `JSBSim\scripts\f104_cruise.xml` | 新增巡航脚本 | 标准采集场景 1 |
| `JSBSim\scripts\f104_maneuver.xml` | 新增机动脚本（盘旋+爬升） | 标准采集场景 2 |
| `JSBSim\scripts\f104_training_demo.xml` | 新增六阶段训练轨迹演示脚本 | 训练数据采集模板 |
| `JSBSim\scripts\ap_debug.xml` | 新增自驾仪调试脚本 | 系统调试模板 |

> 备份提醒：`JSBSim\` 目录整体复制即完成环境迁移；上述改动已包含在目录内，无需单独备份。
