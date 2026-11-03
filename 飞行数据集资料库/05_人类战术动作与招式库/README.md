# 05 人类战术动作与招式库

本目录用于把“人类飞行/空战知识”整理成可执行的 `ManeuverSpec`，供引导行为树、
JSRL 教师和专家智能体使用。**数据用于研究与教学，不重新分发第三方原始文件。**

## 目录

```text
05_人类战术动作与招式库/
├── 01_手册与教材/          ← 手册、论文、招式说明
├── 02_真实轨迹/            ← UAVDB、CAFUC、MIT 等真实/训练轨迹（待补）
├── 03_仿真与ACMI/
│   └── DCS_BVR/            ← DCS 公共 ACMI 训练记录 + manifest
├── 04_参考代码/            ← BVRGym、CloseAirCombat、WEZ 等参考仓库
│   └── _incomplete/        ← 未下载完整的归档
├── 04_标注与招式表/
│   └── maneuver_catalog_v1.csv
└── tools/
    └── fetch_dcs_acmi.py   ← ACMI 索引爬取/筛选/断点下载/整理
```

## 数据来源与使用边界

| 来源 | 用途 | 注意 |
|---|---|---|
| DCS 公共 ACMI（jedi.gildia.org） | BVR 训练轨迹模式提取 | 社区/游戏数据，非真实飞行；不要重新分发 |
| MIT AIA Maneuver ID | 18 个基础机动的标准描述/参数 | 需 Data Sharing Agreement，不得公开转发 |
| F-16 MCH 11-F16 Vol 5 | BFM/ACM/Intercept 战术定义 | 公开手册；用于战术定义，不直接复制 F-104 参数 |
| UAVDB | 真实基础机动/响应数据 | 需引用 UAVDB/AIAA 论文 |
| CAFUC / CAFUC2 | 真实训练机动轨迹 | IEEE DataPort 账号与数据集条款 |
| WEZ / AirCombat-WEZ | Rmax/NEZ 包线校验 | 开放数据；不是招式库 |
| BVRGym / CloseAirCombat | 环境/行为树/接口参考 | 逐个检查许可证；BVRGym 为 GPL-3.0 |

## ACMI 下载工具

```bash
# 只列出 BVR 相关文件
python 飞行数据集资料库/05_人类战术动作与招式库/tools/fetch_dcs_acmi.py list \
  --include "bvr|BFM|F-16|F16|F-18|F18" --size

# 批量下载 BVR 相关文件（默认限制 200MB/文件，可断点续传）
python .../fetch_dcs_acmi.py download --include "bvr" --max-size-mb 80

# 整理已有手动下载文件（复制到 DCS_BVR 并生成 manifest）
python .../fetch_dcs_acmi.py import "新增资料"
```

脚本特性：

- 递归读取公开目录索引，按文件名/日期/大小筛选；
- `--dry-run` 只列出，不下载；
- `.part` 断点文件 + 重试；
- `manifest.csv/json` 记录文件大小、SHA256、来源 URL、状态；
- 默认串行、限速，避免对站点造成压力。

## 当前 ACMI 状态

- `03_仿真与ACMI/DCS_BVR/` 当前约 47 个 `.acmi`，约 300 MB；
- `manifest.csv` 记录 47 个文件；
- 已包含 2023/2025/2026 的 BVR/训练记录；
- 站点索引偶尔超时，可重复运行 `download`，已存在文件会自动跳过。

## 下一步

1. 用 `maneuver_catalog_v1.csv` 把 10–15 个核心招式落到 `ManeuverSpec`；
2. 从 ACMI 提取 `crank / notch / launch-and-crank / extend` 的轨迹片段；
3. 为每个招式补前置条件、参数、退出条件、安全边界和固定 seed 验证；
4. 接入 `GuideStep` / `BTGuideProvider`，再做 80–120 km / 600 步验证。
