# UAVDB 真实机动飞行数据（已下载）

## 基本信息

| 项目 | 内容 |
|------|------|
| 来源 | UAVDB（University of California, Davis，AIAA 飞行试验数据库），http://uavdb.org/cub-data.php |
| 平台 | 26% Cub Crafters CC11-100 Sport Cub S2 缩比模型（固定翼，全尺寸试验机） |
| 数据量 | 约 60 个机动段（多个架次），`数据\Cub_flight_data_mat.zip`（29.8 MB，未解压） |
| 格式 | .mat（MATLAB v5），用 MATLAB 或 Python `scipy.io.loadmat` 读取 |
| 引用 | UAVDB 网站要求引用其 AIAA 论文（见 uavdb.org/cub-pubs.php） |

## 文件命名规范（可直接仿照）

`data_<架次号>_<秒区间>_<机动名>_<构型>.mat`
例：`data_421_784-800_spin_no_flap_L_high_rate.mat` = 第421架次、784-800秒、螺旋机动、无襟翼、向左、高速率。

## 包含的机动清单（按动作名称）

| 机动 | 英文名 | 数据片段数 | 说明 |
|------|--------|:---:|------|
| 怠速下滑 | idle_descent | 7 | 收油门下滑，含 trim 变体与半襟翼构型 |
| 副翼响应 | ail_resp | 5 | 左/右/左右交替副翼阶跃响应 |
| 升降舵响应 | elev_resp | 6 | 拉杆/推杆/高低速率响应 |
| 方向舵响应 | rud_resp | 9 | 短时/长时方向舵输入响应 |
| 长周期运动 | phugoid | 3 | 无襟翼/全襟翼，推杆与失速进入 |
| 失速/深度失速 | stall / deep_stall | 12 | 无/半/全襟翼，高低速率 |
| 螺旋 | spin | 4+ | 左/右螺旋，半襟翼构型 |
| 滚转 | roll | 1 | 全滚转机动 |
| 破S机动 | split_S | 1 | 半滚倒转 |
| 着陆 | landing | 3 | 半襟翼着陆段 |

## 与本项目的关系

1. **机动动作定义参考**：真实试飞中对"失速/螺旋/破S/长周期"等动作的构型与进入方式，可映射到本项目 JSBSim 数据采集清单（《05》§六）；
2. **响应特性参照**：升降舵/副翼/方向舵阶跃响应的数据形态，是校验 JSBSim 模型（以及我方采集的 F-104 数据）合理性的对照样本；
3. **机动识别数据基础**：文件名即标签（机动名+构型+方向），可直接用于机动识别/行为克隆的参考格式设计。

## 读取示例（Python）

```python
import scipy.io as sio
m = sio.loadmat("data_421_784-800_spin_no_flap_L_high_rate.mat")
print(m.keys())          # 查看变量名（时间、状态量、舵面等）
```
> 需先解压 zip。若未安装 scipy：`pip install scipy`。
