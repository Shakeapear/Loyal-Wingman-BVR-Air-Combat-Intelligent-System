# CAFUC 民航训练机动数据集（需免费账号下载）

## 基本信息

| 项目 | 内容 |
|------|------|
| 名称 | CAFUC Flight Maneuver Dataset |
| 来源 | IEEE DataPort，DOI: 10.21227/19xm-dn61，页面：https://ieee-dataport.org/documents/cafuc-0 |
| 数据 | 2023 年民航飞行学院（CAFUC）真实训练飞行，C172S / SR20 固定翼 |
| 规模 | 150,000 帧（1 Hz，共 41.6 小时），64 维特征，14,356 个基础机动，168 名飞行学员 |
| 文件 | `dataset_cafuc.tar.gz`（仅 813 KB，压缩率很高） |
| 组织方式 | 按机动轨迹几何形状分为 4 类文件夹：**Eight_Turn（八字盘旋）、Rectangle（矩形航线）、Steep_Turn（大坡度盘旋）、TearDrop（泪滴/过顶转弯）** |

## 下载步骤（组员 5 分钟）

1. 打开 https://ieee-dataport.org/documents/cafuc-0 ；
2. 右上角 "Create Free Account" 注册（学校邮箱即可，免费）；
3. 登录后返回该页面，点击 "LOGIN TO ACCESS DATASET FILES" 下方的 `dataset_cafuc.tar.gz` 下载；
4. 解压后放入本文件夹 `数据\` 子目录，并在下面登记。

> 同作者还有 CAFUC2（进阶版），页面 https://ieee-dataport.org/documents/cafuc2 ，可一并下载。

## 数据用途（本项目）

1. **机动识别数据基础**：4 类机动 + 人工标签 + 64 维特征（含位置/姿态/发动机参数），是机动识别模型的标准数据样例；
2. **真实训练机动轨迹参考**：与 JSBSim 采集的 `trajectory_v1` 轨迹对照，理解真实飞行数据的噪声水平与量纲分布；
3. **组员范例**：文件名按机动类型分文件夹组织，可参考其分类思路组织本项目采集数据。

## 下载登记

| 日期 | 姓名 | 是否成功 | 备注 |
|------|------|---------|------|
| （示例）2026-08-16 | 唐煜皓 | 待下载 | 需先注册 IEEE DataPort |
