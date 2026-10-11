# 09 — 真实数据、seed 与 neosurf 任务

## 目的与计划

将两任务共用的检索、patch、注册、IPA、碰撞和文件输出组成可替换的对象流程；使用真实公开预处理数据比较原函数，明确原始几何预处理与随机注册的验收边界。

## 实现与代码改动

- `design/data.py` 的 SurfaceRecord 拥有共索引的 geometry/descriptors/iface/邻居/原子；提供原目录与安全 NPZ adapter。目标可以仅有 flipped，源库必须有 straight。source 筛选读取原 NPY iface，target 位点读取 PLY iface，避免舍入改变门槛。
- `design/features.py` 保留 site patch 均值、平局/半值规则、strict cutoffs、四列 IPA 特征和原 CA-heavy 碰撞行为。距点下限 .5 Å、距原子下限 .25 Å；descriptor distance 无 epsilon。
- `design/registration.py` 定义最小 align 契约；ReplayRegistration 要求每个 key 的最终原坐标 pose，默认 fitness 为距离严格 <1 Å 的比例。Open3DRegistration 用现代 RANSAC+point-plane ICP，单列功能 gate。
- `design/workflow.py` 用组合复用两任务：match、search 和已对齐 score。输出 JSON、safe NPZ、一次坐标变换后的 PDB；保留候选序和拒绝原因。
- `tasks.py` 延迟注册全部 NN 与设计任务及上游名称别名，支持第三方 run() 对象；不新增无功能的 base/SeedNN/NeoSurfNN。
- CLI 增加 run/pack-design，保留已有命令。`examples/*.json` 提供任务配置；`pyproject.toml` design extra 和环境快照记录实际安装。
- `fetch_design_fixtures.py` 与 `tests/fixtures/design_members.json` 支持校验或下载 43 个固定公开成员；`smoke_design.py` 为 CPU/GPU 输出和现代对齐 gate。

## 数据来源与边界

[官方 Zenodo](https://zenodo.org/records/14225758) 10 GB archive 中选择 with_ligand 的 1A7X A/B 和 decoy 2GKW_B，另有部分 without_ligand 文件。43 件 size/CRC32/SHA256 核验通过，compressed payload 合计 11,863,313 bytes；整个 archive MD5 未验证。源 ZIP 有 32 位 offset wrap，按 archive order 恢复物理 offset，每个 local header 名称、长度、CRC 再独立核验。原计划另七件遇 HTTP 429 停止下载；交付 downloader 固定为已验证的 43 件，不请求缺失部分。

1A7X targets 有 PDB/PLY、iface、flipped descriptors、邻居，没有 NN input/rho/theta/mask。2GKW_B 则有 9/12 Å 完整 NN 输入。真实 site 对照使用有限 peptides 备份，search 使用固定有限历史 checkpoint；它们不是坏的主 checkpoint 的等价替身，未比较论文预测指标或宣称等同发布的描述符。

## 数值与运行结果

- subagent 工具直接 AST 提取上游原函数，注入旧 Open3D 名称到现代相同基础对象；**83/83 几何数组/诊断逐位相同**。覆盖两目标、源库、原目录 adapter、normal flip、outward shift、target sites、strict matches、四列 features、descriptor score、碰撞边界/early-abort。[证据](numerical_design_geometry.json)
- 真实 2GKW 665×100×5 完整 site 与 32×200×5 search 输入，CPU/GPU 每层、输出、loss、全部梯度 **4/4 默认严格门槛通过**。[证据](numerical_real_surfaces.json) 历史 search 的严格结果也通过；早先合成大尺度输入的梯度失败仍保留于原报告。
- CPU/GPU seed 固定姿态和 neosurf score 流程均输出概率、特征、NPZ 与 PDB；每组各一项通过，CLI run 也实际执行。[证据](numerical_design_workflows.json)
- 现代 Open3D 在独立合成已知 translation 上恢复，坐标最大误差 9.33e-15 Å，fitness=1。[同一证据](numerical_design_workflows.json)

## 未验证与数值边界

真实 smoke pose 为**构造的 translation**，阈值放宽只为输出覆盖；它不是旧 RANSAC 导出的姿态。旧 Open3D 的 max_validation=500 与现代 confidence 不同，随机 docking 最终姿态不能声称逐位一致。要对原实验做同姿态回归，需提供保存的 final matrix（合并初始随机旋转）及 fitness。

neosurf 可读取 protein+ligand 原表面，但本次未从原始 PDB/ligand 重新运行 MSMS/APBS/PyMesh 生成表面，也未运行 Rosetta grafting/refinement 或完整设计 benchmark。缺失几何工具不由未经验证的新表面算法替换。点重要性的原退化 NaN 保留并标记；完全相同描述符的无穷逆距离明确拒绝。以上范围已同步 README。
