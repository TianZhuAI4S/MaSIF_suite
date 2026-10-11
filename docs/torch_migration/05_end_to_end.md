# 05 — 安装与运行验收

## 目的与计划

提供可重建安装流程，验证目标环境不依赖 TF，执行预计算数据到模型训练/预测/输出的完整入口，并核对原始 PDB 到 patch 的外部几何依赖是否可用。

## 实现与代码改动

- `torch/pyproject.toml`：独立 editable package、`masif-torch` 命令，不把本目录设成 Python `torch` 包；核心依赖只有 Torch、NumPy、打包 ligand 所需 SciPy。
- `torch/environments/`：两套 Conda 环境、runtime 精确版本与 pip freeze 快照。Torch 锁文件剔除指向未包含本次改动的上游 editable Git 地址，本地包显式安装。
- `torch/README.md`：沿原 README 结构，新系统要求、CPU/GPU 安装、数据准备、转换、三应用、Score、续训、数值验收、PyMOL、限制与引用。
- `torch/INSTALL_TORCH.md`：短安装入口。
- `torch/tools/smoke_workflows.py` 和对应 JSON：可重跑的 NN/data 入口验收。

## 已执行验证

在本机实际创建新的 masif_nn 与 masif_torch 后安装依赖和本地包，TF `2.21.0` 可读取/执行有效旧图；Torch `2.14.1+cu126` CPU 和 GPU 可运行。`python -m masif_torch doctor` 确认无 TF；`pip check` 无冲突。

NN 层面的端到端链路：独立 TF checkpoint → export → Torch checkpoint → predict NPY；NPZ/原 surface/cache → train → save → resume → evaluate，均有通过记录。命令和误差见 [04](04_workflows.md)、[numerical_workflows.json](numerical_workflows.json)。11 unit tests 全部通过，无 skip。

## 未通过的完整端到端验收项

| 项目 | 实际情况 |
|---|---|
| 原始 PDB → 几何表面 → 特征/patch → Torch 输出 | MSMS/reduce/APBS/multivalue/PDB2PQR/PyMesh 无完整可运行安装，未执行 |
| 真实预计算 protein patch 的跨框架回归 | 仓库没有这些数组/TFRecord；公开存档主要为大体积 PDB/PLY，没有本次可直接使用的旧 NN fixture，未执行 |
| 全部原主预训练模型都可用 | 7 组原权重 NaN，明确拒绝；备份/历史模型不等价替代 |
| RANSAC、PD-L1 完整 benchmark、PyMOL coloring | 依赖原几何/Open3D 环境，未迁移运行 |
| 纯 CPU wheel、macOS/Windows 复装 | 给出 CPU recipe，实际仅 Linux CUDA wheel 及其 CPU 后端验证 |

没有以未经比较的几何替代库产生新 surface 来伪造旧数据一致性；没有声称原 PDB 预处理、全量科学指标或全部原权重已经通过。初始计划步骤 05 的完整真实样本 gate 因此**未完成**。

## 后续验收需要的材料

使用项目已有的真实 precomputation/TFRecord 与有限的原训练 checkpoint，按 README 的 `--input` 导出同一数据参考后验证。几何全链路需要恢复原工具/容器并固定其版本，先比较网格、特征、patch，再将输出交给新 NN。当前实现和安装流程已支持这一回归入口。
