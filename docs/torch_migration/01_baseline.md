# 01 — TensorFlow 基线与隔离环境

## 目的与计划

使用仓库原始 `.meta` 图及 checkpoint 产生独立参考，避免把重新写出的公式作为唯一判定依据。新建 `masif_nn` 与 `masif_torch`，以固定输入采集前向、梯度和权重。

## 实现与代码改动

- 新代码统一位于 `torch/`，采用 `src/masif_torch` 包结构，提供独立安装入口。
- 已创建 `/home/tian/miniconda3/envs/masif_nn` 和 `/home/tian/miniconda3/envs/masif_torch`，两个环境均使用 Python 3.12.15。
- 实时核对 PyPI 与官方 wheel 索引发现最新可用版本为 PyTorch 2.14.1，且有 Python 3.12/CUDA 12.6 wheel，故采用此版本。初始计划中的 2.13 选择由实际发布信息更新。
- 原始代码评审确认 site 参数错位、variance 损失、max ties 梯度和 TF Adam 的 epsilon 放置，均纳入数值验收。
- 环境定义及安装快照：`torch/environments/masif_nn.yml`、`masif_torch.yml`、`runtime-requirements.txt`、两个 `.lock.txt`。
- `torch/tools/export_tf.py`：枚举旧变量、检测非有限值、从原 `.meta` 图采集层输出/loss/梯度、记录源文件 SHA256、实际 optimizer 超参数和版本；ScoreNN 无 `.meta`，按旧架构构建 TF primitive ops 参考。
- `torch/tools/create_source_oracles.py`：实际执行原三模型源码，仅替换移除的 contrib 构图器和过时的参数数量打印，不以 Torch 公式代替旧图。
- `torch/src/masif_torch/validation.py`：固定种子、线程、TF32、确定性策略，输出形状/finite/绝对相对误差/RMSE/relative L2。

## 数值测试

```bash
conda activate masif_nn
python torch/tools/export_tf.py --repo . --output torch/artifacts/tf_exports
python torch/tools/create_source_oracles.py --source source --output torch/artifacts/source_oracles
conda activate masif_torch
python -m masif_torch doctor
```

原 TF 图对同一输入运行两遍，所有导出的有效参考逐位一致（manifest `repeat_exact=true`）。枚举 11 组仓库权重，4 组有效，7 组各有 18 个非有限参数/Adam 槽张量；拒绝清单保存在 [checkpoint 数值报告](numerical_checkpoints.json)。不是转换引入 NaN。

目标自检：Torch `2.14.1+cu126`，CUDA 12.6，可用 L40S，`tensorflow_installed=false`；`pip check` 无依赖冲突。参考 TF `2.21.0`，CPU 单线程，禁用 oneDNN，NumPy `2.5.3`。

## 边界

`masif_nn` 用 compat.v1 执行保存图/原源码构图工具，不是全部原 TF1/PyMesh 程序直接兼容的环境。输入目前都是可审计的合成夹具，没有真实蛋白质 patch 数据。后续阶段不能把这些参考宣称为论文复现。
