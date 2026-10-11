# 03 — TensorFlow 权重与优化器转换

## 目的与计划

显式映射旧变量，保持其数值，输出目标环境可读取的 checkpoint；枚举仓库所有预训练权重，拒绝坏模型并记录原因。对有效权重做完整前向/loss/gradient 对照，避免只检查能否载入。

## 实现与代码改动

- `torch/tools/export_tf.py`：TF checkpoint reader、原保存图参考、配置推断/显式覆盖、模型/Adam finite 检查和源文件 hashes。
- `torch/src/masif_torch/checkpoints.py`：逐项映射和 shape/dtype 检查，FC kernel 转置；高斯卷积 kernel 保持原方向；site 后两层按旧图实际语义交换名字；Score Conv1D kernel `[K,in,out] → [out,in,K]`，BN buffers 映射完整。
- 每个映射参数反向恢复为 TF layout，`array_equal` 验证 **bitwise reversible**；拒绝缺参数、未知模型变量和非有限数值。
- `--include-optimizer` 保存 m/v、beta powers 和原图实际 learning rate/betas/epsilon；不擅自用默认学习率替换旧图配置。Score 对象 checkpoint 没有 optimizer，显式拒绝导入不存在的槽。
- `torch/tools/fetch_reference_weights.py`：固定官方历史 search revision 与三文件 SHA256，仅写入明确指定的新目录。
- `torch/tools/run_validation.py`：按源数据文件 SHA256 识别历史模型，严格报告和放宽梯度报告分别保留。

## 数值测试与发现

```bash
conda activate masif_torch
python torch/tools/run_validation.py --exports torch/artifacts/tf_exports \
  --output torch/artifacts/validated_checkpoints \
  --report docs/torch_migration/numerical_checkpoints.json --cuda --historical-gradient-roundoff
```

仓库 11 组 checkpoint 中 4 组有限且成功转换，所有权重逆向检查精确一致；另加入一组官方历史 search 权重。各有效模型 CPU/GPU 前向均通过原门槛 `1e-5 + 1e-4*abs(ref)`。

| 有限权重 | CPU logits/descriptor max abs | GPU max abs | 梯度 |
|---|---:|---:|---|
| 原 ligand all_feat | 1.64e-4 | 9.16e-5 | 默认门槛通过 |
| peptides site model_data.bak | 1.91e-6 | 1.91e-6 | 默认门槛通过 |
| 两组 ScoreNN（相同参数） | 7.63e-6 | 1.91e-6 | 默认门槛通过 |
| 官方历史 search `51e5422…` | 2.86e-6 | 4.77e-6 | 默认严格门槛失败，单独梯度策略通过 |

ligand logits 值较大，relative L2 约 4.94e-7/2.67e-7，因此绝对差 1.64e-4 仍满足原相对门槛；不能把 atol 单独当作判据。Score 每组包含 47 个参考张量（层输出、logits、概率、未加权 eval loss、全部参数梯度），梯度 max abs CPU 5.72e-6 / GPU 1.91e-6。

### 历史 search 梯度

严格结果完整保存在每个历史模型记录的 `strict_result`。其 synthetic loss 约 3.39e4，loss max abs .0078125，relative error 2.31e-7；大尺度参数梯度 max abs CPU .0390625 / GPU .01953125。多数非零梯度 relative L2 约 1e-6；最终 bias 在理论抵消的位置有约 2.44e-4 残差，近零梯度的 relative L2 可大于 1。

因此，仅此**已固定 SHA 的历史模型**采用 `grad_atol=.01, grad_rtol=.001`，前向/概率/loss 门槛保持原值。报告不隐藏严格失败，也不将其称为严格训练精度全部相同。小源码 search oracle 的所有梯度仍按默认严格门槛通过。

### 拒绝清单

以下每组均有 18 个非有限模型参数/Adam 槽张量；坏值位于第一 feature 的高斯参数、卷积 kernel/bias 及其槽，不是目标代码生成：

```text
data/masif_site/nn_models/all_feat_3l/model_data/model
data/masif_ppi_search/nn_models/sc05/all_feat/model_data/model
data/masif_pdl1_benchmark/nn_models/sc05/all_feat/model_data/model
data/masif_pdl1_benchmark/nn_models/all_feat_3l/model_data/model
data/masif_peptides/nn_models/sc05/all_feat/model_data/model
data/masif_peptides/nn_models/all_feat_3l/model_data/model
data/masif_ppi_search_ub/nn_models/sc05/all_feat/model_data/model
```

本地读取与当前官方上游对应文件对照，坏模型内容同源；不能通过重新下载相同字节修复。没有置零、忽略这些模型参数或覆盖原文件。逐变量 nonfinite counts 见 [numerical_checkpoints.json](numerical_checkpoints.json)。

## 使用与边界

新 `.pt` 文件包含 config、state_dict、provenance 和可选 optimizer；用 `torch.load(..., weights_only=True)` 读取。当前生成文件在 ignored 的 `torch/artifacts/validated_checkpoints/`，可由 README 命令重建。

备份 site 与历史 search 是独立、显式选择的模型，不等价于坏掉的主模型。历史 search 位于旧预处理时期，未验证其匹配新几何处理或原论文指标。旧 TF checkpoint 不保存 sampler/RNG/epoch 日志，因此只能恢复已存在的优化器状态，不能复原全部原训练状态。
