# 08 — IPA 网络与双格式权重

## 目的与计划

将两个上游共用的 AlignmentEvaluationNN 移入新 Torch 包，保持实际层序、BN、padding、特征列和采样流；转换 TF checkpoint 与真实 HDF5。对权重、算子、官方源码和训练分别建立证据。

## 实现与代码改动

- `models.py` 在已有 ScoreNN 构造中参数化卷积宽度，派生五层 IPA（16/32/64/128/256）；共享 dense/BN，不新增重复网络实体。
- `AlignmentEvaluationNN.prepare/eval_model` 保留真实点数、100/200 补零、持续 RandomState(42)、signed point importance 和退化 NaN mask。
- `tools/legacy_weights.py` 按真实格式读取 TF 或 HDF5，canonical mapping 与转置逐元素逆向核验；`checkpoints.py` 计算正确的层 offset。
- `training.py` / `data.py` / `cli.py` / `validation.py` 加 alignment task。默认 lr=1e-4、epsilon=1e-7、class weights=(.01,1)。原 ScoreNN 默认仍为 class weights=(.005,1)。
- `export_design_weights.py` 批量处理两个固定上游的五组权重，9 Å 指定 100 点，其余 200 点。
- `export_ipa_source.py` 执行官方 Keras 原码，仅处理 API 名称兼容（session/seed、l2/lr、eager predict），不重写参考层。记录源和每个权重文件 SHA256。
- `validate_ipa_source.py` 比较连续采样、重要性、训练 loss/梯度/BN，追加真实 raw 特征、三步独立训练及相同梯度下的优化器更新。

## 数值结果

| 验收 | 实现结果 | 证据 |
|---|---|---|
| 两个仓库共十组权重的精确转换 | 全参数逆向 array_equal | 每个 `.conversion.json` |
| CPU/GPU 每层、输出、loss、全部梯度 | 20/20 默认 FP32 门槛通过 | [权重对照](numerical_design_weights.json) |
| 官方源码 padding/exact/连续 trim/importance、loss/grad/BN | CPU/GPU 默认门槛通过 | [源码对照](numerical_ipa_source.json) |
| 固定姿态真实 IPA raw features | CPU/GPU 默认门槛通过 | [真实与训练分组](numerical_ipa_source_real.json) |
| 同一组 Keras 梯度下 Adam 三步参数更新 | CPU/GPU 默认门槛通过，非逐位承诺 | 同上 `same_gradient_optimizer_updates` |
| 各框架独立梯度的三步参数更新 | **严格失败**，CPU 最大差 1.6096e-4，GPU 1.3390e-4 | 同上 `independent_training_parameter_updates` |
| 三步更新后的 eval 概率 | CPU/GPU 默认门槛通过 | 同上 `updated_eval_probabilities` |

默认门槛为 `atol=1e-5, rtol=1e-4`。最后报告整体 `passed:false` 并以退出码 1 返回，因为真实存在训练参数轨迹失败；没有将这些失败从总判定中删除。

## 原因与限制

训练 BN 减均值使前一层卷积 bias 的理论梯度为零。FP32 reduction 残差在 Torch/TF 中约 1e-6，可能符号相反；它们仍通过单步梯度门槛，但 Adam epsilon=1e-7 放大残差，导致 bias 更新约 1e-4 差异。subagent 独立复现并通过相同梯度的 Keras/TFAdam 实验定位。没有屏蔽 bias 梯度或增加 epsilon 改写上游数学。因此可以确认转换后推理精度，不可承诺独立训练轨迹完全一致。

benchmark TF 对象 checkpoint 带 Keras optimizer slots。转换只支持推理参数与 BN，清单显式记录忽略的槽；`--include-optimizer` 会明确拒绝。Torch 原生 checkpoint 支持 Trainer 状态恢复。旧 Keras optimizer 续训转换尚未实现。
