# 02 — 模型、损失与优化器迁移

## 目的与计划

保持原网络的数学定义、张量形状、参数布局和梯度，使固定权重/固定 packet 在 Torch 下具有相同科学含义。共享真实重复的高斯卷积，保留各任务独立的 `nn.Module`；先核对源码，再逐模型做 TF 对照。

## 实现与代码改动

- `torch/src/masif_torch/models.py`：`GaussianConv`、共享 `PatchEncoder`、`MaSIF_site`、`MaSIF_ppi_search`、`MaSIF_ligand`、`ScoreNN`；明确 rho Å、theta radians、输入 `[N,V,F]`、索引 int64、device/dtype。每个函数有 docstring，检查 NaN/Inf、空维度、掩码与索引范围。
- 高斯 grid radial-major；mask 后按顶点 sum 归一化，加 `1e-5`；先 rotation max，再 ReLU。用 `amax` 重现 TF 并列 max 的分配梯度。FC 保持 MatMul 后 bias-add 顺序。
- site 支持实际工作的 1–3 层，保留 l2/l3 的 `mu_theta`/`sigma_rho` 错位使用和 reshape/mean 顺序。第四层原实现存在 incompatible shape，明确拒绝。输出 logits `[N,2]`；loss 为两个输出的 sigmoid BCE 的 **sum**。
- search 输出未 L2 归一化 `[N,G]` 描述符；四块距离损失使用 mean + population variance，不把原变量名中的 std 解释为标准差。
- ligand 用 `HᵀH/N` 非中心化 Gram，row-major flatten，默认原训练 dropout=0，输出 `[1,1,C]`。
- `KerasBatchNorm1d` 保存 population moving variance；moving factor=.01、eps=.001，与 Torch 默认 BN running variance 不同。
- `torch/src/masif_torch/optim.py`：`TFAdam`，epsilon 位于未做 bias correction 的 sqrt(v) 后；core epsilon=1e-8，ScoreNN epsilon=1e-7。参数组的 beta powers、moment 更新和 checkpoint 恢复均显式处理。
- `torch/tests/test_numerics.py`：独立 TF graph/source oracle、BN train/eval、Adam 小梯度、非法输入与恢复测试。

## 数值测试

```bash
conda activate masif_torch
python torch/tools/run_validation.py --exports torch/artifacts/source_oracles \
  --output torch/artifacts/source_validated --report docs/torch_migration/numerical_source.json --cuda
python -m unittest discover -s torch/tests -v
```

参考来自原源码实际构图：仅用同样的 TF primitives 替代已删除 contrib 构图器。小模型配置 F=3、G=4、rotations=3，包含全零 mask、角度边界、padding 与邻居 gather。比较每层输出、logits/概率、loss 与**每个 trainable parameter 的梯度**。

| 配置 | CPU output max abs | GPU output max abs | CPU / GPU loss max abs |
|---|---:|---:|---:|
| site 1 层 | 4.77e-7 | 2.38e-7 | 9.54e-7 / 9.54e-7 |
| site 2 层 | 1.12e-8 | 2.98e-8 | 9.54e-7 / 9.54e-7 |
| site 3 层 | 2.98e-8 | 5.96e-8 | 0 / 0 |
| search | 4.77e-7 | 4.77e-7 | 7.63e-6 / 1.91e-5 |
| ligand | 1.19e-7 | 1.19e-7 | 0 / 0 |

10 个模型/设备组合全部按 `atol=1e-5, rtol=1e-4` 通过，包括所有梯度。门槛是绝对加相对误差，不要求所有 max abs 小于 atol。完整记录见 [numerical_source.json](numerical_source.json)。

TF Adam 三步 oracle（含 1e-9 小梯度）默认 epsilon=1e-8 的三步参数逐位一致；Score epsilon=1e-7 的独立 oracle 也通过。BN training 输出、输入梯度、moving mean/variance、随后 eval 均通过。11 项 unit tests 无 skip 通过。

## 剩余限制

随机初始化和跨框架随机采样轨迹不承诺一致。source oracle 不是实际蛋白质样本。大尺度历史 search 权重的梯度有单独的严格失败记录，见下一阶段，不能扩大此处小模型的通过范围。
