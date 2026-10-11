# 04 — 数据、训练、恢复与推理入口

## 目的与计划

在无 TF 的目标环境直接使用原 numeric 预计算数组；旧 TFRecord 做一次性精确转换。保持任务输出 shape/含义，检查样本、索引、标签、优化器恢复和保存文件，并记录必要的数据错误修复。

## 实现与代码改动

- `torch/src/masif_torch/data.py`：原 p1/p2 `.npy` 读取、feature mask、ragged neighbor 自索引 padding、binder reflection、safe NPZ、deterministic PacketDataset 与原 search cache memory-map。
- `torch/tools/convert_records.py`：原 ligand TF Example 的数组/shape/PDB 解码，按结构保存 NPZ，每个数组逐元素精确回读检查；TF 仅在一次性转换环境使用。
- `torch/tools/pack_ligand.py`：从预计算数组与 ligand 原子坐标按原 3 Å KDTree cutoff 标注七类 pocket，保留显式 split；其真实分子输入尚未回归。
- `torch/src/masif_torch/training.py`：训练/验证、TFAdam、finite 梯度检查、最后/最佳 loss checkpoint、恢复 optimizer/epoch/step/seed/class weights/RNG/history/best。
- `torch/src/masif_torch/cli.py`：doctor、convert、validate、predict、surface、pack-surface、evaluate-ligand、train；原 TF 入口不改，用户显式选择 Torch CLI。
- `torch/tools/smoke_workflows.py`：调用真正安装的 CLI，CPU/GPU 训练→恢复→预测、原 surface 打包、cache 训练、ligand 评估保存。

## 数据处理的明确变化

1. ligand 原训练使用 `mask[pocket_points[:32]]` 配合随机选中的 features/rho/theta，导致错配；现在四个数组均用同一个 selected。此为修复，不是逐位重现旧数据 bug。
2. ligand 新流程均匀选择 eligible pockets（至少 32 个点），旧流程可能随机选到小 pocket 后跳过，更新频率不同。无可用 pocket 的结构在 Dataset 初始化时排除，并记录原因。
3. site 单层 subset 原排列为 negative/positive，但原 pos_idx/neg_idx 名称相反；新 idx 指向正确类别。被选中集合和 sum BCE 不变，但逐样本报告顺序可以改变。
4. partial feature mask 的 binder reflection 按原列身份保留疏水性，不按原脚本的 feature 个数 hack 判断。默认五特征行为相同。
5. ScoreNPZ 按 batch size 分批、每轮每 alignment 使用一次，默认 32；class weights 仅训练应用，按样本数除，验证不加权。训练/验证日志末 batch 均按 sample 数加权。
6. 数据分割由输入目录显式提供；没有复制原 site/search 的 AUC selection/early stopping。best 按验证 loss；这改变训练策略，不能宣称历史实验训练轨迹一致。

## 数值测试

```bash
conda activate masif_nn
python torch/tools/convert_records.py --input torch/artifacts/source_oracles/synthetic.tfrecord \
  --output torch/artifacts/records
conda activate masif_torch
python -m unittest discover -s torch/tests -v
python torch/tools/smoke_workflows.py --oracles torch/artifacts/source_oracles \
  --score-export torch/artifacts/tf_exports/data/masif_pdl1_benchmark/models/nn_score/trained_model.hdf5 \
  --output torch/artifacts/workflows --report docs/torch_migration/numerical_workflows.json --cuda
```

TFRecord 的 features/rho/theta/mask/pocket_labels/PDB 全部 array_equal；unit tests 检查过滤、Score 分批覆盖、反射、padding、weighted reduction、非法输入。固定 packet loss/gradient 与 TF 的对照见步骤 02/03。

最终真实输入导出 regression（仍为合成数值）：site3/search/ligand 的二维/三维几何全部原 TF 输出和梯度逐位相同；去掉 unused indices、自动生成 site loss indices、含 PDB 字符串的 custom export→convert→validate，6 个 CPU/GPU 组合默认门槛全部通过。证据：[TF rank 报告](numerical_input_shapes.json)、[Torch 回归](numerical_input_shapes_torch.json)。

[numerical_workflows.json](numerical_workflows.json)：11 条 workflow 检查全部通过，包含四 task × CPU/GPU 的真实 CLI 调用、原 surface NPY 与 pack-surface、ligand 两次采样评估（shape `[1,2,1,1,3]`）、search cache 两步训练。恢复 epoch=1、history 两行，所有 checkpoint 参数 finite。

保存 NPY 对独立 TF 参考的 max abs：site 5.96e-8、search 4.77e-7、ligand 2.98e-8、Score 5.96e-8；CPU/GPU 均通过默认门槛。

subagent 额外实际 CLI 检查：ligand dropout=.25、seed31，连续两轮 vs 一轮后恢复再一轮，全部 state_dict 逐位相同，seed/history 正确恢复。此为同一框架 native resume 测试，不是跨框架随机训练复现。

## 边界

烟测输入为明确的 synthetic packet。尚未执行真实 split 的全量训练/科学评估，没有计算论文 ROC AUC 或 pocket accuracy；RANSAC 与原 PDB 几何脚本未接入新环境。对于现成预计算数据，新的 NN 入口已能运行。
