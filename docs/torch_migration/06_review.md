# 06 — 独立代码评审与修复

## 目的与计划

按用户要求使用 subagent review，从原源码数学语义与新实现两方面检查：参数映射、shape、mask、device/dtype、BN、优化器、训练恢复、输入异常、安装和 README。每次修复重跑受影响的数值检查。

## 实现与评审方式

subagent `legacy_review` 独立只读原实现和新目录，没有修改代码；主 agent 实施修复。评审分原 TF 行为分析、Torch 实现评审、再次实测、README 命令核对。所有实现位于新目录，原 TensorFlow 文件未改，未提交 Git commit。

## 发现与实际代码修复

| 问题 | 修复位置与验证 |
|---|---|
| site l2/l3 名称交换、max tie 梯度、search variance | models/checkpoints；源 TF 各层/loss/全部梯度对照 |
| TF Adam epsilon 与 Torch Adam 语义不同 | optim.py；独立 TF 三步小梯度 oracle |
| 转换 optimizer 默认 lr 不一定等于旧图值 | export_tf/checkpoints；读取原图超参数，不用猜测值 |
| BN running variance 默认 Torch 使用 unbiased | KerasBatchNorm1d；TF train/eval/statistics/gradient oracle |
| negative neighbor/loss index 被 Torch 当作合法负索引 | models.py 显式范围检查；异常测试 |
| 参数组 beta powers 未按各组 betas 初始化 | optim.py；custom group regression |
| ScoreNPZ 不分 batch 忽略 batch-size | data.py/cli.py；alignment 全覆盖和 bounded batch 测试 |
| 不可训练 pocket/site 终止整次训练 | Dataset 初始化过滤、CLI excluded.json；filter test |
| resume 缺 seed、class weights、Torch/CUDA RNG | checkpoints/Trainer/CLI；带 dropout 下一步和实际 CLI 两轮逐位恢复 |
| resume 丢 history/覆盖旧 best | checkpoint 保存/恢复 best_metric/history；resume history test |
| Score Adam epsilon 应为 1e-7 | Trainer 分任务选择；第二个 TF Adam oracle |
| Score validation 不该应用训练 class weights | compute/evaluate；train weighted/eval unweighted sanity test |
| 最后 Score batch 日志需按 sample 数平均 | fit/evaluate 都按样本数加权；完整 CLI smoke |
| 真实 NPZ 的 NV1 geometry 被参考导出重复扩维 | export_tf 的 geometry_array；TF 两种 rank 全输出/梯度精确一致及自定义导出回归 |

评审还要求文档明确 ligand mask 错配、eligible sampling、site idx 命名修复、partial feature reflection 的变化；详见步骤 04，未把固定输入数学相同表述为完整训练采样轨迹相同。

## 独立实际测试

reviewer 运行完整 11 项 unit tests，全部通过。额外 CLI：ligand dropout=.25、seed31，连续两轮与一轮→resume→一轮，state_dict 全部逐位一致，seed 自动恢复、history 两行；predict 成功且概率 shape `(1,1,3)`。修复后未发现阻断运行的代码问题。

README 命令与实际 CLI 一致，最终评审未发现科学范围或数值精度 overclaim；阶段链接均已补齐。数值报告分别保留严格结果、历史 search 的特殊梯度结果、拒绝权重；不把失败或未运行测试改成通过。

reviewer 最终在临时目录独立复现 geometry rank 修复：site3/search/ligand 原 TF 所有输出/梯度 array_equal；自定义输入导出转换后分别 46/26/30 个张量通过默认门槛，含字符串 metadata 和缺省 unused indices 正常。最终只读评审无剩余 actionable 问题。
