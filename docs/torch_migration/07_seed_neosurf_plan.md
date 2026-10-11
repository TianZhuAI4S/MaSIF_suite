# 07 — seed / neosurf 集成：基线与设计

## 目的与计划

在现有独立 `torch/` 包内集成 seed 和 neosurf：共用 site/search NN，新增 IPA 对齐评分网络与数据/检索/注册/评分流程；保留科学数组含义。参考相邻 `X/skill/references/architecture.md` 的共享实现、任务归属、单向依赖、最小接口和无空包原则，采用 OOP 组合，不复制两个相同任务流程，不引入不需要的 Hydra 依赖。

## 已检查的基线

- seed 官方提交 `77153911d8252d02193a6d1d6d0539bf02c5faba`；neosurf 官方提交 `b522fdc537e0724e7c574753f9068f88e687c736`。
- 两者 site/search 模型源码与现有原 TF 模型一致；两者 IPA `AlignmentEvaluationNN` 源码逐字相同。
- IPA 使用 5 个 Conv1D/BN/ReLU（16→32→64→128→256），原 ScoreNN 为 6 层（8 起）；两者 dense 和 BN 数学可共享，但不能用原 ScoreNN 权重布局代替 IPA。
- IPA 文件既有 TF 对象 checkpoint，也有真正 HDF5；benchmark 对象 checkpoint 包含 optimizer slots，必须显式记录，不能误称全部没有。
- 两个上游 site/search 主 checkpoint 仍含 NaN；IPA 所有检查权重有限。
- neosurf 特定几何来自 protein+ligand 预处理；没有独立新 NN，不能把 seed 参数换名字当成完整小分子几何支持。

## 数值验收计划

1. 所有 IPA 变体权重精确逆向，CPU/GPU 每层/输出/loss/全部梯度对 TF 对照；真实 HDF5 也同样测试。
2. 运行原 IPA 源码构建的 Keras 层作第二参考，覆盖 padding、超长 RandomState 采样、point importance。
3. 真实公开预处理 fixture 比较 target site selection、strict descriptor cutoffs、target normal flip/shift、距离特征、IPA。
4. 同一组 legacy transforms 下做确定性几何/评分回归。现代 Open3D RANSAC 改了停止条件（原 max_validation vs 新 confidence），新增可用后端须单列功能验证，不能声称其随机姿态逐位相同。
5. 保留历史全部 11 unit tests 与模型 gates；继续 subagent 只读 review、修复后重跑相关测试。

## 实施中的代码

`models.py` 参数化已有 ScoreNN 的卷积构造，派生 IPA；`legacy_weights.py` 支持 HDF5/TF 对象权重；checkpoint mapping 根据实际 conv 数量计算 dense offset；现有 trainer/validator/CLI 兼容 alignment task。接下来的 workflow code 将放在任务所属 `design/`，CLI 延迟导入，已有公共模型不导入任务代码。

本文件是实施记录，尚不声明所有阶段已完成。最终误差、运行范围与限制将记录在后续阶段及整体报告。
