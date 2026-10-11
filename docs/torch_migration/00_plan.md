# MaSIF PyTorch 迁移计划

记录日期：2026-10-10。此文件是实施计划，尚未表示迁移完成。

## 目标与验收口径

让 MaSIF-site、MaSIF-ppi-search、MaSIF-ligand 及其实际调用的训练、推理、数据读取流程在新的 PyTorch 环境中运行；提供从现有 TensorFlow checkpoint 到 PyTorch `state_dict` 的可复现转换和安装流程。保留原始模型和权重作为数值参考，避免在迁移时同时更改网络数学定义。

原始网络以 `float32` 运行。不同框架、设备及矩阵乘法后端的浮点计算顺序可能不同，因此**不能承诺逐位相同**。验收分三层：

1. 权重：逐项核对变量名、形状、转置规则、元素数量；转换后逆向检查权重值，要求逐元素精确一致。
2. 算子与模型：在同一份固定输入上比较 TensorFlow 与 PyTorch 的每层中间张量、最终 logits、概率和描述符；记录最大绝对误差、最大相对误差、均方根误差、NaN/Inf、形状和 dtype。初始门槛为 CPU `float32` 的 `atol=1e-5, rtol=1e-4`，实际门槛须经基线测量确认并说明任何调整。
3. 业务输出：用真实预计算样本检查 site 的打分、ppi-search 的描述符与距离、ligand 的分类概率及评估指标；还要检查 CPU/GPU 差异。若没有真实样本，单元测试通过只能说明数学模块和转换器通过，不能宣布端到端数值一致。

数值测试默认关闭混合精度和 TF32，固定随机种子、输入顺序及 dropout 状态；训练过程比较前向值、loss 和梯度，不能要求不同框架的随机训练轨迹逐位重合。每一步形成对应 Markdown 日志，填写命令、输入 checkpoint/样本、版本、设备、数值结果和未通过项。

## 已检查的现状

- 核心模型位于 `source/masif_modules/MaSIF_site.py`、`MaSIF_ppi_search.py`、`MaSIF_ligand.py`。三者共享旋转高斯 patch 卷积，但后续层、损失及输出不同，适合抽取一个确有重复的卷积模块，再以三个 `torch.nn.Module` 保持任务接口。
- `MaSIF_site` 支持多层卷积，默认三层；ppi-search 输出描述符；ligand 对 patch 描述符构造 Gram 矩阵后分类。TensorFlow `tf.contrib.layers.fully_connected` 的参数布局与 `torch.nn.Linear` 不同，转换时必须转置 kernel。
- TensorFlow 还出现在三个训练/评估入口、ligand TFRecord 生成与读取、第二阶段评分网络中。第二阶段评分网络和所有现有模型变体必须列入覆盖清单。
- 仓库有 site、ppi-search、ligand 以及肽、PD-L1 等多个预训练 `.index/.data/.meta` checkpoint；没有检出的 TFRecord，只有少量 `.npy` 列表。真实预计算样本须另外定位或制作可审计的小型夹具。
- README 指向 Python 3.6 / TensorFlow 1.9，`requirements.txt` 未固定版本且列有系统工具。当前主环境为 Python 3.12.2，有 PyTorch、无 TensorFlow/PyMesh/BioPython/open3d；GPU 为 NVIDIA L40S，驱动 560.35.03。旧式 API 和外部几何工具是独立安装风险。

## 分步实施

### 01. 锁定基线与环境

**目的**：形成迁移前可重复运行的参照。**计划**：枚举全部 checkpoint 的变量和形状，核对各配置的输入维度；建立 `masif_nn`（仅旧图执行和 checkpoint 读取）与 `masif_torch`（目标运行时）两个隔离环境。优先用当前 TensorFlow 的 `compat.v1` 导入已有 `.meta` 图来执行推理；若旧图有不兼容算子，再使用独立的旧版容器/环境，不把 TF1.9 装进目标环境。**预期改动**：环境文件、环境自检脚本、基线采集工具、`01_baseline.md`。**数值测试**：同一个 checkpoint 和固定输入重复运行两次；保存参考中间层和输出，记录其稳定性及运行环境。

### 02. 迁移共享算子与三个模型

**目的**：忠实重现旧模型的计算顺序、维度和参数。**计划**：按原实现的高斯激活、mask 后归一化、角度取模、旋转最大池化及线性层顺序实现共享 `nn.Module`；site 的层数、ppi-search 的描述符和距离损失、ligand 的 Gram 矩阵与 dropout 分别实现。保持 `float32`，显式检查输入形状、device、dtype 和退化 mask；所有新增函数写文档字符串，对非显然的坐标/形状操作加注释。**预期改动**：PyTorch 模型模块、模型级测试、`02_models.md`。**数值测试**：每完成一个算子或任务，立即对照 TensorFlow 中间张量、输出、loss 和梯度，记录误差；不能仅用 PyTorch 自我对照。

### 03. 转换 checkpoint

**目的**：让仓库已有权重可直接供 PyTorch 推理使用。**计划**：用 TensorFlow checkpoint reader 读取旧变量，提供按任务与层数明确映射的转换命令；对 dense kernel 作转置，对高斯参数和偏置保持原顺序；排除 Adam 槽变量；拒绝缺失、重复、形状不符和未识别的模型变量。输出 `.pt` 文件、源 checkpoint 摘要、模型配置、转换清单及校验结果。**预期改动**：转换工具、映射测试、`03_checkpoints.md`。**数值测试**：所有仓库预训练 checkpoint 逐项校验，装载后对固定输入再次比较完整输出；抽样做逆向权重核对。

### 04. 迁移数据读取、训练与推理入口

**目的**：用户无需 TensorFlow 就能在目标环境运行三项任务。**计划**：保留现有 `.npy` 数据含义，给 ligand 的 TFRecord 增加目标环境可读的读取或一次性转换流程；将 site、ppi-search、ligand 的训练、预测、评估脚本接到新模型上，并保持原输出文件的形状和语义。第二阶段评分网络单独核对其 Keras 权重和输出后迁移。避免一次性重写几何预处理。**预期改动**：数据适配、入口脚本、`04_workflows.md`。**数值测试**：固定 batch 的样本顺序、标签、特征、loss、梯度以及保存的 `.npy` 输出与旧流程比较；训练做最小一步更新的有限值和形状检查。

### 05. 安装与端到端验证

**目的**：得到可重建、易安装的环境和清楚的边界。**计划**：提供 CPU 与本机可用 GPU 的固定依赖安装说明和一键自检，分别说明 MSMS/APBS/PyMesh 等外部几何工具的获取与检查；仅在验证替代库结果后才替换几何依赖。用公开或仓库内真实样本执行“预处理 → 加载转换权重 → 推理 → 输出”与最小训练烟雾测试。**预期改动**：环境锁定文件、`INSTALL_TORCH.md`、示例命令、`05_end_to_end.md`。**数值测试**：三任务的真实样本回归结果、CPU/GPU 差异和复装结果均留档。

### 06. 独立代码评审与总报告

**目的**：让实现和数值证据经第二视角检查。**计划**：请 subagent 对每一阶段的代码做独立 review，特别检查维度、参数映射、mask、角度周期、dropout、标签、设备及异常路径；修复后重跑受影响的最小数值测试。**预期改动**：review 记录、修复日志、`06_review.md` 与 `REPORT.md`。**数值测试**：最终汇总全部 checkpoint、测试样本、误差与未通过项；只有所有验收项通过才写“完成”。

## 顺序与完成条件

按 01 → 02 → 03 → 04 → 05 → 06 执行。每一步先记录目的和计划，再记录实际代码改动及数值测试；测试失败时先定位差异，不扩大下一步范围。完成条件是：两个环境可重建，全部目标模型和权重可用，三项任务的训练与推理可运行，旧输出与新输出的误差满足已经记录的门槛，独立 review 的高优先级问题已解决，最终报告给出剩余限制。

## 版本选择的依据

实施时核对实际发布与 wheel：最终安装 Python 3.12.15、PyTorch 2.14.1 的 CUDA 12.6 wheel，当前驱动无需升级；TensorFlow 参考环境采用 tensorflow-cpu 2.21.0，亦使用 Python 3.12。初始计划的 Torch 2.13 / Python 3.11 选择已由实际安装证据更新。依赖锁定与具体安装命令见 `torch/environments/` 和 `torch/README.md`。

参考：[PyTorch 版本安装命令](https://pytorch.org/get-started/previous-versions/)、[PyTorch 2.14 发布说明](https://pytorch.org/blog/pytorch-2-14-release-blog/)、[TensorFlow 安装版本](https://www.tensorflow.org/install/pip)、[PyTorch 数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。
