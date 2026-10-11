# 10 — 集成评审与收尾

## 目的与计划

持续由 subagent legacy_review 独立检查实际上游数学、权重格式、真实数据读取、训练与数值声明；修复后运行最小相关 gate，保留失败结果。

## 审查发现与改动

| 发现 | 修复/实现 | 验证 |
|---|---|---|
| 每次重新 seed 会重复抽取超长 patch | model/scorer 使用持续 RandomState(42) | 连续 trim 的官方源码 oracle |
| 目标数据只有 flipped | descriptors 可为 None，NPZ 保留缺失语义 | 真实 A/B adapter 与 roundtrip |
| source iface 应读 NPY，target 读 PLY | 明确 source/target role | 真实 adapter exact oracle |
| Open3D Feature 将 descriptor 转为 double | 匹配保留原 dtype，patch 特征保持 double | 真 Open3D Feature 实测/原函数回归 |
| 浮点 neighbor index 会被截断 | 先检查整数 dtype/offset | malformed packet unit test |
| replay 恰 1 Å 应不计 fitness | 改为 strict <1，与 Open3D correspondence 一致 | exact-radius regression |
| 上游名称作为 task 会 Unknown | 增加 masif_seed、masif-neosurf 等 aliases | 注册接口检查 |
| seed 输出氢原子集合与原实现不同 | search 按原 atom-name 规则移H；score-only原PDB逐字节复制 | 含H的合成PDB测试与原码审查 |
| Keras 独立训练 bias 更新不一致 | 定位 BN 消减+Adam 放大；追加相同梯度与更新后预测 gate | 失败原样保留，源码 audit exit 1 |

## 独立证据

reviewer 直接调用官方 model.load_weights 加载真正 HDF5，未复用转换 reader，前向、padding、trim 输出 max_abs=0；重要性退化 NaN mask 相同。随后独立生成 AST 原函数 geometry oracle，83 项 exact；审查 workflow 的 cutoff/order/frame/PDB 变换、nearest-center 和 descriptor diagnostics。18 项 unit 已独立重跑通过。

## 当前结果与限制

所有成功转换权重精确；已审查设计入口可运行。IPA 每层/输出/loss/梯度/BN 和真实蛋白对照满足默认门槛。三步独立 Keras/Torch 参数轨迹严格失败已公开，不能当作全部数值验收完成；随机 docking 和原始表面/Rosetta 全流程也不在已验证范围。代码、README、安装流程、逐步记录与整体报告同步这些边界。

## 最终检查

主 agent 与 reviewer 独立重跑完整18项，无skip；修复后的CPU/GPU设计输出重跑通过。pip check无损坏依赖，doctor确认Torch环境无TensorFlow且GPU可用。公共task导入不载入TensorFlow/Open3D；AST扫描所有新增函数都有docstring，本地Markdown链接通过。[最终检查摘要](numerical_design_tests.json)
