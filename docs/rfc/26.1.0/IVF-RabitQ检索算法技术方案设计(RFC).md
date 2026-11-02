# IVF-RabitQ检索算法技术方案设计(RFC)

**状态 (Status):** Draft
**作者 (Authors):** @xiangjie10
**创建日期 (Created):** 2026-05-06
**更新日期 (Updated):** 2026-10-31
**相关 Issue/PR:** [#33](https://gitcode.com/Ascend/IndexSDK/issues/33)

---

# 1. 概述

## 1.1 简介

本提案旨在为 IndexSDK 新增 IVF-RabitQ 检索算法支持。IVF-RabitQ (Inverted File System with Random Binary Quantization) 是一种基于倒排索引和随机二进制量化的大规模向量检索算法。RaBitQ 是一种高维向量量化算法，通过结合倒排文件系统（IVF）和随机正交矩阵量化技术（RaBitQ），在保证检索精度的同时，实现高性能的近似最近邻搜索。该算法适合大规模向量库的检索场景。

### 核心特性

- **随机正交矩阵**：通过随机正交变换提高量化精度
- **两级检索**：L1阶段筛选聚类，L2阶段在选定聚类中计算距离
- **查找表加速**：使用LUT（Look-Up Table）加速距离计算
- **Refine机制**：可选的精排机制，提高召回率

## 1.2 动机

### 背景

随着向量检索场景的库规模增长，传统的精确检索方法在时延和内存占用上面临巨大挑战。IVF-RabitQ 算法通过倒排索引和快速量化技术，能够在一定程度保证召回率的前提下，大幅降低计算复杂度和内存占用。

### 痛点

- 大规模向量库检索时延高，无法满足实时业务需求
- 内存占用大，成本高昂

### 价值

- 提供高性能的近似检索能力
- 降低内存占用，减少硬件成本

## 1.3 目标

### 目标

- 支持 `IVF-RabitQ` 算法
- 支持完整的生命周期：训练、索引构建、入库、检索
- 精度与 CPU 基线对齐
- 保持与现有 Index 接口的兼容性，不破坏现有功能

### 平台支持

- **Atlas A2推理系列产品/Atlas A3推理系列产品平台**：支持
- **Ascend 950PR系列产品平台**：支持

### 功能范围

- **支持的功能**：
  - 随机正交矩阵变换（可配置是否使用）
  - 两级检索（L1筛选聚类 + L2距离计算）
  - LUT查找表加速
  - Refine精排机制（可选）
  - copyTo/copyFrom（CPU与NPU数据互拷）
  - 删除和更新操作

- **距离度量**：
  - L2距离

# 2. 用例分析

## 2.1 功能需求

### 核心功能

- **训练**: 支持聚类中心训练
- **索引构建**: 支持倒排索引和快速量化码本构建
- **入库**: 支持向量添加到索引
- **检索**: 支持基于倒排索引的近似检索

### 参数规格

- **nlist**: 倒排列表数量，支持 1024、2048、4096、8192
- **dim**: 向量维度，128
- **batchSize**: 批量大小，支持 1、2、4、8、16、32、64
- **topk**: 返回数量，≤320 (召回 300)
- **nprobe**: 32， 64
- **base**: 库规模，1000w

### 功能要求

- 支持完整的训练、构建、入库、检索流程
- 支持 L2 距离度量
- 支持 batch 模式下的并发查询
- 返回结果保证排序正确性

## 2.2 性能需求

### 精度验收标准

与 CPU 结果相比，TOPK300 一致性不低于 95%

### 性能验收标准

在 A3 单卡场景下:

- `nlist=1024`， `dim=128`
- `batchSize=1/2/4/8/16/32/64`
- `topk=300`
- `nprobe <= nlist` (召回 32/64)
- `base=1000w`
- 目标时延: `1.2ms`、`1.8ms`

### 精度测试约束

1. 对比时需要将聚类步骤放在 CPU 上计算，与 Faiss 代码和参数配置保持一致
2. 需要将随机旋转矩阵配置为单位矩阵，避免随机性引起精度误差

## 2.3 DFX 要求

### 可靠性

- 算法计算结果正确性保证
- 异常输入能够检查并输出异常

### 可测试性

- 提供单元测试用例
- 提供性能基准测试用例
- 提供精度对比测试用例

### 兼容性

- 与现有 Index 接口保持兼容
- 不影响现有其他算法的功能

# 3. 方案设计

## 3.1 总体方案

### 设计思路

IVF-RabitQ 算法结合了倒排文件系统（IVF）和随机正交矩阵量化技术（RabitQ），主要包含以下核心步骤：

1. **训练阶段**：
   - 生成随机正交矩阵（可选）
   - 执行 K-Means 聚类，生成聚类中心
   - 计算聚类中心的LUT查找表

2. **索引构建阶段**：
   - 对聚类中心应用随机正交变换
   - 计算聚类中心的L2范数
   - 构建倒排索引结构

3. **入库阶段**：
   - 对向量应用随机正交变换
   - 计算向量的L2范数
   - 进行RabitQ量化编码
   - 计算预计算常数（L1、L2）
   - 将编码后的向量添加到对应倒排列表

4. **检索阶段**：
   - **L1阶段**：查询向量旋转 → 计算查询向量LUT → 与所有质心距离计算 → 选择最近的nprobe个聚类
   - **L2阶段**：在选定的聚类中 → 查表计算距离 → TopK选择
   - **Refine阶段**（可选）：对TopK结果进行精排

### 技术架构

```text
用户应用层
    ↓
AscendIndexIVFRaBitQ API
    ↓
AscendIndexIVFRaBitQImpl
    ↓
├── 训练模块
│   ├── 随机正交矩阵生成
│   ├── K-Means 聚类
│   └── 质心LUT计算
├── 入库模块
│   ├── 向量旋转
│   ├── L2范数计算
│   ├── RabitQ编码
│   └── 预计算常数
├── 检索模块
│   ├── L1阶段（聚类筛选）
│   ├── L2阶段（距离计算）
│   └── Refine精排（可选）
└── 辅助功能
    ├── copyTo/copyFrom
    ├── 删除/更新
    └── 多设备管理
    ↓
ACL Runtime
    ↓
NPU (Atlas A2推理系列产品/Atlas A3推理系列产品/Ascend 950PR系列产品)
```

### 核心流程

#### 训练阶段

1. **随机正交矩阵生成**：
   - 使用Givens旋转生成随机正交矩阵
   - 矩阵维度：dim × dim
   - 可配置是否使用（useRandomOrthogonalMatrix）
   - 可配置随机种子（matrixSeed）

2. **K-Means聚类**：
   - 输入训练数据集
   - 执行K-Means聚类，生成nlist个聚类中心
   - 复用IVFFlat的一阶段检索能力加速聚类过程

3. **质心预处理**：
   - 对质心应用随机正交变换
   - 计算质心的L2范数
   - 计算质心的LUT查找表

#### 入库阶段

1. **向量预处理**：
   - 对向量应用随机正交变换
   - 计算向量的L2范数

2. **RabitQ编码**：
   - 根据 `quantBits` 将向量编码为 1/2/4-bit 紧凑表示
   - 编码长度：`dim * quantBits / 8` 字节
   - 2/4-bit 使用每 8 维一组、MSB 到 LSB 排列的 bit-plane 格式

3. **预计算常数**：
   - 计算L1常数：用于距离计算的偏移项
   - 计算L2常数：用于距离计算的缩放项

4. **倒排索引构建**：
   - 根据向量所属聚类，添加到对应倒排列表
   - 存储编码数据、索引、预计算常数

#### 检索阶段

##### L1阶段（聚类筛选）

1. **查询向量预处理**：
   - 对查询向量应用随机正交变换
   - 计算查询向量的L2范数
   - 计算查询向量的LUT查找表

2. **距离计算**：
   - 计算查询向量与所有质心的距离
   - 使用LUT加速计算

3. **TopNprobe选择**：
   - 选择距离最近的nprobe个聚类
   - 返回聚类索引和距离

##### L2阶段（距离计算）

1. **距离计算**：
   - 在选定的nprobe个聚类中
   - 使用LUT查表计算查询向量与库向量的距离
   - 公式：`distance = L2_norm + L1_offset + LUT_lookup`

2. **TopK选择**：
   - 对距离结果进行TopK排序
   - 返回TopK个最近邻的索引和距离

##### Refine阶段（可选）

1. **精排计算**：
   - 对TopK结果进行重排序
   - 使用原始向量计算精确距离
   - 提高召回率

## 3.2 技术选型

### 方案

基于现有 IVF 框架，新增 RabitQ 量化算子，实现两级检索机制

### 选择理由

1. **复用现有架构**：IVF框架已在IndexSDK中实现，复用可以降低开发风险
2. **随机正交矩阵优势**：通过随机正交变换，可以减少量化误差，提高召回率
3. **两级检索高效**：L1阶段快速筛选聚类，L2阶段精确计算，平衡性能和精度
4. **LUT加速**：使用查找表加速距离计算，大幅提升性能

### RabitQ 核心原理

#### 1. 随机正交矩阵

- **作用**：通过随机正交变换，使向量在各个维度上的分布更加均匀，减少量化误差
- **生成方法**：使用Givens旋转生成随机正交矩阵
- **配置参数**：
  - `useRandomOrthogonalMatrix`：是否使用随机正交矩阵
  - `matrixSeed`：随机种子，保证可复现性

#### 2. RabitQ量化编码

- **编码方式**：`quantBits=1` 保持原二值量化；`quantBits=2/4` 使用多 bit bit-plane 量化
- **编码长度**：`dim * quantBits / 8` 字节
- **兼容性**：`quantBits` 默认值为 1，保持原有调用和 1-bit 索引格式不变
- **编码过程**：
  1. 对向量应用随机正交变换
  2. 计算向量的L2范数
  3. 计算向量相对所属聚类中心的残差
  4. 对 1-bit 执行符号量化；对 2/4-bit 按 RMS 步长量化并输出 bit-plane

#### 3. LUT查找表

- **作用**：加速距离计算，避免重复计算
- **实现方式**：
  - 将向量按8位分段
  - 每个 bit-plane 使用 256 项 LUT
  - LUT 列数为 `256 * quantBits`
  - 检索时按位权累加各 bit-plane 的查表结果

#### 4. 预计算常数

- **L1常数**：距离计算的偏移项
- **L2常数**：距离计算的缩放项
- **作用**：进一步加速距离计算

#### 5. Refine机制

- **作用**：对TopK结果进行精排，提高召回率
- **实现方式**：使用原始向量计算精确距离
- **配置参数**：
  - `needRefine`：是否启用Refine
  - `refineAlpha`：精排倍数（TopK × refineAlpha）

## 3.3 功能与性能设计

### 功能实现方案

#### 训练模块

**功能**：生成聚类中心和随机正交矩阵

**实现细节**：

1. **随机正交矩阵生成**：
   - 使用Givens旋转算法
   - 矩阵维度：dim × dim
   - 存储到设备内存

2. **K-Means聚类**：
   - 复用IVFFlat的一阶段检索能力
   - 支持K-Means++初始化（可配置）
   - 生成nlist个聚类中心

3. **质心预处理**：
   - 对质心应用随机正交变换
   - 计算质心的L2范数
   - 计算质心的LUT查找表

**输入**：

- 训练数据集（n × dim）
- 配置参数（nlist, useRandomOrthogonalMatrix, matrixSeed等）

**输出**：

- 聚类中心（nlist × dim）
- 随机正交矩阵（dim × dim）
- 质心LUT（nlist × lut_dim）

#### 入库模块

**功能**：对向量进行RabitQ编码并添加到倒排索引

**实现细节**：

1. **向量预处理**：
   - 对向量应用随机正交变换
   - 计算向量的L2范数

2. **RabitQ编码**：
   - 按 `quantBits` 将向量编码为 1/2/4-bit 表示
   - 编码长度：`dim * quantBits / 8` 字节
   - 2/4-bit 采用每 8 维交错存放的 bit-plane 格式

3. **预计算常数**：
   - 计算L1常数：偏移项
   - 计算L2常数：缩放项

4. **倒排索引构建**：
   - 根据向量所属聚类，添加到对应倒排列表
   - 存储编码数据、索引、预计算常数

**输入**：

- 向量数据（n × dim）
- 向量ID（n）

**输出**：

- 编码数据（倒排列表）
- 索引映射

#### 检索模块

**功能**：执行两级检索，返回TopK结果

**实现细节**：

##### L1阶段（聚类筛选）

1. **查询向量预处理**：
   - 对查询向量应用随机正交变换
   - 计算查询向量的L2范数
   - 计算查询向量的LUT查找表

2. **距离计算**：
   - 计算查询向量与所有质心的距离
   - 使用LUT加速计算

3. **TopNprobe选择**：
   - 选择距离最近的nprobe个聚类
   - 返回聚类索引和距离

**输入**：

- 查询向量（nq × dim）
- nprobe参数

**输出**：

- 选定的聚类索引（nq × nprobe）
- 聚类距离（nq × nprobe）

##### L2阶段（距离计算）

1. **距离计算**：
   - 在选定的nprobe个聚类中
   - 使用LUT查表计算查询向量与库向量的距离
   - 公式：`distance = L2_norm + L1_offset + LUT_lookup`

2. **TopK选择**：
   - 对距离结果进行TopK排序
   - 返回TopK个最近邻的索引和距离

**输入**：

- 查询向量（nq × dim）
- 选定的聚类索引
- TopK参数

**输出**：

- TopK距离（nq × k）
- TopK索引（nq × k）

##### Refine阶段（可选）

1. **精排计算**：
   - 对TopK结果进行重排序
   - 使用原始向量计算精确距离
   - 提高召回率

**输入**：

- TopK结果
- 原始向量库

**输出**：

- 精排后的TopK结果

#### 辅助功能

**copyTo/copyFrom**：

- 实现CPU与NPU数据互拷
- 支持完整的状态转移

**删除/更新**：

- 支持删除指定ID的向量
- 支持更新向量数据

**多设备支持**：

- 支持多卡部署
- 自动管理ID到设备的映射

### 算子设计

#### 离线模型与实际算子映射

`ivfrabitq_generate_model.py` 根据 `dim`、`nlist`、距离类型和 `quant_bits` 生成以下模型。Host 侧的
Center/Index/Query 封装会复用同一类底层算子，通过不同 shape 区分执行阶段。

| 阶段 | 实际算子名 | 多 bit 变化 |
| --- | --- | --- |
| 中心、底库和查询旋转及 L2 | `RotateAndL2AtFP32` | 无，输入仍为 FP32 原始向量 |
| 中心和查询 LUT | `MatmulAtFP32` | LUT 列数变为 `256 * quant_bits` |
| 底库编码与因子预计算 | `IndexCodeAndPrecompute` | 新增 `quant_bits` 属性；编码输出宽度变为 `dim * quant_bits / 8` |
| IVF-RaBitQ L2 距离 | `DistanceIVFRabitqL2FP32` | 新增 `quant_bits` 属性；压缩码和 LUT shape 随 bit 数变化 |
| A5/SIMT L2 距离 | `DistanceIVFRabitqL2FP32Simt` | 当前只支持 1-bit |
| L1 距离 | `DistanceFlatL2MinsAtFP32` | 不受量化位数影响 |
| L2 TopK | AICPU TopK/调度算子；满足现有切换条件时可走 `TopkIvfRabitqAicore` | 不受编码格式影响，消费 L2 距离结果；大 TopK 路径需独立验证 |

`quant_bits` 同时进入 `IndexCodeAndPrecompute` 和 `DistanceIVFRabitqL2FP32` 的 tiling 数据。生成模型的
bit 数必须与运行时 `AscendIndexIVFRaBitQConfig::quantBits` 完全一致。

#### 训练阶段算子

**1. 中心旋转L2算子 (CenterRotateL2Op)**：

- **功能**：对聚类中心应用随机正交变换，并计算L2范数
- **输入**：
  - `centroid`：聚类中心（nlist × dim）
  - `vectorSize`：向量大小
  - `matrix`：随机正交矩阵（dim × dim）
- **输出**：
  - `rotateCentroid`：旋转后的聚类中心
  - `centroidl2`：聚类中心的L2范数

**2. 中心LUT算子 (CenterLUTOp)**：

- **功能**：计算聚类中心的LUT查找表
- **输入**：
  - `centroid`：聚类中心
  - `vectorSize`：向量大小
- **输出**：
  - `centroidslut`：聚类中心的LUT

#### 入库阶段算子

**3. 索引旋转L2算子 (IndexRotateL2Op)**：

- **功能**：对入库向量应用随机正交变换，并计算L2范数
- **输入**：
  - `index`：入库向量
  - `vectorSize`：向量大小
  - `matrix`：随机正交矩阵
- **输出**：
  - `rotateIndex`：旋转后的向量
  - `indexl2`：向量的L2范数

**4. 索引编码预计算算子 (IndexCodeAndPreComputeOp)**：

- **功能**：按 `quant_bits` 对向量进行 1/2/4-bit RaBitQ 编码，并计算预计算常数
- **输入**：
  - `vectorSize`：向量大小
  - `rotateIndex`：旋转后的向量
  - `indexl2`：向量的L2范数
- **属性**：
  - `metric_type`：距离类型
  - `quant_bits`：每维量化位数，取值为 1、2、4
- **输出**：
  - `codeVec`：编码后的向量，shape 为 `n × (dim * quant_bits / 8)`
  - `indexl1`：L1预计算常数
  - `indexl2`：L2预计算常数

#### 检索阶段算子

**5. 查询旋转L2算子 (QueryRotateL2Op)**：

- **功能**：对查询向量应用随机正交变换，并计算L2范数
- **输入**：
  - `batch`：批大小
  - `queries`：查询向量
  - `vectorSize`：向量大小
  - `matrix`：随机正交矩阵
- **输出**：
  - `rotateQueries`：旋转后的查询向量
  - `queryl2`：查询向量的L2范数

**6. 查询LUT算子 (QueryLUTOp)**：

- **功能**：计算查询向量的多 bit LUT 查找表
- **输入**：
  - `batch`：批大小
  - `queries`：查询向量
  - `matrix`：随机正交矩阵
- **输出**：
  - `querieslut`：查询向量的LUT，列数为 `256 * quant_bits`

**7. L1距离算子 (L1DistOp)**：

- **功能**：计算查询向量与所有聚类中心的距离
- **输入**：
  - `batch`：批大小
  - `queries`：查询向量
  - `queryl2`：查询向量L2范数
  - `centroidslut`：聚类中心LUT
  - `centroidl2`：聚类中心L2范数
- **输出**：
  - `l1Dist`：L1阶段距离结果

**8. L1 TopK算子 (L1TopkOp)**：

- **功能**：选择距离最近的nprobe个聚类
- **输入**：
  - `dists`：距离矩阵
  - `nprobe`：选择的聚类数量
- **输出**：
  - `topkIndices`：TopK聚类索引
  - `topkDists`：TopK距离

**9. L2距离算子 (L2DistOp)**：

- **功能**：在选定聚类中按 `quant_bits` 对各 bit-plane 查表并计算查询向量与库向量的距离
- **输入**：
  - `queryL2Vec`：查询向量L2范数
  - `subQuerylut`：查询向量LUT
  - `centroidslut`：聚类中心LUT
  - `subQueryid`：查询ID
  - `subCentroidsid`：聚类ID
  - `subCentroidsl2`：聚类中心L2范数
  - `codeVec`：编码向量
  - `subOffset`：偏移量
  - `subBaseSize`：基础大小
  - `subIndexl2`：索引L2常数
  - `subIndexl1`：索引L1常数
- **属性**：
  - `metric_type`：距离类型
  - `quant_bits`：每维量化位数，取值为 1、2、4
- **输出**：
  - `subDis`：距离结果
  - `subVcMaxDis`：最大距离
  - `subOpFlag`：操作标志

**10. L2 TopK算子 (L2TopkOp)**：

- **功能**：对距离结果进行TopK排序
- **输入**：
  - `batch`：批大小
  - `distResult`：距离结果
  - `k`：TopK参数
- **输出**：
  - `topkIndices`：TopK索引
  - `topkDists`：TopK距离

#### 算子实现要点

**1. 数据格式**：

- 使用ND格式存储数据，提高内存访问效率
- 数据按burst长度对齐（A2/A3: 64, Ascend950: 32/64）
- 压缩码长度由 `dim * quant_bits / 8` 推导
- 多 bit 查询和中心 LUT 的列数为 `256 * quant_bits`

**2. 并行策略**：

- 使用多核并行计算
- 使用向量指令加速计算

**3. 内存管理**：

- 使用内存池管理算子内存
- 支持异步执行模式

**4. 错误处理**：

- 检查输入参数合法性
- 检查内存分配是否成功
- 检查计算结果是否合法

### 性能优化策略

#### 1. 内存优化

- **紧凑编码**：使用 1/2/4-bit 编码，在召回率与内存占用之间提供可配置权衡
- **内存对齐**：数据按burst长度对齐，提高内存访问效率
- **连续存储**：倒排列表连续存储，减少内存碎片

#### 2. 计算优化

- **LUT加速**：预计算查找表，减少检索时计算量
- **批量处理**：支持batch模式，充分利用NPU并行能力
- **两级检索**：L1快速筛选，L2精确计算，平衡性能和精度

#### 3. 流水线优化

- **计算与传输重叠**：重叠计算和数据传输，提高吞吐量
- **异步执行**：使用异步执行模式，减少等待时间

#### 4. 平台适配优化

- **Atlas A2推理系列产品/Atlas A3推理系列产品平台**：使用高burst长度（64）
- **Ascend 950PR系列产品平台**：根据batch size动态调整burst长度（32/64）

### 影响范围

#### 新增文件

1. **AscendIndexIVFRaBitQ.h**：
   - 对外API接口定义
   - 配置结构体定义

2. **AscendIndexIVFRaBitQImpl.h/cpp**：
   - 实现类定义
   - 核心逻辑实现

3. **IndexIVFRaBitQ.h/cpp**：
   - 底层索引实现
   - 算子调用逻辑

4. **算子实现文件**：
   - 中心旋转L2算子
   - 中心LUT算子
   - 索引旋转L2算子
   - 索引编码预计算算子
   - 查询旋转L2算子
   - 查询LUT算子
   - L1距离算子
   - L1 TopK算子
   - L2距离算子
   - L2 TopK算子

#### 修改文件

1. **AscendIndexIVF.h/cpp**：
   - 新增IVFRaBitQ相关接口

2. **CMakeLists.txt**：
   - 新增IVFRaBitQ相关编译配置

3. **测试文件**：
   - 新增单元测试文件

#### 不影响

- 现有其他算法的实现
- 现有其他平台的功能
- 对外API接口的兼容性

### 多Bit量化扩展

#### 概述与动机

在 1-bit 量化基础上增加 2-bit 和 4-bit 量化能力，通过 `AscendIndexIVFRaBitQConfig::quantBits` 在 1、2、4 之间选择每维量化位数。目标不是提升量化距离算子的峰值性能，而是提高候选排序质量，使检索任务在达到相同精排召回率时使用更小的候选 TopK，从而降低 TopK、结果搬运和精排开销，改善端到端 QPS。

1-bit 每维只保留符号信息，量化误差较大，高召回场景需要较大的候选 TopK 再精排，带来 AICPU TopK 耗时、候选搬运量和下游精排计算量的压力。多 bit 用额外的 AICore 计算和 HBM 占用换取更小的候选 TopK。

目标：

- 保持 1-bit 默认行为和已有索引兼容；
- 在 A3/Cube L2 路径支持 2-bit、4-bit 编码和检索；
- 复用既有训练、入库、删除、更新和检索接口，支持 `copyTo`/`copyFrom` 和 Faiss I/O；
- 以"达到相同精排后 Recall 时的端到端 QPS"为主要评价口径。

非目标：不支持 2/4-bit 的 IP 距离、Ascend 310P/A5 SIMT 路径和 CPU 直接检索落盘对象；不改变 TopK 算子功能边界；不把原始向量纳入多 bit 索引；不承诺索引文件跨 `quantBits` 转换或混用。

#### 配置与支持范围

```cpp
struct AscendIndexIVFRaBitQConfig : public AscendIndexIVFConfig {
    // 每维量化位数。默认 1，当前可选 1、2、4。
    int quantBits = 1;
};
```

| 项目 | 1-bit | 2-bit | 4-bit |
| --- | --- | --- | --- |
| 距离类型 | L2、IP（保持原能力） | L2 | L2 |
| 当前平台 | 原有支持范围 | A3/Cube | A3/Cube |
| 当前维度 | 原有支持范围 | 64、128、256、384、512 | 64、128、256 |
| `copyTo/copyFrom` | 支持 | 支持 | 支持 |

配置语义：

- 未设置时默认为 1，保持源码兼容；训练、入库、检索、`copyTo` 和 `copyFrom` 必须使用同一个值；
- `quantBits` 是索引格式的一部分，入库后不能修改；加载索引时调用方需显式构造相同 bit 数的 Ascend 索引；
- `nlist`、`nprobe`、候选 TopK 和多卡分片行为沿用现有接口与约束。

#### 多bit量化编码

对向量 `x` 及其所属聚类中心 `c`，沿用现有流程计算旋转残差 `r = R(x - c)`。对 2/4-bit 计算残差均方根并推导量化步长：

```text
rms = sqrt(sum(r_i^2) / d)
quant_step = rms * 2 / quantBits
```

即 2-bit 步长为 `1.0 * rms`，4-bit 步长为 `0.5 * rms`。令 `b=quantBits`，`M=2^(b-1)`，每维先计算幅值级别再映射为无符号码：

```text
m_i = clip(floor(abs(r_i) / quant_step), 0, M - 1)
u_i = M + m_i       , r_i >= 0
u_i = M - 1 - m_i   , r_i < 0
```

距离计算时中心化：`z_i = u_i - (2^b - 1) / 2`。对应量化级别：

| 位数 | 中心化级别 | 当前步长 | 外围覆盖 |
| --- | --- | --- | --- |
| 2-bit | -1.5、-0.5、0.5、1.5 | `1.0 * rms` | `1.5 * rms` |
| 4-bit | -7.5、-6.5、…、7.5 | `0.5 * rms` | `3.75 * rms` |

1-bit 保持原有符号量化分支，不套用上述多 bit 步长逻辑。

bit-plane 布局：每 8 个维度为一组，每个 bit-plane 使用 1 字节表示这 8 个维度对应位，组内按从 MSB 到 LSB 存储，压缩码长度为 `dim * quantBits / 8` 字节。该布局使距离算子继续使用 8-bit 地址查表：每个 plane 取一个字节作为 LUT 下标，再按位权累加各 plane 的查表结果。

#### 距离估计

每个向量除压缩码外，继续保存两个 FP32 校正因子，沿用现有 RaBitQ 距离估计框架：

```text
factor = 2 * ||x-c||^2 / <x-c, z>
distance = ||x-c||^2 + ||q-c||^2 - factor * <q-c, z>
```

实现中通过 bit-plane LUT 累加得到 `<q-c, u>`，再减去无符号码中心偏置：

```text
center_offset = (2^quantBits - 1) / 2
<q-c, z> = <q-c, u> - center_offset * sum(q-c)
```

各 plane 的 LUT 权重依次为 `2^(b-1-plane)`。2-bit 和 4-bit 共用同一个距离核控制流程，仅 shape、plane 数和位权不同。

#### 数据布局与容量

NPU 每向量保存 `[dim * quantBits / 8 字节压缩码][4B L2因子][4B L1因子][8B ID]`。以 `dim=128` 为例：

| 位数 | 压缩码 | code + 两个因子 | 加 ID 后的理论主数据 |
| ---: | ---: | ---: | ---: |
| 1-bit | 16 B | 24 B | 32 B/向量 |
| 2-bit | 32 B | 40 B | 48 B/向量 |
| 4-bit | 64 B | 72 B | 80 B/向量 |

表中未包含倒排结构、聚类中心、LUT、workspace 和运行时开销，实际 HBM 以 `npu-smi` 稳态增量为准。

每 8 维使用 256 项 LUT，查询和中心 LUT 列数为 `256 * quantBits`，随 bit 数线性增长。当前 UB 设计决定了 2-bit 最大支持 `dim=512`、4-bit 最大支持 `dim=256`。

#### 算子改动

- **IndexCodeAndPrecompute**：新增必选属性 `quant_bits`；输出 shape 从 `n * dim / 8` 扩展为 `n * dim * quant_bits / 8`；`quantBits=1` 保留原有符号编码，2/4-bit 计算 RMS、幅值级别、符号码和 bit-plane；继续输出两个 FP32 校正因子。
- **MatmulAtFP32**：LUT 变换矩阵列数从 256 扩展为 `256 * quant_bits`；查询和聚类中心分别生成与 bit-plane 顺序一致的 LUT。
- **DistanceIVFRabitqL2FP32**：新增必选属性 `quant_bits`；从 `codeLength` 和 `quant_bits` 恢复实际维度；对每个 bit-plane 按位权查表累加并减去中心偏置；距离输出、过滤、调度和 TopK 接口保持不变。
- **离线模型生成**：`ivfrabitq_generate_model.py` 增加 `--quant-bits`，三个 bit 数的模型生成到同一个 `op_models` 目录，不应在各次生成之间清空目录。

```bash
python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910_9392 -m L2 --quant-bits 1
python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910_9392 -m L2 --quant-bits 2
python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910_9392 -m L2 --quant-bits 4
```

#### 索引持久化

`copyTo` 使用 `faiss::IndexIVFRaBitQ` 作为 CPU 落盘载体：设置 `rabitq.nb_bits = quantBits`，倒排列表按 `[code][L2 factor][L1 factor]` 保存；2/4-bit 在有效载荷后附加 12 字节 Ascend 格式标记和版本号（只用于格式识别，不上传 NPU、不参与距离计算）。以 `dim=128` 为例：

| 位数 | NPU payload | Faiss 落盘 carrier | 加 8B ID 后 |
| ---: | ---: | ---: | ---: |
| 1-bit | 24 B | 24 B | 32 B/向量 |
| 2-bit | 40 B | 52 B | 60 B/向量 |
| 4-bit | 72 B | 84 B | 92 B/向量 |

`copyFrom` 必须校验文件 `d`、`nlist`、`rabitq.nb_bits`、`code_size` 与目标索引一致，2/4-bit trailer magic 和版本合法，不满足时直接失败，避免误加载到错误的算子模型或数据布局。

多设备 `copyFrom` 延续 shard 语义：按负载均衡把向量分配到各设备，不在每卡复制全库。性能测试工具中的 `search-mode=replica` 是每卡加载一份全库再切分 query batch，不是 `copyFrom` 的 replica 模式，两类结果不能混为一谈。

#### 兼容性与错误处理

- `quantBits` 默认值为 1，未设置该字段的现有调用不需要修改代码；新增公开配置字段发布前需按项目 ABI 策略评估二进制兼容；
- 原有 1-bit 索引格式不变；2/4-bit 使用扩展载体格式，旧版本 IndexSDK 不能加载，2-bit 与 4-bit 索引不能互换；
- Faiss 原生多 bit RaBitQ 索引即使 `nb_bits` 相同，也会因缺少 Ascend trailer 被拒绝。

参数校验：

| 条件 | 处理 |
| --- | --- |
| `quantBits` 不为 1、2、4 | 初始化失败 |
| 2/4-bit 使用 IP | 初始化失败 |
| 2-bit `dim > 512` / 4-bit `dim > 256` | 初始化失败 |
| 2/4-bit 使用 A5/310P | 初始化失败 |
| 文件 bit 数、码长或格式标记与配置不一致 | `copyFrom` 失败 |

升级与回滚：上线初期保持默认 `quantBits=1`；2/4-bit 使用新文件名；离线模型同时保留 1/2/4-bit；发生问题时可切回原 1-bit 索引和配置；索引不存在跨 bit 原地升级，切换 bit 数必须重新编码底库。

#### 性能评价口径

多 bit 不改变 TopK 接口和返回语义，采用两阶段口径：`candidate_topk` 为 RaBitQ 初选返回数量，`target_topk` 为精确重排后的最终返回数量。参数选型不应直接比较同一 `candidate_topk` 下的 QPS，而应：固定 `target_topk`，对每个 bit 扫描 `candidate_topk`，找到达到目标精排召回率的最小候选数，再比较等召回配置的端到端 QPS、时延和 HBM。

成本与收益：

- 成本：2-bit 压缩码容量约为 2 倍、4-bit 约为 4 倍；LUT 宽度线性增加；距离核处理更多 bit-plane；HBM 带宽和加载时间增加；
- 收益：候选排序更接近精确距离排序，相同候选 TopK 下候选覆盖率提高，达到相同精排 Recall 时所需候选 TopK 降低，TopK、搬运和精排成本下降；
- 量化位数只增加 L2 量化距离部分的工作量，若原链路主要受 AICPU TopK 或固定开销限制，同候选 TopK 下 QPS 不一定明显下降；但这不是性能保证，必须以目标部署规模上的等召回实测为准。

#### 测试与验收要点

功能测试至少覆盖：1/2/4-bit 初始化、训练、add、search 与全部 batch；2/4-bit 维度边界；非法 bit、距离、平台和维度；多次 add、删除和更新；单卡与多卡 shard；`copyTo -> write_index -> read_index -> copyFrom` 前后结果一致；bit、维度、nlist 或 carrier 不匹配时加载失败。

正确性验收：同 bit 建库直接检索与落盘重载后结果一致；1-bit 与基线一致；2/4-bit 距离无 NaN/Inf、返回 ID 合法且按距离升序；候选 TopK 增大时 `cand_cov` 原则上不下降；召回测试查询数建议不少于 1000。

性能验收：固定底库、查询、GT、`nlist`、`nprobe`、batch 和最终 TopK，分别扫描各 bit 的候选 TopK，先找到等召回配置再比较 QPS 和时延，同时记录 HBM。单卡矩阵覆盖 bit × `nprobe` 低中高三档 × `nq`（1~256）× 候选 TopK 档位；多卡分别验证 shard（容量与吞吐）与 replica（并行吞吐与扩展效率）；HBM 记录 baseline、稳态、P95 和 peak。

验收判定顺序：功能与持久化正确 → 1-bit 无回归 → 2/4-bit 能以更小候选 TopK 达到 1-bit 基线等召回 → 等召回配置下端到端 QPS 不劣于 1-bit 大候选配置 → HBM、文件大小和加载时间满足部署约束。若 4-bit 召回收益无法抵消额外成本，优先选择 2-bit。

主要风险与应对：

| 风险 | 应对措施 |
| --- | --- |
| 4-bit HBM 增长导致单卡放不下 | 部署前测稳态 HBM；改用 shard 或 2-bit |
| LUT UB 随 bit 数增加，高维不支持 | 保留当前维度限制；后续设计分块 LUT |
| 同 K 下 4-bit QPS 下降 | 按等召回比较，并缩小候选 K |
| Faiss 原生格式与 Ascend 格式混用 | trailer magic、version、code_size 多重校验 |

备选方案：继续使用 1-bit 并优化大 TopK（索引最小、距离核最快，但 TopK 和精排成本高）；2-bit 是精度、HBM 和距离计算之间的折中，为优先验证的产品化候选；4-bit 候选排序质量最高但成本最大，适用于高召回且容量充足的场景。

## 3.4 安全隐私与DFX设计

### 安全隐私

- 不涉及用户敏感数据处理
- 算子实现遵循安全编码规范

### 兼容性

- API 接口保持向后兼容
- 支持与现有 Index 接口无缝对接

### 可维护性

- 代码结构清晰，遵循项目编码规范
- 提供详细的注释和文档

### 可测试性

- 提供完整的单元测试
- 提供性能基准测试
- 提供精度验证测试

### 可靠性

- 算法计算结果正确性保证
- 异常情况处理（内存不足、参数错误等）

## 3.5 编程与调用设计

### 3.5.1 编程模型基本设计

#### 开发环境

- 硬件平台: Ascend NPU
- 软件环境: CANN 工具链， ACL Runtime

#### 开发约束

- 支持 C++11 及以上标准
- 需要安装 CANN 软件栈
- 需要配置 NPU 驱动和固件

#### 可验收设计

- 功能验收: 通过单元测试和集成测试
- 性能验收: 通过性能基准测试
- 精度验收: 与 CPU 参考实现对比

### 3.5.2 接口定义与设计

#### 配置结构体

**AscendIndexIVFRaBitQConfig**：

```cpp
struct AscendIndexIVFRaBitQConfig : public AscendIndexIVFConfig {
    // 构造函数
    AscendIndexIVFRaBitQConfig();
    AscendIndexIVFRaBitQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);
    AscendIndexIVFRaBitQConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);
    AscendIndexIVFRaBitQConfig(std::vector<int> devices, bool useRandomOrthogonalMatrix_,
                               bool needRefine_, int matrixSeed_, float alpha_,
                               int64_t resourceSize = IVF_DEFAULT_MEM);

    // 配置参数
    bool useRandomOrthogonalMatrix;  // 是否使用随机正交矩阵（默认true）
    bool needRefine;                 // 是否使用Refine精排（默认false）
    int matrixSeed;                  // 随机正交矩阵的种子（默认12345）
    float refineAlpha;               // 精排倍数（默认2）
    int quantBits = 1;               // 每维量化位数，可选1、2、4
};
```

**参数说明**：

- `devices`：设备ID列表，支持多设备
- `resourceSize`：资源池大小，默认IVF_DEFAULT_MEM
- `useRandomOrthogonalMatrix`：是否使用随机正交矩阵，建议启用
- `needRefine`：是否启用Refine精排，高召回率场景建议启用
- `matrixSeed`：随机种子，保证可复现性
- `refineAlpha`：精排倍数，TopK × refineAlpha个候选进行精排
- `quantBits`：每维量化位数，默认 1；2/4-bit 当前仅支持 A3/Cube L2 路径

#### 主要类接口

**AscendIndexIVFRaBitQ**：

```cpp
class AscendIndexIVFRaBitQ : public AscendIndexIVF {
public:
    // 构造函数
    AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist,
                         AscendIndexIVFRaBitQConfig config = AscendIndexIVFRaBitQConfig());

    // 析构函数
    virtual ~AscendIndexIVFRaBitQ();

    // 禁用拷贝构造和赋值
    AscendIndexIVFRaBitQ(const AscendIndexIVFRaBitQ&) = delete;
    AscendIndexIVFRaBitQ& operator=(const AscendIndexIVFRaBitQ&) = delete;

    // 核心方法
    void train(idx_t n, const float *x) override;
    void copyFrom(const faiss::IndexIVFRaBitQ *index);
    void copyTo(faiss::IndexIVFRaBitQ *index) const;
    void remove_ids(size_t n, const idx_t* ids);
    std::vector<idx_t> update(idx_t n, const float* x, const idx_t* ids);

protected:
    std::shared_ptr<AscendIndexIVFRaBitQImpl> impl_;
};
```

#### 核心方法说明

**构造函数**：

- `dims`：向量维度
- `metric`：距离度量类型，支持METRIC_L2
- `nlist`：聚类中心数量
- `config`：配置参数

**训练方法**：

- `train(n, x)`：训练聚类中心
  - `n`：训练向量数量
  - `x`：训练向量数据

**数据拷贝方法**：

- `copyFrom(index)`：从CPU索引拷贝到NPU索引
- `copyTo(index)`：从NPU索引拷贝到CPU索引

**数据操作方法**：

- `remove_ids(n, ids)`：删除指定ID的向量
- `update(n, x, ids)`：更新指定ID的向量

**继承的方法**：

- `add_with_ids(n, x, ids)`：添加向量到索引
- `search(nq, x, k, distances, labels)`：检索TopK结果
- `setNumProbes(nprobe)`：设置检索时访问的聚类数量

### 3.5.3 编程手册设计

#### IVF-RaBitQ 算子模型生成

算子模型的 shape 与 `dim`、`nlist`、距离类型和 `quantBits` 相关。运行时配置使用哪一种 bit，必须提前生成相同 bit 数的模型。命令格式如下：

```bash
python3 ivfrabitq_generate_model.py \
  -d <dim> -c <coarse_centroid_num> --cores <core_num> \
  -p <process_id> -pool <pool_size> -t <npu_type> \
  -m <metric_type> --quant-bits <quant_bits>
```

例如生成 A3 (可以通过 npu-smi info -t board -i 0 -c 0 命令进行查询，获取NPU Name信息，910_NPU Name即是npu_type的取值)、dim=128、nlist=1024、L2 的 1/2/4-bit 模型：

```bash
python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910_9382 -m L2 --quant-bits 1
python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910_9382 -m L2 --quant-bits 2
python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910_9382 -m L2 --quant-bits 4
```

三次生成应使用同一个 `op_models` 输出目录，不要在各次生成之间清空该目录。相关限制如下：

- `quant_bits` 取值为 1、2、4，默认值为 1；
- 2/4-bit 仅支持 L2；
- 2-bit 支持 dim=64、128、256、384、512；
- 4-bit 支持 dim=64、128、256；
- 2/4-bit 仅支持 A3/Cube 距离路径，不支持 310P 和 Ascend950/A5 SIMT 路径；
- 修改 bit 数、dim、nlist、距离类型或升级算子输入后，都需要重新生成匹配的模型。

需要在现有《IndexSDK 用户指南》中新增以下章节：

#### IVF-RabitQ 算法使用指南

1. **环境准备**：
   - 安装 CANN 工具链
   - 配置 NPU 驱动和固件
   - 设置环境变量

2. **使用示例**：

   ```cpp
   // 创建 IVF-RabitQ 索引
   int dim = 128;
   int nlist = 1024;
   faiss::MetricType metric = faiss::METRIC_L2;

   faiss::ascend::AscendIndexIVFRaBitQConfig config({0});  // 使用设备0
   config.useRandomOrthogonalMatrix = true;  // 使用随机正交矩阵
   config.matrixSeed = 12345;  // 随机种子
   config.quantBits = 2;  // 每维量化位数，可选1、2、4
   config.needRefine = false;  // 不使用Refine

   faiss::ascend::AscendIndexIVFRaBitQ index(dim, metric, nlist, config);

   // 训练
   int trainNum = nlist * 40;
   index.train(trainNum, trainData);

   // 添加向量
   index.add_with_ids(ntotal, baseData, ids);

   // 设置检索参数
   int nprobe = 64;
   index.setNumProbes(nprobe);

   // 检索
   int k = 10;
   index.search(nq, queryData, k, distances, labels);
   ```

3. **配置参数说明**：
   - `useRandomOrthogonalMatrix`：是否使用随机正交矩阵（默认true）
   - `matrixSeed`：随机正交矩阵的种子（默认12345）
   - `needRefine`：是否使用Refine精排（默认false）
   - `refineAlpha`：精排倍数（默认2）
   - `useKmeansPP`：是否使用K-Means++初始化（默认false）

4. **参数调优建议**：
   - **nlist选择**：建议为sqrt(ntotal)到ntotal/1000之间
   - **nprobe选择**：建议为nlist的1/16到1/32之间
   - **Refine使用**：当召回率要求高时启用，建议refineAlpha=2

5. **常见问题与解决方案**：
   - **精度不达预期**：
     - 检查是否使用了随机正交矩阵
     - 尝试启用Refine精排
     - 增加nprobe值

   - **性能不达预期**：
     - 检查batch size是否合理
     - 检查nprobe是否过大
     - 检查NPU利用率

   - **内存占用过大**：
     - 检查nlist是否过大
     - 检查向量维度是否合理

# 4. 缺点和风险

## 4.1 潜在风险

### Breaking Change

- 无 Breaking Change，完全向后兼容

### 性能风险

- 不同 NPU 平台性能可能存在差异，需要充分测试
- 大规模向量库可能存在内存压力

### 复杂度提升

- 新增算法实现，维护成本略有提升

## 4.2 负面影响

### 对现有功能的影响

- 不影响现有其他算法的实现
- 不影响现有其他平台的功能

### 对用户的影响

- 用户需要升级到支持 IVF-RabitQ 的版本
- 需要重新训练索引

## 4.3 实现成本

### 开发成本

- 预计开发工作量: 8 人周
- 主要工作: 算法实现、性能优化、测试验证

### 维护成本

- 长期维护成本中等
- 需要跟进 CANN 版本更新

## 4.4 应对措施

- 充分的单元测试和集成测试
- 性能基准测试和优化
- 详细的文档和示例代码
- 版本兼容性测试

# 5. 现有技术

## IndexSDK 现有实现

- AscendIndexIVFFlat 已在 IndexSDK 中实现，提供 IVF 基础框架
- 本提案复用现有 IVF 框架，结合 RabitQ 量化技术

## 学术研究

- RabitQ 论文: [Rapid and Accurate Binary Quantization](https://arxiv.org/abs/2405.12497)

# 6. 未解决问题

无

---

## 附录

### 参考资料

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)
- [IndexSDK 用户指南](../../zh/05_user_guide.md)
- [Ascend NPU 开发文档](https://www.hiascend.com/document)
- [RabitQ 论文](https://arxiv.org/abs/2405.12497)

### 术语表

- **IVF**: Inverted File System，倒排文件系统
- **RaBitQ**: Random Binary Quantization，随机二进制量化，一种高维向量量化算法
- **nlist**: 倒排列表数量，聚类中心数量
- **nprobe**: 检索时访问的倒排列表数量
- **LUT**: Look-Up Table，查找表
- **L1阶段**: 第一阶段检索，筛选最近的nprobe个聚类
- **L2阶段**: 第二阶段检索，在选定聚类中计算距离
- **Refine**: 精排机制，对TopK结果进行重排序

### 文档更新计划

- RFC 评审通过后，更新《IndexSDK 用户指南》
- 更新《快速开始指南》，添加 IVF-RabitQ 使用说明
