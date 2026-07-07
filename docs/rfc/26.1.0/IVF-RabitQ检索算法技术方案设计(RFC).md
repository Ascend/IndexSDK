# IVF-RabitQ检索算法技术方案设计(RFC)

**状态 (Status):** Draft
**作者 (Authors):** @xiangjie10
**创建日期 (Created):** 2026-05-06
**更新日期 (Updated):** 2026-05-07
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

- **A2/A3平台**：支持
- **A5平台**：支持

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
NPU (A2/A3/A5)
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
   - 将向量编码为紧凑的二进制表示
   - 编码维度：dim / 8 字节

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

- **编码方式**：将向量编码为二进制表示
- **编码维度**：dim / 8 字节（每个维度1bit）
- **编码过程**：
  1. 对向量应用随机正交变换
  2. 计算向量的L2范数
  3. 将向量归一化
  4. 对归一化向量进行二值量化

#### 3. LUT查找表

- **作用**：加速距离计算，避免重复计算
- **实现方式**：
  - 将向量按8位分段
  - 预计算每段的距离贡献
  - 检索时通过查表快速计算距离

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
   - 将向量编码为二进制表示
   - 编码维度：dim / 8 字节

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

- **功能**：对向量进行RabitQ编码，并计算预计算常数
- **输入**：
  - `vectorSize`：向量大小
  - `rotateIndex`：旋转后的向量
  - `indexl2`：向量的L2范数
- **输出**：
  - `codeVec`：编码后的向量
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

- **功能**：计算查询向量的LUT查找表
- **输入**：
  - `batch`：批大小
  - `queries`：查询向量
  - `matrix`：随机正交矩阵
- **输出**：
  - `querieslut`：查询向量的LUT

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

- **功能**：在选定聚类中计算查询向量与库向量的距离
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
- 数据按burst长度对齐（A2/A3: 64, A5: 32/64）

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

- **紧凑编码**：使用二进制编码，降低内存占用
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

- **A2/A3平台**：使用高burst长度（64）
- **A5平台**：根据batch size动态调整burst长度（32/64）

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
};
```

**参数说明**：

- `devices`：设备ID列表，支持多设备
- `resourceSize`：资源池大小，默认IVF_DEFAULT_MEM
- `useRandomOrthogonalMatrix`：是否使用随机正交矩阵，建议启用
- `needRefine`：是否启用Refine精排，高召回率场景建议启用
- `matrixSeed`：随机种子，保证可复现性
- `refineAlpha`：精排倍数，TopK × refineAlpha个候选进行精排

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

- RabitQ 论文: [Rapid and Accurate Binary Quantization](https://arxiv.org/abs/2405.12456)

# 6. 未解决问题

无

---

## 附录

### 参考资料

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)
- [IndexSDK 用户指南](../../zh/user_guide.md)
- [Ascend NPU 开发文档](https://www.hiascend.com/document)
- [RabitQ 论文](https://arxiv.org/abs/2405.12456)

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
