# TS FlatIP Int8Cos Operator Technical Design(RFC)

**状态 (Status):** Draft
**作者 (Authors):** @xiangjie10
**创建日期 (Created):** 2026-05-06
**更新日期 (Updated):** 2026-05-06
**相关 Issue/PR:** [#29](https://gitcode.com/Ascend/IndexSDK/issues/29)

---

# 1. 概述

## 1.1 简介

本提案旨在为 IndexSDK 补齐 A2/A3 NPU 平台上的 TS FlatIP Int8Cos 算子能力。TS 是时空库检视算法，FlatIP 和 Int8Cos 是向量检索中的常用的距离计算类型。通过本提案的实现，IndexSDK 将能够在 Ascend A2/A3 NPU 上，支持属性过滤场景下的高效检索，扩展向量检索链路，满足大规模向量检索场景的性能需求。

## 1.2 动机

### 背景

IndexSDK 为昇腾平台实现了高效的向量特征检索引擎，用户可以在此引擎上实现面向应用场景的检索系统。当前在 A2/A3 NPU 平台上，TS FlatIP Int8Cos 检索能力缺失，导致向量检索链路不完整，无法充分发挥 A2/A3 NPU 的硬件加速能力。

### 痛点

- A2/A3 NPU 平台缺少 TS FlatIP Int8Cos 检索支持，限制了向量检索场景的应用

- 用户在 A2/A3 平台上无法使用完整的向量检索能力

### 价值

- 补齐 A2/A3 平台的向量检索能力，完善 IndexSDK 的硬件支持矩阵

- 充分利用 NPU 加速能力，提升向量检索性能

## 1.3 目标

### 目标

- 支持 A2/A3 NPU 上的 TS FlatIP Int8Cos 计算算法

- 覆盖 dim=256 的向量维度场景

- 支持批量查询 batch=1-256

- 向量检索返回 topk=200

### 功能范围

- **支持的功能**：

  - 非共享mask模式（每个查询使用独立的过滤条件）

  - 自定义属性（customAttr）

  - 时空属性过滤（时间范围和token集合过滤）

- **不支持的功能**（与310P平台的差异）：

  - 共享mask模式（多个查询共享同一个过滤条件）

  - 额外相似度（extraScore）

  - scale量化

# 2. 用例分析

## 2.1 功能需求

### 核心功能

- **FlatIP**: 内积距离计算

- **Int8Cos**: Int8 量化余弦相似度计算

### 参数规格

- 向量维度: 256

- 库规模:

  - A2: 6000万 (6000w)

  - A3: 1.25亿 (1.25e)

- 批量大小: 1-256

- 返回数量: topk=200

### 功能要求

- 支持 int8 量化向量的向量建库与检索

- 支持余弦相似度排序

- 支持 batch 模式下的并发查询

- 返回结果保证排序正确性

## 2.2 性能需求

### 验收标准

- A2 上可完成 6000w 库规模、dim=256、batch=1-256、topk=200 的 TS FlatIP Int8Cos 检索，结果正确

- A3 上可完成 1.25e 库规模、dim=256、batch=1-256、topk=200 的 TS FlatIP Int8Cos 检索，结果正确

- 检索精度符合预期，与参考实现对齐

## 2.3 DFX 要求

### 可靠性

- 算子计算结果正确性保证

- 异常输入能够检查并输出异常

### 可测试性

- 提供单元测试用例

- 提供性能基准测试用例

- 提供精度对比测试用例

### 兼容性

- 与现有 AscendIndexTS 接口保持兼容

- 不影响现有其他算子的功能

# 3. 方案设计

## 3.1 总体方案

### 设计思路

TS FlatIP Int8Cos 算子已在 310P 平台上实现，本提案旨在将其扩展到 A2/A3 (910B) 平台。总体逻辑保持一致，对外接口不涉及修改。主要工作包括：

- 新增 AscendC 算子实现：针对 A2/A3 NPU 架构特性，开发 DistanceBatchMaskGenerator、DistanceBatchMaskGeneratorWithExtra、AscendcDistanceFlatIPMaxsWithMask、AscendcDistanceInt8CosWithMasks 等算子

- 平台适配：根据硬件平台类型（IsAscend910B）动态选择算子实现

- 模型生成工具：提供算子模型生成脚本，支持不同 batch size 的模型生成

**注意**：A2/A3 平台与 310P 平台存在功能差异，具体如下：

- A2/A3 平台仅支持非共享mask模式，不支持共享mask模式

- A2/A3 平台不支持额外相似度（extraScore）和scale量化功能

- A2/A3 平台支持自定义属性（customAttr）和时空属性过滤功能

### 技术架构

```text

用户应用层
    ↓
AscendIndexTS API
    ↓
TSFlatIP / TSInt8FlatCos
    ↓
距离计算算子 (FlatIP / Int8Cos)
    ↓
ACL Runtime
    ↓
A2/A3 NPU

```

### 核心流程

#### 建库阶段

1. 向量数据准备：将向量数据按照 ND 格式存储

2. 属性数据准备：将时间、空间等属性数据转换为 token 序列

3. 索引构建：构建 TS 索引结构，支持时空过滤

#### 检索阶段

1. **Mask 生成阶段**：

   - 输入：查询的时间、空间属性（token 序列）

   - 处理：根据硬件平台类型选择算子

     - 310P：使用 DistanceMaskGenerator / DistanceMaskGeneratorWithExtra

     - A2/A3：使用 AscendcDistanceBatchMaskGenerator / AscendcDistanceBatchMaskGeneratorWithExtra

   - 输出：mask 矩阵，标识哪些向量符合时空条件

2. **距离计算阶段**：

   - 输入：查询向量、库向量、mask 矩阵

   - 处理：根据硬件平台类型选择算子

     - 310P：使用 TIK 实现的 AICORE 算子

     - A2/A3：使用 AscendC 算子（AscendcDistanceFlatIPMaxsWithMask / AscendcDistanceInt8CosWithMasks）

   - 输出：距离矩阵

3. **TopK 选择阶段**：

   - 输入：距离矩阵

   - 处理：选择距离最小的 topk 个向量

   - 输出：检索结果（距离和索引）

#### 数据存储与Mask生成原理

##### 数据存储方式

###### 1. 底库向量存储

- **存储位置**：设备内存（Device Memory）

- **存储格式**：按block存储，每个block包含 `featureAttrBlockSize`（默认256K）个向量

- **数据类型**：

  - FlatIP: float16

  - Int8Cos: int8

###### 2. 时间信息存储

- **存储位置**：设备内存

- **存储格式**：按block存储，与向量存储对应

- **数据类型**：int32_t 数组

- **存储内容**：每个向量的时间戳

###### 3. 空间信息存储

- **存储位置**：设备内存

- **存储格式**：按block存储，与向量存储对应

- **数据类型**：分解为两部分存储

  - **tokenQ (Quotient)**: int32_t 数组，存储 `tokenId / 8 * 2`，表示在tokenBitSet中的字节偏移

  - **tokenR (Remainder)**: uint8_t 数组，存储 `1 << (tokenId % 8)`，表示在该字节中的位掩码

- **存储原理**：

  - tokenId范围：0 ~ tokenNum-1

  - 将tokenId分解为商和余数，便于快速进行位运算

  - 例如：tokenId = 10

    - tokenQ = 10 / 8 * 2 = 2（字节偏移）

    - tokenR = 1 << (10 % 8) = 1 << 2 = 4（位掩码）

##### Mask生成原理

###### 输入数据

1. **查询条件**：

   - queryTime: 查询的时间范围 [timesStart, timesEnd]

   - tokenBitSet: 待查询的token id集合，按位表示（每个bit代表一个token）

2. **库向量属性**：

   - attrTimes: 库向量的时间信息（int32_t数组）

   - attrTokenQs: 库向量的token Q值（int32_t数组，字节偏移）

   - attrTokenRs: 库向量的token R值（uint8_t数组，位掩码）

###### 处理流程

```text

对于每个库向量 i：

1. 时间过滤：
   if (attrTimes[i] >= timesStart && attrTimes[i] <= timesEnd) {
       timeMatch = true
   } else {
       timeMatch = false
   }

2. 空间过滤：
   // 使用tokenQ定位到tokenBitSet中的字节
   byteOffset = attrTokenQs[i]
   // 使用tokenR与tokenBitSet中的对应字节进行位运算
   if (tokenBitSet[byteOffset] & attrTokenRs[i]) {
       tokenMatch = true
   } else {
       tokenMatch = false
   }

3. 生成mask：
   mask[i] = timeMatch && tokenMatch ? 1 : 0

```

###### 优化设计

- **并行处理**：使用AICORE算子并行处理所有向量

- **批量处理**：支持批量查询，一次生成多个mask矩阵

- **内存对齐**：数据按8字节对齐，提高内存访问效率

## 3.2 技术选型

### 方案

基于 AscendC 实现 A2/A3 平台的算子，复用现有的 TSFlatIP 和 TSInt8FlatCos 框架

### 选择理由

1. **复用现有架构**：TS FlatIP Int8Cos 已在 310P 平台上实现，架构成熟稳定，复用可以降低开发风险

2. **AscendC 性能优势**：AscendC 是华为提供的算子开发框架，可以充分利用 A2/A3 NPU 的硬件特性，实现高性能算子

3. **平台适配简单**：通过运行时判断硬件平台类型（IsAscend910B），动态选择算子实现，对上层透明

### 技术细节

1. **Mask 生成算子**：

   - 310P：DistanceMaskGenerator / DistanceMaskGeneratorWithExtra（TIK 实现的 AICORE 算子）

   - A2/A3：AscendcDistanceBatchMaskGenerator / AscendcDistanceBatchMaskGeneratorWithExtra（AscendC 实现）

2. **距离计算算子**：

   - 310P：TIK 实现的 AICORE 算子

   - A2/A3：AscendcDistanceFlatIPMaxsWithMask / AscendcDistanceInt8CosWithMasks（AscendC 实现）

3. **算子模型生成**：

   - mask_generate_model.py：生成 Mask 生成算子模型

   - flat_generate_model.py：生成 FlatIP 算子模型

   - int8flat_generate_model.py：生成 Int8Cos 算子模型

4. **Mask 模式**：

   - **共享mask（shareAttrFilter = true）**：所有查询共享同一个attrFilter，只生成一次mask矩阵，适用于多个查询使用相同的过滤条件

   - **非共享mask（shareAttrFilter = false）**：每个查询有自己的attrFilter，为每个查询生成独立的mask矩阵，适用于每个查询使用不同的过滤条件

   - **平台支持**：

     - 310P：支持共享mask和非共享mask两种模式

     - A2/A3：当前仅支持非共享mask模式

5. **其他功能支持**：

   - **自定义属性（customAttr）**：所有平台都支持

   - **额外相似度（extraScore）**：仅310P平台支持

   - **scale量化**：仅310P平台支持

## 3.3 功能与性能设计

### 功能实现方案

#### Mask 生成算子

**功能**：根据查询的时间、空间属性生成 mask 矩阵，标识哪些向量符合时空条件

**实现细节**：

- **输入**：

  - queryTimes：查询时间范围（batch, 8）

  - tokenIds：token 序列（batch, maxTokenNum）

  - baseTimes：库向量时间范围（blockSize）

  - baseTokenIds：库向量 token 序列（blockSize）

  - extraMask：额外 mask（batch, blockSize）

- **处理流程**：

  1. 遍历每个查询的时间范围和 token 序列

  2. 与库向量的时间范围和 token 序列进行匹配

  3. 生成 mask 矩阵，符合条件的位置为 1，否则为 0

- **输出**：

  - mask：mask 矩阵（batch, blockSize）

**平台差异**：

- 310P：使用 TIK 实现的 AICORE 算子，通过 DistanceMaskGenerator / DistanceMaskGeneratorWithExtra 算子

- A2/A3：使用 AscendC 实现，通过 AscendcDistanceBatchMaskGenerator / AscendcDistanceBatchMaskGeneratorWithExtra 算子

#### FlatIP 算子

**功能**：计算查询向量与库向量的内积距离

**计算公式**：`distance = inner_product(query, base)`

**实现细节**：

- **输入**：

  - queries：查询向量（batch, dim）

  - mask：mask 矩阵（batch, blockSize）

  - base：库向量（blockSize, dim）

  - actualSize：实际向量数量

- **处理流程**：

  1. 利用 NPU 的矩阵乘法加速单元（Cube）

  2. 执行 query 与 base 的矩阵乘法：`result = query × base^T`

  3. 应用 mask 矩阵，过滤不符合时空条件的向量

  4. 取负值作为距离

- **输出**：

  - distances：距离矩阵（batch, blockSize）

**平台差异**：

- 310P：使用 TIK 实现的 AICORE 算子

- A2/A3：使用 AscendC 实现（AscendcDistanceFlatIPMaxsWithMask），充分利用 Cube 单元进行矩阵乘法加速

#### Int8Cos 算子

**功能**：计算查询向量与库向量的 Int8 量化余弦相似度

**计算公式**：`similarity = inner_product(query, base) / (||query|| * ||base||)`

**实现细节**：

- **输入**：

  - queries：查询向量（batch, dim），int8 量化

  - mask：mask 矩阵（batch, blockSize）

  - base：库向量（blockSize, dim），int8 量化

  - actualSize：实际向量数量

- **处理流程**：

  1. 对查询向量和库向量进行归一化处理

  2. 利用 NPU 的矩阵乘法加速单元（Cube）

  3. 执行 query 与 base 的矩阵乘法：`result = query × base^T`

  4. 应用 mask 矩阵，过滤不符合时空条件的向量

  5. 除以范数乘积得到余弦相似度

- **输出**：

  - similarities：相似度矩阵（batch, blockSize）

**平台差异**：

- 310P：使用 TIK 实现的 AICORE 算子

- A2/A3：使用 AscendC 实现（AscendcDistanceInt8CosWithMasks），充分利用 Cube 单元进行矩阵乘法加速

### 性能优化策略

#### 1. 内存优化

- **ND 格式存储**：向量数据采用 ND（N-dimensional）格式存储，避免内存重排，提高访问效率

- **内存对齐**：向量数据按照 CUBE_ALIGN（16）对齐，充分利用 NPU 的内存带宽

- **连续内存分配**：使用连续内存分配，减少内存碎片，提高内存利用率

#### 2. 计算优化

- **矩阵乘法加速**：利用 NPU 的 Cube 单元进行矩阵乘法加速，大幅提升计算性能

- **批量处理**：支持 batch 模式下的并发查询，充分利用 NPU 的并行计算能力

- **Tiling 策略**：

  - query 循环量限制最大为 128

  - codeNum 每次循环处理 512 个向量

  - 动态调整 tiling 参数，适应不同规模的计算

#### 3. 流水线优化

- **异步执行**：使用异步执行模式，减少等待时间

#### 4. Mask 生成优化

- **批量 Mask 生成**：支持批量生成 mask，减少算子调用次数

- **并行处理**：利用 NPU 的并行计算能力，并行处理多个查询的 mask 生成

### 影响范围

#### 新增文件

1. **Mask 生成算子**：

   - **AscendcDistanceBatchMaskGenerator**：批量 Mask 生成算子

     - 作用：根据查询的时间、空间属性生成 mask 矩阵

     - 输入：queryTimes（查询时间范围）、tokenIds（token 序列）、baseTimes（库向量时间范围）、baseTokenIds（库向量 token 序列）

     - 输出：mask 矩阵（标识哪些向量符合时空条件）

   - **AscendcDistanceBatchMaskGeneratorWithExtra**：带额外 mask 的批量 Mask 生成算子

     - 作用：在基础 mask 生成的基础上，支持额外的 mask 输入

     - 输入：queryTimes、tokenIds、baseTimes、baseTokenIds、extraMask（额外 mask）

     - 输出：合并后的 mask 矩阵

2. **FlatIP 算子**：

   - **AscendcDistanceFlatIPMaxsWithMask**：带 mask 的 FlatIP 距离计算算子

     - 作用：计算查询向量与库向量的内积距离，支持 mask 过滤

     - 输入：queries（查询向量）、mask（mask 矩阵）、base（库向量）、actualSize（实际向量数量）

     - 输出：distances（距离矩阵）

3. **Int8Cos 算子**：

   - **AscendcDistanceInt8CosWithMasks**：带 mask 的 Int8Cos 相似度计算算子

     - 作用：计算查询向量与库向量的 Int8 量化余弦相似度，支持 mask 过滤

     - 输入：queries（查询向量，int8 量化）、mask（mask 矩阵）、base（库向量，int8 量化）、actualSize（实际向量数量）

     - 输出：similarities（相似度矩阵）

4. **模型生成工具**：

   - **mask_generate_model.py**：Mask 生成算子模型生成工具

     - 作用：生成 Mask 生成算子的离线模型

     - 主要参数：-token（最大 token 数量）、-pool（进程池大小）

   - **flat_generate_model.py**：FlatIP 算子模型生成工具

     - 作用：生成 FlatIP 距离计算算子的离线模型

     - 主要参数：batch size、向量维度

   - **int8flat_generate_model.py**：Int8Cos 算子模型生成工具

     - 作用：生成 Int8Cos 相似度计算算子的离线模型

     - 主要参数：batch size、向量维度

#### 修改文件

1. **TSBase.cpp**：

   - 新增平台判断逻辑：根据 `IsAscend910B()` 选择不同的算子类型

   - 新增 `ASCENDC_ITI_MASK_GENERATOR` 和 `ASCENDC_ITI_MASK_WITH_EXTRA_GENERATOR` 索引类型

2. **TSFlatIP.cpp**：

   - 新增 `runAscendcDistMaskCompute` 函数调用

   - 新增平台判断逻辑：根据 `IsAscend910B()` 选择不同的距离计算算子

3. **TSInt8FlatCos.cpp**：

   - 新增 `runAscendcDistMaskCompute` 函数调用

   - 新增平台判断逻辑：根据 `IsAscend910B()` 选择不同的距离计算算子

4. **DistComputeOpsManager.h**：

   - 新增 AscendC 算子的注册和管理

5. **CMakeLists.txt**：

   - 新增 AscendC 算子的编译配置

#### 不影响

- 现有 310P 平台的实现

- 现有其他算子的功能

- 对外 API 接口

## 3.4 安全隐私与DFX设计

### 安全隐私

- 不涉及用户敏感数据处理

- 算子实现遵循安全编码规范

### 兼容性

- API 接口保持向后兼容

- 支持与现有 AscendIndexTS 接口无缝对接

### 可维护性

- 代码结构清晰，遵循项目编码规范

- 提供详细的注释和文档

### 可测试性

- 提供完整的单元测试

- 提供性能基准测试

- 提供精度验证测试

### 可靠性

- 算子计算结果正确性保证

- 异常情况处理（参数错误等）

## 3.5 编程与调用设计

### 3.5.1 编程模型基本设计

#### 开发环境

- 硬件平台: Ascend A2/A3 NPU

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

不修改AscendIndexTS对外接口

### 3.5.3 编程手册设计

需要在现有《IndexSDK 编程手册》中新增以下章节：

#### A2/A3 平台 TS FlatIP Int8Cos 算子使用指南

1. **环境准备**：

   - 安装 CANN 工具链（支持 A2/A3）

   - 配置 NPU 驱动和固件

   - 设置环境变量

2. **算子模型生成**：

   ```bash
   # 生成 Mask 生成算子模型
   python3 mask_generate_model.py -token 2500 -pool 16
   
   # 生成 FlatIP 算子模型
   python3 flat_generate_model.py
   
   # 生成 Int8Cos 算子模型
   python3 int8flat_generate_model.py
   ```

3. **使用示例**：

   ```cpp
   // 创建 TS FlatIP 索引
   faiss::ascend::AscendIndexTS index;
   index.Init(device_id, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
   
   // 添加向量
   std::vector<faiss::ascend::FeatureAttr> attrs(n);
   // 设置 attrs[i].time 和 attrs[i].tokenId
   index.AddFeature(n, vectors, attrs.data(), labels);
   
   // 检索
   faiss::ascend::AttrFilter attrFilter;
   attrFilter.timesStart = startTime;
   attrFilter.timesEnd = endTime;
   attrFilter.tokenBitSet = tokenBitSet;  // 按位表示的token集合
   attrFilter.tokenBitSetLen = tokenBitSetLen;
   
   index.Search(queryNum, queries, &attrFilter, true, k, labels, distances, validNums);

   ```

4. **性能调优建议**：

   - 合理设置 batch size（建议 32-128）

   - 优化向量维度对齐

   - 调整 topk 参数

#### 常见问题与解决方案

1. **算子模型加载失败**：

   - 检查 CANN 版本是否匹配

   - 检查算子模型文件是否存在

   - 检查环境变量是否正确设置

2. **性能不达预期**：

   - 检查向量数据是否对齐

   - 检查 batch size 是否合理

   - 检查 NPU 利用率

3. **精度问题**：

   - 检查向量归一化是否正确

   - 检查 int8 量化是否正确

   - 对比 CPU 参考实现结果

# 4. 缺点和风险

## 4.1 潜在风险

### Breaking Change

- 无 Breaking Change，完全向后兼容

### 性能风险

- A2/A3 平台性能可能存在差异，需要充分测试

- 大规模向量库可能存在内存压力

### 复杂度提升

- 增加了平台适配代码，维护成本略有提升

## 4.2 负面影响

### 对现有功能的影响

- 不影响现有其他平台的实现

- 不影响现有其他算子的功能

### 对用户的影响

- 用户需要升级到支持 A2/A3 的版本

- 需要重新生成算子模型文件

## 4.3 实现成本

### 开发成本

- 预计开发工作量: 4 人周

- 主要工作: 多个算子开发适配、性能优化、测试验证

### 维护成本

- 长期维护成本较低

- 需要跟进 CANN 版本更新

## 4.4 应对措施

- 充分的单元测试和集成测试

- 性能基准测试和优化

- 详细的文档和示例代码

- 版本兼容性测试

# 5. 现有技术

## IndexSDK 现有实现

- TSFlatIP TSInt8FlatCos 已在 IndexSDK 中实现，本提案复用现有架构，扩展支持 A2/A3

# 6. 未解决问题

无

---

## 附录

### 参考资料

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)

- [IndexSDK 用户指南](../../zh/user_guide.md)

- [Ascend NPU 开发文档](https://www.hiascend.com/document)

### 术语表

- **TS (TimeSpace)**: 时间空间过滤

- **FlatIP**: 内积距离计算

- **Int8Cos**: Int8 量化余弦相似度计算

- **A2/A3**: Ascend NPU 型号

### 文档更新计划

- RFC 评审通过后，更新《IndexSDK 用户指南》

- 更新《快速开始指南》，添加 A2/A3 平台使用说明
