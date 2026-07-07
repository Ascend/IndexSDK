# IVFPQ Algorithm A2/A3 Support Technical Design(RFC)

**状态 (Status):** Draft
**作者 (Authors):** @xiangjie10
**创建日期 (Created):** 2026-05-06
**更新日期 (Updated):** 2026-05-06
**相关 Issue/PR:** [#31](https://gitcode.com/Ascend/IndexSDK/issues/31)

---

# 1. 概述

## 1.1 简介

本提案旨在为 IndexSDK 补齐 A2/A3 NPU 平台上的 IVFPQ(L2) 算法能力。IVFPQ (Inverted File System with Product Quantization) 是推荐检索场景中常用的近似检索算法，能够在保证检索精度的同时，大幅降低内存占用和计算开销。通过本提案的实现，IndexSDK 将能够在 Ascend A2/A3 NPU 上完整支持 IVFPQ 算法的训练、索引构建、入库和检索全流程。

## 1.2 动机

### 背景

在推荐检索场景中，IVFPQ 是常用的近似检索算法。随着库规模增长，CPU 方案在时延上已无法满足业务需求，需要通过 NPU 进行加速。

### 痛点

- 大规模向量检索场景下，CPU 方案时延成为瓶颈

- A2/A3 NPU 平台缺少 IVFPQ 算法支持，无法充分发挥硬件加速能力

### 价值

- 补齐 A2/A3 平台的近似检索能力，完善 IndexSDK 的算法支持矩阵

- 充分利用 NPU 加速能力，大幅提升大规模向量检索性能

## 1.3 目标

### 目标

- 支持 `IVFPQ(L2)` 算法

- 支持完整的生命周期：训练、索引构建、入库、检索

- 精度与 CPU 基线对齐，误差控制在 `1e-5` 以内

- 在 A3 单卡场景完成性能评估，满足召回推荐参数配置

# 2. 用例分析

## 2.1 功能需求

### 核心功能

- **训练**: 支持聚类中心训练

- **索引构建**: 支持倒排索引和乘积量化码本构建

- **入库**: 支持向量添加到索引

- **检索**: 支持基于倒排索引的近似检索

### 参数规格

- **nlist**: 倒排列表数量，支持 1024、2048、4096、8192

- **dim**: 向量维度，128

- **m**: 乘积量化子空间数量，支持 2、4、8、16

- **batchSize**: 批量大小，支持 1、2、4、8、16、32、64

- **topk**: 返回数量，≤320 (召回 300)

- **nprobe**: 检索时访问的倒排列表数量，≤nlist (召回 32、64、128)

- **base**: 库规模，1000w、8000w

### 功能要求

- 支持完整的训练、构建、入库、检索流程

- 支持 L2 距离度量

- 支持 batch 模式下的并发查询

- 返回结果保证排序正确性

## 2.2 性能需求

### 精度验收标准

在以下场景中与 CPU 对比误差 `<= 1e-5`:

- `nlist`: 1024、2048、4096、8192

- `dim`: 128

- `m`: 2、4、8、16

- `batchSize`: 1、2、4、8、16、32、64

- `topk <= 320` (召回 300)

- `nprobe <= nlist` (召回 32、64、128)

- `base`: 1000w、8000w

### 性能验收标准

在 A3 单卡场景下:

- `nlist=1024`, `dim=128`, `m=4`

- `batchSize=1/2/4/8/16/32/64`

- `topk=300`

- `nprobe <= nlist` (召回 32/64/128)

- `base=1000w`

- 目标时延: `3.9ms`、`5.3ms`、`8.75ms`

## 2.3 DFX 要求

### 可靠性

- 算法计算结果正确性保证

- 异常输入能够检查并输出异常

### 可测试性

- 提供单元测试用例

- 提供性能基准测试用例

- 提供精度对比测试用例

### 兼容性

- 与现有 AscendIndexIVFPQ 接口保持兼容

- 不影响现有其他算法的功能

# 3. 方案设计

## 3.1 总体方案

### 设计思路

基于 IndexSDK 现有的 AscendIndexIVFPQ 架构，扩展支持 A2/A3 NPU 平台。主要工作包括：

- 训练模块：实现聚类中心训练，生成倒排列表和乘积量化码本

- 索引构建：构建倒排索引结构，存储向量编码

- 入库模块：支持向量添加到索引，进行聚类和量化编码

- 检索模块：实现基于倒排索引的近似检索，支持 nprobe 参数控制精度

### 技术架构

```text
用户应用层
    ↓
AscendIndexIVFPQ API
    ↓
AscendIndexIVFPQImpl
    ↓
├── 训练模块 (K-Means 聚类)
├── 索引构建模块 (倒排索引 + PQ 编码)
├── 入库模块 (向量编码)
└── 检索模块 (倒排查询 + 距离计算)
    ↓
ACL Runtime
    ↓
A2/A3 NPU
```

### 核心流程

#### 训练阶段

1. 输入训练数据集

2. 执行 K-Means 聚类，生成 nlist 个聚类中心

3. 对每个子空间执行 K-Means，生成乘积量化码本

4. 保存聚类中心和码本

#### 索引构建阶段

1. 加载聚类中心和码本

2. 对每个向量计算所属聚类

3. 对向量进行乘积量化编码

4. 将编码后的向量添加到对应倒排列表

#### 检索阶段

1. 计算查询向量所属聚类

2. 选择最近的 nprobe 个倒排列表

3. 计算查询向量与候选向量的距离

4. TopK 选择和结果返回

## 3.2 技术选型

### 方案

复用现有 AscendIndexIVFPQ 框架，针对 A2/A3 进行开发 AscendC 算子

### 选择理由

复用现有 AscendIndexIVFPQ 的算子框架，针对 A2/A3 进行适配优化，可以在保证性能的同时，降低开发和维护成本。

## 3.3 功能与性能设计

### 功能实现方案

#### 训练模块

- **聚类算法**: K-Means

- **实现方式**: 利用 NPU 的并行计算能力

- **数据流**: 训练数据 → K-Means 聚类 → 聚类中心 → 码本生成

#### 索引构建模块

- **倒排索引**: 基于聚类中心构建倒排列表

- **乘积量化**: 将向量分割为 m 个子向量，每个子向量量化为码本索引

- **存储格式**: 紧凑的编码格式，降低内存占用

#### 检索模块

- **倒排查询**: 根据查询向量找到最近的 nprobe 个倒排列表

- **距离计算**: 使用预计算的距离表（LUT）加速

- **TopK 选择**: 基于堆的 TopK 算法

### 性能优化策略

1. **内存优化**: 优化编码存储格式，提高内存访问效率

2. **计算优化**: 利用 NPU 的并行计算能力，优化批量处理

3. **流水线优化**: 重叠计算和数据传输，提高吞吐量

4. **预计算优化**: 预计算距离表，减少检索时计算量

### 影响范围

- 新增文件: A2/A3 平台的算子实现文件

- 修改文件: AscendIndexIVFPQ 相关配置和初始化逻辑

- 不影响现有其他平台的实现

## 3.4 安全隐私与DFX设计

### 安全隐私

- 不涉及用户敏感数据处理

- 算子实现遵循安全编码规范

### 兼容性

- API 接口保持向后兼容

- 支持与现有 AscendIndexIVFPQ 接口无缝对接

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

#### 配置结构体

**AscendIndexIVFPQConfig**：

```cpp
struct AscendIndexIVFPQConfig : public AscendIndexIVFConfig {
    // 构造函数
    AscendIndexIVFPQConfig();
    AscendIndexIVFPQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);
    AscendIndexIVFPQConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);
};

```

**参数说明**：

- `devices`：设备ID列表，支持多设备

- `resourceSize`：资源池大小，默认IVF_DEFAULT_MEM

#### 主要类接口

**AscendIndexIVFPQ**：

```cpp
class AscendIndexIVFPQ : public AscendIndexIVF {
public:
    // 构造函数
    AscendIndexIVFPQ(int dims, faiss::MetricType metric, int nlist, int msubs, int nbits,
                     AscendIndexIVFPQConfig config = AscendIndexIVFPQConfig());

    // 析构函数
    virtual ~AscendIndexIVFPQ();

    // 禁用拷贝构造和赋值
    AscendIndexIVFPQ(const AscendIndexIVFPQ&) = delete;
    AscendIndexIVFPQ& operator=(const AscendIndexIVFPQ&) = delete;

    // 核心方法
    void train(idx_t n, const float* x) override;
    void copyFrom(const faiss::IndexIVFPQ* index);
    void copyTo(faiss::IndexIVFPQ* index) const;
    void remove_ids(size_t n, const idx_t* ids);
    std::vector<idx_t> update(idx_t n, const float* x, const idx_t* ids);

protected:
    std::shared_ptr<AscendIndexIVFPQImpl> impl_;
};

```

#### 核心方法说明

**构造函数**：

- `dims`：向量维度

- `metric`：距离度量类型，支持METRIC_L2

- `nlist`：聚类中心数量

- `msubs`：乘积量化子空间数量

- `nbits`：每个子空间的编码位数

- `config`：配置参数

**训练方法**：

- `train(n, x)`：训练聚类中心和乘积量化码本

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

### 算子设计

#### 训练阶段算子

**1. 训练距离算子 (TrainDistOp)**：

- **功能**：计算训练向量与聚类中心的距离

- **输入**：

  - `queries`：训练向量（batch × dim）

  - `centroids`：聚类中心（nlist × dim）

  - `codesDouble`：距离累加器

- **输出**：

  - `distances`：距离矩阵

  - `vmdists`：最小距离

  - `opFlag`：操作标志

**2. 训练TopK算子 (TrainTopkOp)**：

- **功能**：选择每个训练向量最近的聚类中心

- **输入**：

  - `dists`：距离矩阵

  - `vmdists`：最小距离

  - `sizes`：聚类大小

  - `flags`：操作标志

  - `attrs`：属性信息

- **输出**：

  - `outdists`：TopK距离

  - `outlabel`：TopK标签

#### 检索阶段算子

**3. L1距离算子 (L1DistOp)**：

- **功能**：计算查询向量与所有聚类中心的距离

- **输入**：

  - `batch`：批大小

  - `queries`：查询向量

  - `centroidsDev`：聚类中心

- **输出**：

  - `dists`：距离矩阵

  - `vmdists`：最小距离

  - `opFlag`：操作标志

**4. L1 TopK算子 (L1TopkOp)**：

- **功能**：选择距离最近的nprobe个聚类

- **输入**：

  - `dists`：距离矩阵

  - `vmdists`：最小距离

  - `sizes`：聚类大小

  - `flags`：操作标志

  - `attrs`：属性信息

- **输出**：

  - `outdists`：TopK距离

  - `outlabel`：TopK标签

**5. L2距离算子 (L2DistOp)**：

- **功能**：计算查询向量与码本的子空间距离

- **输入**：

  - `batch`：批大小

  - `queries`：查询向量

  - `codeBook`：乘积量化码本

- **输出**：

  - `dists`：子空间距离矩阵

**6. L3距离算子 (L3DistOp)**：

- **功能**：在选定聚类中计算查询向量与库向量的距离

- **输入**：

  - `batch`：批大小

  - `queryPQ`：查询向量PQ编码

  - `codeBase`：库向量PQ编码

  - `offset`：编码偏移

  - `baseSize`：基础大小

  - `topk`：TopK参数

  - `labelBase`：标签基础

  - `labelOffset`：标签偏移

- **输出**：

  - `dists`：距离结果

  - `topkIndex`：TopK索引

  - `topkValue`：TopK值

  - `topkIndexFinal`：最终TopK索引

  - `topkValueFinal`：最终TopK值

  - `opFlag`：操作标志

**7. L3 TopK算子 (L3TopkOp)**：

- **功能**：对距离结果进行TopK排序

- **输入**：

  - `topkIndex`：TopK索引

  - `topkValue`：TopK值

  - `flags`：操作标志

  - `attrs`：属性信息

- **输出**：

  - `outdists`：TopK距离

  - `outlabel`：TopK标签

#### 算子实现要点

**1. 数据格式**：

- 使用ND格式存储数据，提高内存访问效率

- 数据按burst长度对齐（A2/A3: 64）

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

### 3.5.3 编程手册设计

需要在现有《IndexSDK 用户指南》中新增以下章节：

#### A2/A3 平台 IVFPQ 算法使用指南

1. **环境准备**：

   - 安装 CANN 工具链

   - 配置 NPU 驱动和固件

   - 设置环境变量

2. **使用示例**：

   ```cpp
   // 创建 IVFPQ 索引
   int dim = 128;
   int nlist = 1024;
   int m = 4;
   int nbits = 8;
   faiss::MetricType metric = faiss::METRIC_L2;

   faiss::ascend::AscendIndexIVFPQConfig config({0});  // 使用设备0
   faiss::ascend::AscendIndexIVFPQ index(dim, metric, nlist, m, nbits, config);

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

3. **参数调优建议**：

   - **nlist选择**：建议为sqrt(ntotal)到ntotal/1000之间

   - **m选择**：建议为dim/32到dim/8之间，需要能被dim整除

   - **nprobe选择**：建议为nlist的1/16到1/32之间

4. **常见问题与解决方案**：

   - **精度不达预期**：

     - 增加nprobe值

     - 增加m值

     - 检查训练数据是否充分

   - **性能不达预期**：

     - 检查batch size是否合理

     - 检查nprobe是否过大

     - 检查NPU利用率

   - **内存占用过大**：

     - 检查nlist是否过大

     - 检查m值是否合理

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

- 不影响现有其他算法的功能

### 对用户的影响

- 用户需要升级到支持 A2/A3 的版本

- 需要重新训练索引和生成算子模型文件

## 4.3 实现成本

### 开发成本

- 预计开发工作量: 6 人周

- 主要工作: 算子适配、性能优化、测试验证

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

- AscendIndexIVFPQ 已在 IndexSDK 中实现，本提案复用现有架构，扩展支持 A2/A3

# 6. 未解决问题

无

---

## 附录

### 参考资料

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)

- [IndexSDK 用户指南](../../zh/user_guide.md)

- [Ascend NPU 开发文档](https://www.hiascend.com/document)

### 术语表

- **IVFPQ**: Inverted File System with Product Quantization，倒排文件系统与乘积量化

- **nlist**: 倒排列表数量，聚类中心数量

- **m**: 乘积量化子空间数量

- **nbits**: 每个子空间的编码位数

- **nprobe**: 检索时访问的倒排列表数量

- **A2/A3**: Ascend NPU 型号

### 文档更新计划

- RFC 评审通过后，更新《IndexSDK 用户指南》

- 更新《快速开始指南》，添加 A2/A3 平台 IVFPQ 使用说明
