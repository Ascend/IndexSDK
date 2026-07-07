# 支持Cagra-RabitQ功能技术方案设计(RFC)

**状态 (Status):** Draft
**作者 (Authors):** @chasefhb
**创建日期 (Created):** 2026-05-06
**更新日期 (Updated):** 2026-05-08
**相关 Issue/PR:** [#42](https://gitcode.com/Ascend/IndexSDK/issues/42)

---

# 1. 概述

## 1.1 简介

本需求旨在对 Index SDK 中已有的 **AscendIndexCagra** 类进行功能增强，增加 **RabitQ（Random Binary Quantization）** 量化编码能力，形成 **Cagra-RabitQ** 混合索引。Cagra 是一种基于图的高性能近似最近邻（ANN）搜索算法；RaBitQ 是一种高压缩比、高理论保证的向量量化算法，通过随机正交变换和二进制编码实现极限压缩。两者结合，可在大规模向量检索场景中实现低内存占用、高精度、高吞吐的索引服务。

本需求将扩展 `AscendIndexCagra` 的生命周期方法，支持从训练、索引构建、入库到在线检索的完整流程，并在 A5 卡上完成性能评估，确保距离计算误差 ≤ 1e-5（与 CPU 浮点基线对齐），在典型推荐场景（100w 向量、128维 FP32、Top200）下综合性能达到 1.2 倍对照 GPU 基线（基于 NVIDIA 5090/4090/A100 换算）。

## 1.2 动机

当前 `AscendIndexCagra` 仅支持原始浮点向量存储，存在以下痛点：

- 内存开销大：大规模向量若使用原始浮点存储，单卡内存难以支撑亿级数据；即使采用 Cagra 图结构，邻接表 + 原始向量仍可能超出 A5 等设备内存。
- 量化方案精度不足：现有 IVF-PQ 等方案在超低比特下精度损失明显，无法满足推荐、搜索业务对召回率（>95%）和高精度距离计算（误差 < 1e-5）的严苛要求。
- 性能瓶颈：相比业界领先的 GPU 方案（如 NVIDIA CAGRA on RTX 4090），当前 SDK 在吞吐和延迟上存在差距。

若不实现此功能，用户将被迫采用更高内存成本的方案（如 HNSW+FP32）或降级使用精度较低的量化索引，导致业务成本上升或体验下降。

## 1.3 目标

目标：

- 在现有 `AscendIndexCagra` 类中增加 RabitQ 量化路径，支持将原始向量压缩为二进制码，替换原有的浮点向量存储。
- 扩展 `IndexCagraBuildConfig` 结构体，新增 RabitQ 相关配置参数（随机正交矩阵开关、种子、子向量数等）。
- 新增 `Train` 方法用于训练 RabitQ 码本（如果用户提供原始向量进行构建，也可自动训练）。
- 保持原有接口签名不变，`BuildGraph`、`Search` 等方法在量化模式下行为透明切换（内部根据配置决定是否使用量化）。
- 精度对齐：与 CPU 浮点暴力检索的距离计算误差 ≤ 1e-5；在 Top200 下召回率 ≥ 99%。
- 性能对标：在 A5 单卡、100w 向量（128维 FP32）、Top200 场景中，QPS / 延迟综合性能 ≥ 1.2 × 对照 GPU 基线（取 NVIDIA 5090/4090/A100 上最优 Cagra 实现的实测值）。
- 原有非量化模式必须完全兼容，不破坏任何现有功能。

## 平台支持

- A5平台：支持（优先）

# 2. 用例分析

## 核心功能

- **训练**: 支持RabitQ码本训练
- **索引构建**: 支持cagra和快速量化码本构建
- **入库**: 支持向量添加到索引
- **检索**: 支持基于Cagra-RabitQ索引的检索

# 3. 方案设计

## 3.1 总体方案

本方案在 **AscendIndexCagra** 类的内部实现上进行扩展，不改变对外公开的接口签名。具体做法：

- 在 `IndexCagraBuildConfig` 中增加一个枚举或布尔标志，用于选择索引模式：`kFloat`（原有浮点模式）或 `kRabitQ`（新增量化模式）。
- 新增私有成员 `RabitQQuantizer`，负责码本训练、向量编码、距离计算。
- 修改 `BuildGraph` 的实现：当启用 RabitQ 模式时，构建图仍可使用原始浮点向量（或临时解压向量），但图构建完成后将原始向量丢弃，仅保存压缩码。
- 修改 `Search` 的实现：根据模式调用不同的距离计算函数（原有浮点距离 或 RabitQ 近似距离）。
- 新增 `Train` 方法，用于独立训练 RabitQ 码本（也可在 `BuildGraph` 时自动触发）。

**架构图（文字描述）**：

```cpp
用户程序
│
▼
AscendIndexCagra (扩展)
├─ Init(dim, graphDegree, ...) // 初始化，增加RabitQ模式参数
├─ Train(vectors, num) // 新增：训练码本
├─ BuildGraph(...) // 增强：支持RabitQ压缩
├─ Search(...) // 增强：自动选择距离计算
└─ Serialize/Deserialize // 增强：保存/加载码本和压缩码
```

**核心逻辑**：

1. **训练**：调用 `Train(vectors, num)` 对训练子集运行 RabitQ 算法，生成随机正交矩阵、子向量划分、中心点等码本数据。
2. **构建**：调用 `BuildGraph` 时，若配置为 RabitQ 模式，则：
   - 使用原始浮点向量运行现有 Cagra 图构建算法（kNN 图）。
   - 对所有向量进行 RabitQ 编码得到压缩二进制码，并存储到设备内存。
   - 释放原始向量内存。
3. **入库/序列化**：将图结构（CSR）、压缩码、码本序列化到磁盘；加载时直接映射到 A5 设备内存。
4. **检索**：`Search` 接口接收查询向量，如果是 RabitQ 模式，则先对查询向量进行随机正交变换和归一化，然后调用昇腾核函数批量计算与邻居节点压缩码的距离，通过 Beam Search 迭代得到 TopK。

## 3.2 技术选型

| 备选方案                | 优势                                         | 劣势                                              | 是否采纳             |
| :---------------------- | :------------------------------------------- | :------------------------------------------------ | :------------------- |
| HNSW + PQ               | 社区成熟，内存可控                           | 随机访存对 NPU 不友好，PQ 精度不足（误差 >1e-3）  | 否                   |
| IVF + 原始 FP32         | 精度高                                       | 内存爆炸（100w-512MB 勉强可接受，但亿级无法部署） | 否                   |
| 已有 AscendIndexCagra (无量化) | 检索速度快，实现稳定                   | 内存仍较大（图+原始向量）                          | 作为基础，需加量化   |
| **扩展现有 AscendIndexCagra + RabitQ (本方案)** | 复用图构建、检索框架，仅替换向量存储与距离计算；内存低、精度高、开发成本低（2人月） | 需要训练码本，增加构建复杂度 | **是**               |

### 选择理由

1. **完全复用现有架构**：`AscendIndexCagra` 已经实现了高效的图构建和检索调度，新增 RabitQ 只需要替换向量存储和距离计算模块，风险低，周期短。
2. **随机正交矩阵优势**：RabitQ 通过随机正交变换使向量各维度分布均匀，减少量化误差，在同等压缩率下获得更高召回率。
3. **昇腾友好**：二进制码的距离计算适合昇腾 AI Core 的位运算加速。

### RabitQ 核心原理

#### 1. 随机正交矩阵

- **作用**：对原始向量应用随机正交变换，使得变换后向量的各个维度独立且分布更均匀，降低量化误差。
- **生成方法**：使用 Givens 旋转生成随机正交矩阵（可通过种子保证可复现性）。
- **配置参数**（新增到 `IndexCagraBuildConfig`）：
  - `use_random_orthogonal_matrix`：是否启用随机正交变换（默认 true）。
  - `orthogonal_matrix_seed`：随机种子（默认 42）。

#### 2. RabitQ量化编码

- **编码方式**：将向量编码为二进制表示（每个维度 1 bit）。
- **编码维度**：压缩后大小为 `dim / 8` 字节（当 dim 为 8 的倍数；非对齐时补齐）。
- **编码过程**：
  1. 对向量应用随机正交变换。
  2. 计算变换后向量的 L2 范数，并存储（用于距离还原）。
  3. 将向量归一化为单位向量。
  4. 对归一化向量的每个维度进行二值量化（正为 1，负为 0），得到二进制码。
- **距离计算**：
  - 查询向量同样经过变换、归一化，得到查询二进制码。

## 3.3 功能与性能设计

### 3.3.1 精度对齐（误差 ≤ 1e-5）

- RabitQ 编码配置：
  - 子向量数 `n_subvectors = 16`（128/16=8 维每子向量），每子向量 8bit 量化。
  - 采用残差迭代编码：先减中心点，再对残差进行位量化，步长 Δ = (max - min)/(2^bits-1)，确保舍入误差 < 1e-6。
- 验证方法：与 CPU 浮点暴力检索的距离计算误差 ≤ 1e-5；在 Top200 下召回率 ≥ 99%。
- 召回率补偿：若因量化导致召回下降，可通过增大 `beam_width`（默认 64 → 128）提升至 ≥99%。

### 3.3.2 性能对标 1.2x GPU

对标流程：

1. 由性能实验室在同等数据集（100w 128d，SIFT1M）下，分别测试：
   - 对照 GPU（5090/4090/A100）运行 NVIDIA RAFT CAGRA（浮点版本），记录最优 QPS（满足 Recall@200 ≥ 99%）。
   - A5 运行本方案（`AscendIndexCagra` + RabitQ 模式），调优参数（`graphDegree=32`，`beam_width=64`，开启随机正交矩阵）达到相同召回率，记录 QPS。
2. 预期目标：`QPS_A5 ≥ 1.2 × QPS_gpu`。

优化手段（基于现有 `AscendIndexCagra` 内核修改）：

- **算子融合**：将 RabitQ 距离计算（随机正交变换 + 归一化 + 汉明距离计算）融合为单个昇腾核函数，减少核函数启动和数据搬运开销。
- **缓存优化**：利用 A5 的 L2/SRAM，将热点邻居的压缩码预先加载到缓存中。
- **并行调度**：复用现有 Cagra 的查询级并行机制（每个查询独立使用一个 AI Core），并支持 batch 查询时流水线化。

## 3.4 安全隐私与 DFX 设计

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

- 开发环境：C++17，CANN 9.0+。
- 开发约束：所有公共接口线程安全；不能抛出异常跨模块；设备内存必须通过 SDK 分配器申请。
- 可验收设计：
  - 功能：给定 100w 向量，构建索引入库后，随机查询 100 次，距离误差 < 1e-5，召回率 ≥ 99%。
  - 性能：参考 3.3.2 对标流程，达到 1.2x 目标。

### 3.5.2 接口定义与设计

现有 `AscendIndexCagra` 类的主要公共接口：

```cpp
class AscendIndexCagra {
public:
    AscendIndexCagra();
    virtual ~AscendIndexCagra();

    APP_ERROR Init(int dim, int graphDegree, int hashBitlen,
                   std::vector<int>& deviceList, int64_t ascendResourceSize);

    APP_ERROR add(int n, const float* x);

    APP_ERROR BuildGraph(int64_t n, const float* data, const std::string& graphFilePath,
                         const IndexCagraBuildConfig& buildConfig);

    APP_ERROR Search(int64_t n, const float* x, const uint32_t* hash,
                     int k, float* distances, uint32_t* labels);
};
```

本需求将进行以下扩展（保持二进制兼容性）：

#### 3.5.2.1 扩展 IndexCagraBuildConfig

增加 RabitQ 相关配置字段：
struct IndexCagraBuildConfig {
    // ... 原有字段（如图构建参数等）...

    // 新增字段
    bool enable_rabitq = false;                // 是否启用RabitQ量化模式
    int rabitq_n_subvectors = 16;              // 子向量个数（仅当enable_rabitq=true时有效）
    int rabitq_bits_per_code = 8;              // 每子向量编码位数
    bool use_random_orthogonal_matrix = true;  // 是否使用随机正交变换
    uint32_t orthogonal_matrix_seed = 42;      // 随机种子
};

若 enable_rabitq == false，则行为与原有完全一致（浮点模式）。

若 enable_rabitq == true，则 BuildGraph 内部会：

自动训练 RabitQ 码本（如果尚未训练）。

使用原始向量构建图（复用现有逻辑）。

将所有向量编码为压缩码，释放原始向量。

#### 3.5.2.2 修改 Search 行为

Search 接口签名保持不变，但内部实现会根据 enable_rabitq 标志选择不同的距离计算函数：

浮点模式：调用原有浮点距离算子。

RabitQ 模式：调用新增的昇腾核函数（对查询向量进行随机正交变换 + 归一化 + 汉明距离计算）。

注意：现有 Search 参数中包含 const uint32_t* hash，这是用于其他用途（如 IVF 的哈希）。本需求不改变该参数含义，在 RabitQ 模式下可能忽略该参数。未来若需要统一，可在后续版本调整。

#### 3.5.2.3 序列化扩展

原有 Serialize / Deserialize 方法（未在此处列出，但实际存在）需要增强：

写入/读取 RabitQ 码本（随机正交矩阵、子向量划分信息、中心点等）。

写入/读取压缩码数组。

增加版本号，兼容旧索引格式。

### 3.5.3 编程手册设计

在《Index SDK 编程手册》中更新“Cagra 索引”章节，增加“RabitQ 量化模式”小节，包含：

如何开启 RabitQ 模式（通过配置 IndexCagraBuildConfig::enable_rabitq=true）。

RabitQ 参数调优建议（子向量数、编码位数、随机正交矩阵开关对精度和内存的影响）。

完整代码示例：从原始向量构建 RabitQ 索引并进行检索。

性能基准报告（包含 A5 vs 4090 对比数据）。

常见问题解决（召回不足、延迟超标等）。

# 4. 缺点和风险

## 4.1 潜在风险

### Breaking Change

- 无 Breaking Change，完全向后兼容

### 性能风险

- 不同 NPU 平台性能可能存在差异，需要充分测试
- 大规模向量库可能存在内存压力

## 4.2 负面影响

### 对现有功能的影响

- 不影响现有其他算法的实现
- 不影响现有其他平台的功能

### 对用户的影响

- 用户需要升级到支持 Cagra-RabitQ 的版本
- 需要重新训练索引

## 4.3 实现成本

### 开发成本

- 预计开发工作量: 约 2 人月
- 主要工作: 算法实现、性能优化、测试验证

### 维护成本

- 长期维护成本中等
- 需要跟进 CANN 版本更新

# 5. 现有技术

 - AscendIndexCagra 已在 IndexSDK 中实现，提供 Cagra 基础框架
 - 本提案复用现有 Cagra 框架，结合 RabitQ 量化技术

# 6. 未解决问题

无

附录
参考资料

  - [Faiss 官方文档](https://github.com/facebookresearch/faiss)
  - [IndexSDK 用户指南](../../zh/user_guide.md)
  - [Ascend NPU 开发文档](https://www.hiascend.com/document)
  - [RabitQ 论文](https://arxiv.org/abs/2405.12456)

术语表
RabitQ: Random Binary Quantization，随机二进制量化，一种基于随机正交变换的高维向量量化算法。

Cagra：CUDA-Accelerated Graph-based Nearest Neighbor Search，基于图的 ANN 算法，已移植到昇腾 NPU。

文档更新计划
RFC 评审通过后，更新《IndexSDK 用户指南》，添加 Cagra-RabitQ 使用说明。

更新《快速开始指南》，提供 RabitQ 模式示例代码。
