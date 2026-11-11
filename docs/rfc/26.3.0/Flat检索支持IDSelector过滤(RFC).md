# [RFC] Flat 检索支持 faiss::IDSelectorBitmap 过滤

**状态 (Status):** Approved
**作者 (Authors):** @lan_xin
**创建日期 (Created):** 2026-10-26
**更新日期 (Updated):** 2026-11-11

## 1. 概述

### 1.1 简介

本提案旨在为 Ascend Flat 向量检索提供 `faiss::IDSelectorBitmap` 接入能力，使调用方能够通过外部 ID 位图指定允许参与检索的候选集合，并获得该集合内的 TopK 结果。

方案在主机侧将 Bitmap 转换为底库位置对应的 mask，复用已有设备端过滤检索能力。同时，通过 Bitmap 转换快路径、查询缓冲简化和多设备副本并发，提高端到端检索性能。

主要工作包括：

- 在 `AscendIndexFlat` 中新增 `search_with_selector` 接口，支持 FP32、FP16 查询输入。
- 支持连续 ID、自定义 ID，以及删除向量后的 ID 到底库位置映射。
- 连续 ID 场景直接复制 Bitmap；自定义 ID 场景直接读取位图并重映射。
- 提供多设备完整副本模式，将不同查询调用分发至不同设备。
- 补充正确性测试和单设备、多设备性能基准。

### 1.2 动机

#### 背景

业务通常需要先根据属性、权限或业务条件生成候选 ID 集合，再执行向量检索。`faiss::IDSelectorBitmap` 使用紧凑位图表达允许参与检索的 ID，适合候选集合可以通过位图描述的场景。

现有 Ascend Flat 已支持 `search_with_masks`，但其 mask 按底库位置排列，与业务侧外部 ID 不一定一致。

#### 痛点

- **接入成本较高**：调用方需要自行维护外部 ID 与底库位置的对应关系。
- **数据变更容易影响过滤正确性**：删除向量后，底库位置可能发生变化，直接使用外部 ID 位图会导致过滤位置错误。
- **通用转换存在额外开销**：逐个调用 `IDSelector::is_member()` 并重建 mask，增加主机侧查询耗时。
- **多设备吞吐需要扩展**：对于完整底库可放入单设备的场景，需要利用多个副本并行处理不同请求。

#### 价值

- 调用方直接使用 Faiss Bitmap 表达过滤条件，降低接入成本。
- 在索引内部完成 ID 映射，保证数据变更后的过滤语义。
- 复用现有过滤算子，减少设备端开发和维护工作。
- 优化连续 ID 常见场景，并通过多设备副本提升总体吞吐。

### 1.3 目标

#### 目标

- 支持 `faiss::IDSelectorBitmap` 白名单过滤：bit 为 1 的 ID 允许参与检索。
- 同一次调用中的全部 query 共用一个 Selector。
- 支持 FP32、FP16 查询输入，以及 IP、L2 距离类型。
- 支持连续 ID、非连续自定义 ID、删除后检索。
- 支持单设备和多设备副本模式。
- 在指定性能基准下，16 die 端到端总吞吐达到 100,000 QPS。
- 保持已有普通检索、mask 检索及默认多设备分片行为兼容。

#### 非目标

- 本期不扩展多设备分片模式下的 Selector 查询。
- Range、Batch、Array 等 Selector 不作为本需求的核心验收范围，已有通用路径保持兼容。
- 本期不修改底层算子实现 batch 共享单份设备 mask。
- 本期不增加基于 Selector 对象地址的过滤结果缓存。

## 2. 用例分析

### 2.1 功能需求

#### 核心功能

| 功能 | 说明 |
|------|------|
| Bitmap 过滤检索 | 按外部 ID 位图限定候选集合，返回集合内 TopK |
| 批内共享条件 | 同一 batch 的所有 query 使用相同 Bitmap |
| 连续 ID 快路径 | ID 等于底库位置时，直接复制 Bitmap |
| 自定义 ID 重映射 | 根据当前 ID 映射生成底库位置 mask |
| 数据变更适配 | 添加、删除、重置后重新添加、导入后保持过滤正确 |
| 多设备副本检索 | 各设备保存完整底库，每次调用选择一个设备执行 |
| 结果映射 | 将设备返回的底库位置转换为外部 ID |

#### 参数规格

| 参数 | 取值或约束 | 备注 |
|------|------------|------|
| 向量维度 `d` | 沿用现有 Flat 支持范围 | 性能基准使用 4096 维 |
| 底库规模 `ntotal` | 沿用现有 Flat 限制 | 性能基准使用 100,000 条 |
| 查询数量 `n` | `0 < n < 1e9` | 性能基准使用 32、48 |
| 返回数量 `k` | `1 <= k <= 4096` | 性能基准使用 30 |
| 查询数据类型 | FP32、FP16 | FP32 在主机侧转换为 FP16 |
| 距离度量 | IP、L2 | 复用现有 Flat 实现 |
| Bitmap 长度 | 字节数 | ID 超出位图范围时不参与检索 |
| Bitmap 位序 | 字节内低位优先 | ID `i` 对应 `bitmap[i / 8]` 的第 `i % 8` 位 |
| 过滤条件粒度 | 每次调用一份 | 整个 batch 共用 |
| 多设备模式 | 完整副本 | Selector 查询不支持多设备分片 |

#### 功能要求

- Bitmap 以外部 ID 为索引，不能默认将外部 ID 当作底库位置。
- 过滤条件在设备端 TopK 计算过程中生效。
- Bitmap 短于底库所需范围时，未覆盖的 ID 视为不选中。
- Bitmap 中超出底库范围的部分不影响结果。
- 候选不足 K 时，沿用现有过滤检索的填充规则，无效标签为 `-1`。
- 同一调用期间，Selector 及底层 Bitmap 必须保持有效，且不能被并发修改。
- 不要求调用方维护设备 mask 或同步底库位置变化。

### 2.2 性能需求

#### 精度验收标准

**精度测试方法：**

1. 使用当前 ID 映射和 `IDSelectorBitmap::is_member()`，手工构造参考位置 mask。
2. 使用同一查询分别调用 `search_with_masks` 和 `search_with_selector`。
3. 比较距离、外部 ID、无效结果填充及候选集合成员关系。
4. 多设备副本结果与单设备参考索引对比。
5. 使用小规模数据独立计算过滤集合内 TopK，补充验证过滤语义。

FP32 查询转 FP16 沿用现有数值路径。CPU 独立参考应考虑该转换带来的误差；存在同分候选时，不强制要求同分 ID 的排列顺序一致。

**验收指标：**

| 验收项 | 指标 |
|--------|------|
| 有效结果的成员关系 | 100% 满足 Bitmap 过滤条件 |
| 与参考 mask 路径的一致性 | 相同输入、确定性数据下距离和标签一致 |
| 多设备副本一致性 | 与单设备参考结果一致 |
| 数据变更后正确性 | 添加、删除、重置重建及导入场景全部通过 |
| 边界处理 | 短位图、空位图、尾部无效位及候选不足 K 正确处理 |

#### 性能验收标准

**性能测试配置：**

| 项目 | 配置 |
|------|------|
| 底库规模 | 200,000 条 |
| 向量维度 | 4096 |
| 距离度量 | IP |
| TopK | 30 |
| 查询 batch | 512、1024 |
| 允许通过比例 | 100%、50%、1% |
| 多设备规模 | 16 die |
| 数据布局 | 每个 die 保存完整底库 |

**性能测试方法：**

对比三条路径：

1. 普通 Flat 检索。
2. 预生成 mask 的过滤检索。
3. Bitmap Selector 端到端过滤检索。

每组配置预热后测量 3 次，每次不少于 2 秒，取 QPS 中位数，并记录平均调用耗时。多设备使用并发调用，统计所有线程处理的查询向量总数。

端到端计时包含：

- 查询数据转换。
- Bitmap 到位置 mask 的转换与批内复制。
- 查询和 mask 的设备传输。
- 设备检索。
- 距离转换和外部 ID 映射。

底库构建和 Selector 构造不计入查询耗时。正确性检查失败的配置不采纳性能数据。

**验收指标：**

| 验收项 | 指标 |
|--------|------|
| 16 die 端到端总吞吐 | 达到 100,000 QPS |
| 单 die 吞吐 | 记录各配置实测值；6,250 QPS 作为线性扩展参考 |
| Bitmap 转换优化 | 对比优化前后耗时及主机 CPU 占用 |
| 原有查询路径 | 普通检索、mask 检索执行回归并记录性能变化 |

最终报告应附硬件型号、CANN/驱动版本、编译选项、线程数及各配置结果。不得使用单 die 线性推算代替多 die 实测。

### 2.3 DFX 要求

#### 可靠性

- 校验查询指针、输出指针、Selector 指针及查询参数。
- 检查 ACL 数据传输和设备检索返回码，失败时返回明确异常。
- 开启既有多线程安全机制后，索引写操作与查询互斥，避免查询期间 ID 映射发生变化。
- 同一设备上的查询串行访问现有内部状态。
- 多设备副本删除后检查删除数量及 ID 映射一致性。

#### 可测试性

- 使用 Google Test 覆盖 Bitmap 快路径和重映射路径。
- 覆盖 FP32、FP16、IP、L2、数据变更和多设备并发。
- 无双设备或未启用多线程模式时，副本测试明确报告跳过原因。

#### 兼容性

- 保留现有普通查询和 mask 查询接口。
- 使用现有 `faiss::IDSelector`、`faiss::IDSelectorBitmap` 类型。
- 新增配置参数提供默认值，原有构造调用方式保持兼容。
- 不新增设备算子依赖。

## 3. 方案设计

### 3.1 总体方案

#### 设计思路

采用“主机侧适配、设备端复用”的方式：

1. 接口接收 `const faiss::IDSelector*`，本需求重点支持 Bitmap。
2. 主机侧根据当前 ID 映射，将外部 ID 位图转换为底库位置 mask。
3. ID 与位置一致时直接复制 Bitmap，避免逐 ID 扫描。
4. ID 与位置不一致时直接读位图并重映射，避免逐 ID 虚函数调用。
5. 复用现有 `searchFilter` 完成设备过滤检索。
6. 多设备使用完整副本，每次调用选择一个设备执行，省去跨设备 TopK 合并。

#### 技术架构

```text
用户应用
  │ 查询数据、IDSelectorBitmap
  ▼
接口层：AscendIndexFlat
  │ search_with_selector
  ▼
实现层：AscendIndexFlatImpl
  ├─ 参数校验、FP32 → FP16
  ├─ Bitmap → 底库位置 mask
  ├─ 副本设备轮询、并发控制
  └─ 距离转换、外部 ID 映射
  ▼
现有过滤检索：Index::searchFilter
  ▼
Flat 设备端过滤与 TopK
  ▼
ACL Runtime + Ascend NPU
```

#### 核心流程

```text
输入 query 和 Selector
  │
  ├─ 1. 校验参数并持有索引锁
  ├─ 2. FP32 查询转换为 FP16；FP16 输入直接使用
  ├─ 3. 生成位置 mask
  │     ├─ Bitmap + ID 等于位置：复制有效字节，补零并清理尾部位
  │     ├─ Bitmap + 自定义 ID：逐位置读取对应外部 ID 的 bit
  │     └─ 其他 Selector：沿用 is_member 通用路径
  ├─ 4. 将第一份 mask 复制给 batch 中其他 query
  ├─ 5. 单设备执行，或轮询选择一个完整副本
  ├─ 6. 上传 query，调用现有 searchFilter
  └─ 7. 转换距离并将底库位置映射为外部 ID
```

#### 实现演进

| 提交 | 最终交付内容 |
|------|--------------|
| `21e764d1` | FP32/FP16 Selector接口、位置mask转换、单设备功能与性能验证 |
| `220f7ca9` | 完整副本、轮询调度、并发保护及副本数据维护 |
| `3c0e08b3` | 性能示例增加Bitmap、Array（Array仅1%），未新增专用算子 |
| `6a12d1cc` | Bitmap直接复制/直接位读取、ID快路径状态维护及副本缓冲简化 |

Bitmap快路径的成员判断由逐ID调用变为有效字节复制；非连续ID仍扫描映射。内部仍构造batch份mask并下发设备，未实现共享单份设备mask或Selector缓存。

### 3.2 技术选型

| 项 | 选型 | 说明 |
|----|------|------|
| 过滤条件 | `faiss::IDSelectorBitmap` | 复用 Faiss 类型和位图语义 |
| 设备过滤输入 | 现有位置 mask | 不改变算子输入约定 |
| Bitmap 类型识别 | 每次调用识别一次 | 保留其他 Selector 的通用路径 |
| 连续 ID 状态 | 写操作期间维护 | 查询时不重新扫描 ID 映射 |
| 多设备部署 | 完整副本 | 面向请求并行和吞吐扩展 |
| 设备选择 | 原子计数轮询 | 实现简单，无需跨设备结果合并 |
| 并发控制 | 索引读写锁及每设备互斥锁 | 复用现有多线程管理能力 |
| 资源管理 | 现有 Tensor 和容器 | 临时资源随调用释放 |

### 3.3 功能与性能设计

#### 功能实现方案

##### AscendIndexFlat：接口层

新增 FP32、FP16 两个 `search_with_selector` 重载，转发至实现层。

新增 `AscendIndexFlatConfig::shard`：

- `true`：默认值，保留原有分片模式。
- `false`：完整副本模式，各设备保存相同底库。
- 多设备 Selector 查询要求 `shard=false`。
- 多设备副本要求启动前设置 `MX_INDEX_MULTITHREAD=1`，构造时即检查，即使当前调用方仅使用一个线程。
- 该开关为IndexSDK原有机制，本需求新增的是副本构造检查；它控制索引锁、线程安全内存栈及算子管理，首次读取后缓存。单设备Bitmap接入本身不强制开启，但并发调用须开启。
- 开关不自动创建查询线程，调用方通过固定工作线程并发提交查询。

##### AscendIndexFlatImpl：实现层

| 函数或状态 | 实现要点 |
|------------|----------|
| `search_with_selector` | FP32 参数校验、查询转换、mask 生成和检索 |
| `search_with_selector_fp16` | FP16 参数校验、mask 生成和检索 |
| `selectorToMask` | Bitmap 快路径、ID 重映射和批内 mask 复制 |
| `identityIds` | 表示外部 ID 是否与底库位置一一对应 |
| `updateIdentityIds` | 添加时增量检查，删除后重新检查 |
| `searchWithMaskProcess` | 复用过滤检索，并适配副本结果返回 |
| `nextDeviceIndex` | 原子轮询选择副本 |
| `getLogicalRemoveCount` | 将物理副本删除数转换为逻辑删除数 |

##### Bitmap 转换

连续 ID 条件为：

```cpp
ids[p] == static_cast<idx_t>(p)
```

满足条件时，复制：

```text
min(ceil(ntotal / 8), bitmap字节数)
```

其余位置补零，并清除最后一个字节中超出底库范围的 bit。

不满足条件时，根据 `ids[p]` 读取 Bitmap 对应位，设置位置 `p` 的 mask。Bitmap 范围外的 ID 不选中。

不缓存 Selector 对应的转换结果，调用方在两次调用之间修改 Bitmap 内容时，下一次查询能够读取最新内容。

##### 副本数据维护

- **添加**：向所有设备写入相同向量及 ID，逻辑 `ntotal` 仅增加一次。
- **删除**：各设备执行相同删除条件，更新位置映射并检查副本一致性。
- **导出**：仅取一个副本，避免重复导出数据。
- **重置**：清空所有副本；重新添加时重新建立快路径状态。

#### 性能设计

| 优化项 | 方式 | 减少的开销 |
|--------|------|------------|
| 连续 ID Bitmap | 直接复制有效字节 | 逐 ID 判断和逐 bit 重建 |
| 自定义 ID Bitmap | 直接读取对应 bit | 逐 ID 虚函数调用 |
| 映射状态判断 | 写操作维护，查询读取 | 每次查询验证整个 ID 映射 |
| 副本结果缓冲 | 使用一维 FP16 距离和本地标签数组 | 嵌套容器及临时 FP32 数组 |
| 结果转换 | 直接写调用方输出 | 中间 FP32 距离复制 |
| 多设备查询 | 每次调用只使用一个完整副本 | 跨设备 TopK 合并 |

当前设备过滤接口仍要求每条 query 一份 mask，内存规模为：

```text
batch × ceil(ntotal / 8) 字节
```

10 万底库、batch=32/48 时，分别为 400,000/600,000 字节。本期保留该布局，后续根据实测占比评估共享设备 mask。

#### 影响范围

以下路径以仓库根目录为基准。

| 文件/模块 | 开发类型 | 说明 |
|-----------|----------|------|
| `feature_retrieval/src/ascendfaiss/ascend/AscendIndexFlat.h/.cpp` | 修改 | 查询接口及副本配置 |
| `feature_retrieval/src/ascendfaiss/ascend/impl/AscendIndexFlatImpl.h/.cpp` | 修改 | Bitmap 转换、状态维护、副本查询和数据管理 |
| `feature_retrieval/src/ascendfaiss/ascend/impl/AscendIndexImpl.h/.cpp` | 修改 | 逻辑删除数量扩展点 |
| `feature_retrieval/src/ascendfaiss/ut/cases/flat/TestAscendFlat.cpp` | 修改 | 功能及回归测试 |
| `examples/TestAscendIndexFlat.cpp` | 修改 | 单设备、多设备性能基准 |
| `examples/CMakeLists.txt` | 修改 | 示例线程库依赖 |
| `examples/README.md` | 修改 | 使用方式及测试说明 |
| 现有 Flat 过滤算子 | 复用 | 本期不修改算子 |

### 3.4 安全隐私与 DFX 设计

#### 安全隐私

- 查询向量、Bitmap 和结果沿用现有本地检索数据流。
- 日志不输出查询向量和完整 Bitmap 内容。
- Bitmap 读取检查 ID 范围，短位图不发生越界访问。

#### 兼容性

- 新接口为能力扩展，不改变现有查询接口签名；配置结构新增成员后，应用与库需配套重新编译，不承诺旧二进制ABI兼容。
- `shard` 默认保持分片模式。
- 本方案复用 Faiss Selector 类型，但新增的是 Ascend 扩展查询接口，不宣称等同于 CPU Faiss 的全部查询参数接口。
- 未新增 Python 绑定，Python 接入不纳入本期交付。

#### 可维护性

- Bitmap 专用分支集中在 mask 转换函数。
- 其他 Selector 沿用原有通用逻辑。
- 复用设备端过滤及资源管理，不新增独立算子链路。
- ID 快路径状态只在索引数据变更时维护。

#### 可测试性

重点覆盖：

- 连续、自定义及乱序 ID。
- 分批添加、删除中间元素、删除末尾元素。
- 重置后重新添加、CPU Flat/IDMap 导入。
- 空、全零、全一、短位图和长位图。
- 底库数量不是 8 的倍数。
- 候选不足 K、FP32/FP16 和 IP/L2。
- 多设备轮询、并发查询和副本一致性。

#### 可靠性

- 启用多线程安全开关后，副本查询持读锁、分片路径持写锁；写操作与查询互斥，保证映射和底库稳定。
- 保留每设备互斥锁，保护底层过滤检索使用的成员状态。
- 设备传输或检索失败时抛出带错误信息的异常。
- 多设备分片下调用 Selector 查询时明确报错。

### 3.5 编程与调用设计

#### 3.5.1 编程模型基本设计

##### 开发环境

| 项 | 要求 |
|----|------|
| 硬件平台 | 当前 Flat 过滤算子支持的 Ascend NPU |
| 编程语言 | C++17 |
| 运行时 | ACL Runtime |
| 算子依赖 | 现有 Flat 距离、过滤及 TopK 算子 |
| 构建系统 | 项目现有 CMake 构建流程 |
| 测试框架 | Google Test |

##### 开发约束

- 安装配套驱动、CANN 和目标维度的 Flat 算子。
- 多设备副本的每个设备均需容纳完整底库及查询资源。
- 多设备副本在进程启动前启用 `MX_INDEX_MULTITHREAD=1`。
- 同一 Selector 可供并发查询只读使用，调用期间不得修改底层 Bitmap。

##### 可验收设计

| 验收类型 | 环境 | 标准 |
|----------|------|------|
| 功能验收 | 单设备 | Bitmap、ID 映射和数据变更测试全部通过 |
| 精度验收 | 单设备 | 与参考 mask 及独立过滤 TopK 基线对齐 |
| 并发验收 | 至少双设备 | 副本轮询、并发查询及数据一致性测试通过 |
| 性能验收 | 16 die | 按规定口径验证 100,000 QPS 目标 |
| 回归验收 | 原有测试环境 | 普通检索、mask 检索及分片功能正常 |

#### 3.5.2 接口定义与设计

##### 3.5.2.1 search_with_selector

**接口描述：** 使用指定 Selector 限定 Flat 检索候选集合，整个 batch 共用过滤条件。

**接口原型：**

```cpp
void search_with_selector(
    idx_t n, const float* x, idx_t k,
    float* distances, idx_t* labels,
    const faiss::IDSelector* sel) const;

void search_with_selector(
    idx_t n, const uint16_t* x, idx_t k,
    float* distances, idx_t* labels,
    const faiss::IDSelector* sel) const;
```

**参数说明：**

| 参数 | 输入/输出 | 类型 | 描述 |
|------|-----------|------|------|
| `n` | 输入 | `idx_t` | 查询数量 |
| `x` | 输入 | `const float*` / `const uint16_t*` | 主机查询数据，形状 `[n, d]` |
| `k` | 输入 | `idx_t` | 每条查询返回的结果数量 |
| `distances` | 输出 | `float*` | 距离数组，容量至少 `n × k` |
| `labels` | 输出 | `idx_t*` | 外部 ID 数组，容量至少 `n × k` |
| `sel` | 输入 | `const faiss::IDSelector*` | 非空，本需求使用 Bitmap |

**异常处理：**

- 查询或输出指针无效、参数不满足现有 Flat 约束时抛出异常。
- `sel` 为空时抛出异常。
- 多设备分片模式下调用时抛出异常。
- ACL 调用或设备检索失败时抛出带错误码的异常。

**调用参考代码：**

```cpp
#include <cstdint>
#include <vector>

#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/impl/IDSelector.h>

void searchFiltered(
    int d, faiss::idx_t nb, const float* xb,
    faiss::idx_t nq, const float* xq, faiss::idx_t k,
    const std::vector<faiss::idx_t>& allowedIds)
{
    faiss::ascend::AscendIndexFlatConfig config({0});
    faiss::ascend::AscendIndexFlat index(
        d, faiss::METRIC_INNER_PRODUCT, config);

    // 默认生成连续 ID：[0, nb)。
    index.add(nb, xb);

    std::vector<uint8_t> bitmap((nb + 7) / 8, 0);
    for (auto id : allowedIds) {
        if (id >= 0 && id < nb) {
            bitmap[id / 8] |= uint8_t(1u << (id % 8));
        }
    }

    faiss::IDSelectorBitmap selector(bitmap.size(), bitmap.data());
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    index.search_with_selector(
        nq, xq, k, distances.data(), labels.data(), &selector);
}
```

使用自定义 ID 时，Bitmap 按外部 ID 编码；位图长度应覆盖需要选中的最大非负 ID，而不一定等于 `ceil(ntotal / 8)`。示例中的`nb`字节长度计算仅适用于连续ID。Selector及Bitmap必须在调用期间有效且只读，FP16重载接收FP16位模式，不能强制转换FP32指针。`x`、`distances`、`labels`均位于主机，容量由调用方保证；本设计不额外承诺空底库查询行为。

##### 3.5.2.2 多设备副本配置

**接口描述：** 在多个设备保存完整底库，并将每次查询调用分配到一个副本。

**配置原型：**

```cpp
AscendIndexFlatConfig(
    std::vector<int> devices,
    int64_t resourceSize = FLAT_DEFAULT_MEM,
    bool isShard = true);
```

| 参数 | 说明 |
|------|------|
| `devices` | 设备 ID 列表 |
| `resourceSize` | 沿用现有资源内存配置 |
| `isShard` | `false` 启用完整副本；默认 `true` 为分片 |

**调用参考代码：**

```cpp
std::vector<int> devices = {0, 1};

faiss::ascend::AscendIndexFlatConfig config(
    devices, faiss::ascend::FLAT_DEFAULT_MEM, false);

faiss::ascend::AscendIndexFlat index(
    d, faiss::METRIC_INNER_PRODUCT, config);

index.add(nb, xb);
```

启动前设置：

```bash
export MX_INDEX_MULTITHREAD=1
```

调用方通过多个线程并发发起查询。每次调用的 batch 保持完整，不拆分到多个副本。

### 3.6 测试设计

#### 3.6.1 测试方法与分层

采用单元测试（unit test）、集成测试（integration test）、端到端测试（e2e test）分层验证。参考mask独立使用`IDSelectorBitmap::is_member(ids[p])`逐位置生成，与实际转换结果逐字节比较，再对比设备检索输出。

现有新增功能用例均构造NPU索引，并非无设备可执行的纯主机单测。以下区分代码已覆盖的用例与扩展回归设计，不将设计清单记为全部已通过。

| 层级 | 方法 | 参考与判定 |
|------|------|------------|
| 单元/组件测试 | 直接验证mask转换及`identityIds`状态 | 逐字节比较独立参考mask；状态与ID映射一致 |
| 集成测试 | 公共接口贯通转换、传输、设备过滤和ID映射 | 与手工mask检索、单设备参考索引对比 |
| E2E | 建库、构造Bitmap、查询、变更及多设备并发 | 有效ID均属于候选集，变更后结果正确，性能按实际完成查询数统计 |

#### 3.6.2 单元测试（Unit Test）

`TestAscendIndexFlatUT.bitmapMaskFastPath`已覆盖：

| 编号 | 输入或操作 | 核验内容 |
|------|------------|----------|
| UT-01 | 分两次添加19条连续ID向量 | 增量添加保持快路径 |
| UT-02 | 交替位及末字节全一 | 非8倍数底库的尾部无效位清零 |
| UT-03 | 短位图、空位图、长位图 | 缺失范围补零，超出范围忽略，与参考mask一致 |
| UT-04 | 删除中间ID | 位置变化后退出直接复制路径，重映射正确 |
| UT-05 | 重置并重新添加 | 重新建立连续ID快路径 |
| UT-06 | 仅删除尾部ID | 映射仍连续时保留快路径 |
| UT-07 | CPU Flat、逆序IDMap导入 | 根据导入后的实际映射选择路径 |

该用例同时使用FP32/FP16公共接口对照参考mask检索结果。现有数据配置为64维、IP、batch=3、K=3，不能据此声称已覆盖全部维度与L2。

#### 3.6.3 集成测试（Integration Test）

| 编号 | 现有用例 | 已覆盖内容 |
|------|----------|------------|
| IT-01 | `TestAscendIndexFlatUT.searchWithSelectorSharedByBatch` | 非连续ID、批内共享Range/Batch/Bitmap、FP32/FP16、删除后查询，比较距离及外部ID |
| IT-02 | `TestAscendIndexFlatReplicaUT.sharedBaseAndSelectorSearch` | 双设备完整底库与ID映射、轮询Selector和mask查询、与单设备对比 |
| IT-03 | 同IT-02 | 普通查询、CPU导出与重新构造、副本删除逻辑计数、重置及连续ID Bitmap查询 |
| IT-04 | 同IT-02 | 双线程Range查询一致性；不将其记为不同Bitmap并发场景已覆盖 |

副本测试在未设置多线程开关或设备少于2个时打印`skip replica test`并直接返回。验收必须检查日志，不能只凭进程退出码或测试汇总将该情况判为副本验证通过。

#### 3.6.4 端到端测试（E2E Test）

**功能流程：** 按[AscendIndexFlat的Flat Bitmap过滤检索](../../zh/api/01_full_retrieval/08_AscendIndexFlat.md#flat-bitmap-filtering)构造非连续ID底库，选取候选集合并检索；删除一个候选后不重排Bitmap，再检索，检查所有有效ID属于过滤集合且已删除ID不再出现。多设备流程同时检查各副本的向量数、ID映射和逻辑`ntotal`。

**性能流程：** 使用`examples/TestAscendIndexFlat.cpp`中的`FilterQPS`和`ReplicaFilterQPS`，覆盖第2.2节矩阵，读取`bitmap`行。示例仍包含Range、Batch和Array，Array仅测1%，没有Bitmap专用命令行开关。

- 单设备：先检查全一mask与普通检索一致，再检查Selector与预生成mask结果及成员关系；比较`plain`、`mask`、`idselector`三条路径。
- 多设备：检查每个设备完整底库与ID映射，预热后使用与设备数相同的工作线程并发查询，分别记录3次`total_qps`及其中位数。
- 示例使用连续ID，主要体现Bitmap快路径；不能将其性能结果直接推广为自定义稀疏ID路径的性能。
- 正确性断言失败的数据不作为性能依据。多设备预热检查不等同于对全部计时查询做逐条正确性验证，正确性还需结合功能测试。

#### 3.6.5 扩展回归用例设计

以下用例属于完整回归设计；当前新增用例未逐项实现，不记为已有自动化覆盖：

| 编号 | 场景 | 预期结果 |
|------|------|----------|
| REG-01 | L2 × FP32/FP16，以及更多合法维度和batch | 与同数值路径参考mask检索对齐 |
| REG-02 | 候选数量0、1、K−1、K、K+1及全量候选 | 有效TopK正确，忽略`-1`标签对应距离 |
| REG-03 | 同一Bitmap两次调用间修改；Selector、普通和mask查询交替 | 无过期过滤或前次状态残留 |
| REG-04 | 多线程使用相同/不同只读Bitmap，独立输出缓冲 | 无串扰，结果分别满足各自候选集 |
| REG-05 | 查询与添加/删除并发 | 在锁保护下观察一致状态，无部分更新 |
| REG-06 | 空Selector、非法n/k、多设备分片及未开启副本线程安全开关 | 明确抛出参数或配置异常 |
| REG-07 | 长时间重复查询与更新 | 无持续资源增长，无副本映射漂移 |

小规模独立CPU TopK可作为补充参考，需考虑FP16转换和累加路径；同分候选按距离及集合合法性判断，不新增未经确认的误差阈值。

#### 3.6.6 执行方式与验收记录

按现有UT构建流程生成`TestAscendFlat`后，在该可执行文件目录运行；需有64维Flat/AICPU算子：

```bash
./TestAscendFlat --gtest_filter='TestAscendIndexFlatUT.bitmapMaskFastPath:TestAscendIndexFlatUT.searchWithSelectorSharedByBatch'
MX_INDEX_MULTITHREAD=1 ./TestAscendFlat --gtest_filter=TestAscendIndexFlatReplicaUT.sharedBaseAndSelectorSearch
```

准备4096维算子、编译安装最终IndexSDK后，从仓库根目录运行性能示例：

```bash
cd examples
bash build.sh TestAscendIndexFlat
./build/TestAscendIndexFlat --gtest_filter=TestAscendIndexFlat.FilterQPS
MX_INDEX_MULTITHREAD=1 FLAT_REPLICA_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    ./build/TestAscendIndexFlat --gtest_filter=TestAscendIndexFlat.ReplicaFilterQPS
```

`FLAT_REPLICA_DEVICES`仅由示例解析，默认0到15，按实际设备调整。单设备输出包含`call_ms`，多设备输出`total_qps`和`median`。QPS按查询向量数统计，不是调用次数。示例的达标打印条件为严格大于参考值；验收保留实际中位数及约定阈值，不仅依赖打印结论。

**验收记录：** 需求方已反馈最终提交满足需求指标。具体吞吐、各比例/batch结果和硬件/软件原始日志由实际测试报告留档；本文不据此推定所有扩展回归用例已执行。本次文档完善不重复设备性能验收。

**准出检查：** 功能输出正确、必测设备用例未跳过、已约定性能配置达标，并完成普通查询、mask查询和默认分片模式回归。新增结构成员后使用方重新编译。

## 4. 缺点和风险

### 4.1 潜在风险

#### 正确性风险

- 将外部 ID 位图直接解释为位置 mask，会在自定义 ID 或删除场景下产生错误。
- 快路径状态维护遗漏，可能错误启用直接复制。
- 短位图、非整字节底库和候选不足 K 容易产生边界问题。
- 多副本更新异常可能导致底库或 ID 映射不一致。

#### 性能风险

- 自定义 ID 场景仍需要扫描 ID 映射。
- 每条 query 仍复制并下发一份 mask，开销随 batch 和底库规模增加。
- FP32 查询转换、临时内存分配和设备传输可能限制吞吐。
- 原子轮询不考虑设备实时负载，执行时间差异可能导致等待。
- 16 die 吞吐受主机 CPU、内存带宽、设备连接拓扑及并发配置影响。

### 4.2 负面影响

- 副本模式下，每个设备保存完整底库，增加总体设备内存消耗。
- 添加和删除需要同步维护所有副本。
- 新增 ID 快路径状态，需要覆盖所有相关映射变更路径。
- 不适合用位图紧凑表达的稀疏、大范围外部 ID，可能需要较大的业务侧 Bitmap。

### 4.3 实现成本

#### 维护成本

- 维护 Bitmap 快路径状态与 ID 映射的一致性。
- 维护副本模式下的逻辑数量、导入导出及删除语义。
- 跟随现有 Flat 算子和运行环境进行回归。
- 保留 Bitmap 与通用 Selector 两条转换路径的测试覆盖。

### 4.4 应对措施

- 以当前 ID 映射构造参考 mask，验证快路径和重映射路径。
- 添加时增量检查，删除后重新检查快路径状态。
- 补齐短位图、尾部位、重置、导入及副本测试。
- 保留现有锁保护和副本一致性检查。
- 对 Bitmap 转换、查询转换、数据传输、设备检索分别测量，按实际瓶颈优化。
- 在性能证据不足时，保持现有轮询和算子输入布局。

## 5. 现有技术

### Faiss Selector 机制

`faiss::IDSelector` 通过 `is_member(id)` 表达 ID 是否属于候选集合。

`faiss::IDSelectorBitmap` 使用位图存储集合，ID `i` 对应第 `i / 8` 个字节中的第 `i % 8` 位。位图适合 ID 范围相对紧凑的候选集合。

### 现有 Ascend Flat 过滤能力

`search_with_masks` 接收按底库位置排列的 mask，通过 `Index::searchFilter` 调用现有 Flat 过滤检索链路。

本提案在该能力之上增加外部 ID 适配和 Bitmap 优化，不新增设备算子。

### 可复用组件

| 组件 | 来源 | 复用方式 |
|------|------|----------|
| `IDSelector` / `IDSelectorBitmap` | Faiss | 过滤条件表达 |
| `idxDeviceMap` | Ascend 索引实现 | 外部 ID 与位置映射 |
| `searchFilter` | 现有过滤检索 | 执行设备端过滤 |
| Flat 距离及 TopK 算子 | 现有 Flat | 计算过滤集合内 TopK |
| `AscendMultiThreadManager` | 现有多线程管理 | 索引锁与模式管理 |
| `AscendTensor` | 现有资源管理 | 设备查询和临时资源 |
| `FAISS_THROW_IF_NOT*` | 现有异常体系 | 参数及返回码检查 |

### 借鉴与差异

- **借鉴**：Faiss Selector 的集合语义，以及现有 Ascend Flat 的位置 mask 检索链路。
- **差异**：增加主机侧 ID 到位置的适配，并为连续 ID Bitmap 提供直接复制路径；多设备采用完整副本处理不同请求。

## 6. 未解决问题与后续事项

本期交付范围已按最终代码确定：C++接口、批内共享Selector、单设备与多设备完整副本；不新增Python绑定，不扩展多设备分片Selector查询。需求方已验证满足指标，性能不再表述为尚待实现。

后续事项为测试资料归档与可选演进：

1. 留档需求方实测的设备型号、CANN/驱动、线程配置、各允许比例及batch原始输出，明确达标配置范围。
2. 按第3.6.5节完善扩展回归；这些项目不应在缺少执行记录时标记为通过。
3. 仅在mask复制和传输成为实测瓶颈时，评估设备端共享单份mask；统一查询参数和Python绑定如有需求另行设计。

## 附录

### 参考资料

- Faiss Selector 定义：`faiss/impl/IDSelector.h`
- Ascend Flat 接口：`feature_retrieval/src/ascendfaiss/ascend/AscendIndexFlat.h/.cpp`
- Ascend Flat 实现：`feature_retrieval/src/ascendfaiss/ascend/impl/AscendIndexFlatImpl.h/.cpp`
- 现有过滤检索：`feature_retrieval/src/ascendfaiss/ascenddaemon/impl/Index.cpp`、`IndexFlat.cpp`
- 功能测试：`feature_retrieval/src/ascendfaiss/ut/cases/flat/TestAscendFlat.cpp`
- 性能测试：`examples/TestAscendIndexFlat.cpp`

### 术语表

| 术语 | 含义 |
|------|------|
| Flat | 对底库执行穷举距离计算的向量检索算法 |
| Selector | 指定允许参与处理的 ID 集合 |
| Bitmap | 按外部 ID 编码的过滤位图 |
| mask | 按底库物理位置编码的设备过滤掩码 |
| 外部 ID | 调用方感知的向量标识 |
| 底库位置 | 向量在设备底库中的存储位置 |
| 连续 ID 快路径 | 外部 ID 等于底库位置时直接复制 Bitmap |
| 副本模式 | 每个设备保存完整底库，分担不同查询调用 |
| 分片模式 | 底库分布在多个设备，检索后合并结果 |
| 允许通过比例 | 被 Selector 允许参与检索的底库向量比例 |
| QPS | 每秒处理的查询向量数 |

### 配套文档

- [中文API：AscendIndexFlat](../../zh/api/01_full_retrieval/08_AscendIndexFlat.md#search-with-selector)、[配置](../../zh/api/01_full_retrieval/09_AscendIndexFlatConfig.md#flat-shard-mode)：接口重载、Bitmap语义、副本约束及完整调用示例。
- [English API](../../en/api/01_full_retrieval/08_AscendIndexFlat.md)、[configuration](../../en/api/01_full_retrieval/09_AscendIndexFlatConfig.md)：对应英文说明。
- [示例说明](../../../examples/README.md)：算子准备、单设备与多设备测试命令及结果解释。
- 中英文API变更记录与用户指南目录同步增加入口。实际测试日志另行归档，不虚构性能数据。
