# Ascend950支持Flat算法技术方案设计(RFC)

**状态 (Status):** Draft
**作者 (Authors):** @chasefhb
**创建日期 (Created):** 2026-07-29
**更新日期 (Updated):** 2026-07-29
**相关 Issue/PR:** [#129](https://gitcode.com/Ascend/IndexSDK/issues/129)

---

# 1. 概述

## 1.1 简介

本提案旨在为 IndexSDK 补齐 Ascend950 NPU 平台上的 Flat 算法能力。Flat（全量检索 / Brute-force Search）是指对底库中的所有向量逐一计算距离，返回与查询向量距离最近的 TopK 结果。通过本提案的实现，IndexSDK 将能够在 Ascend950 NPU 上完整支持 Flat 算法的入库和检索全流程。

## 1.2 动机

### 背景

Flat 算法已在 310P、A2（910B）、A3 平台上支持，是向量检索场景中最基础且精度最高的检索方案。Ascend950 作为新一代高性能 NPU 平台，具有更强的计算能力和更高的内存带宽，适合大规模全量检索场景。

### 痛点

- Ascend950 NPU 平台缺少 Flat 算法支持，无法充分发挥硬件加速能力
- 现有 910B 平台上的 Flat 算子需要针对 Ascend950 进行适配优化

### 价值

- 补齐 Ascend950 平台的全量检索能力，完善 IndexSDK 的算法支持矩阵
- 充分利用 Ascend950 NPU 的强大计算能力，实现平均 1.25x 910B4 的性能提升

## 1.3 目标

### 目标

- 支持 `Flat` 算法，包括 L2 距离和内积（IP）距离度量
- 支持完整的入库和检索流程
- 性能平均 1.25x 910B4
- 基本功能支撑，正确执行

### 非目标

- 不涉及训练流程（Flat 为全量检索，无需训练）
- 不涉及属性过滤检索扩展

# 2. 用例分析

## 2.1 功能需求

### 核心功能

- **入库**: 支持 float 向量直接入库存储
- **检索**: 支持 L2 距离和内积（IP）距离的 TopK 检索
- **mask 检索**: 支持 `search_with_masks` 接口
- **数据拷贝**: 支持 `copyFrom`/`copyTo` 与 CPU Index 互转

### 参数规格

- **dim**: 向量维度，支持 32、64、128、256、384、512、768、1024、1408、1536、2048、3072、3584、4096
- **topk**: 返回数量，≤1024
- **base**: 库规模，千万级
- **metric**: 距离度量，支持 `METRIC_L2` 和 `METRIC_INNER_PRODUCT`
- **device**: 支持多设备

### 功能要求

- 支持完整的入库、检索流程
- 支持 L2 和 IP 两种距离度量
- 支持 batch 模式下的并发查询
- 支持 mask 过滤检索
- 返回结果保证排序正确性

## 2.2 性能需求

### 精度验收标准

与 CPU 基线对比，检索结果 recall 对齐：

- `dim`: 128、512、1024
- `topk`: 1、10、100
- `base`: 百万级至千万级
- recall@topk 与 CPU Flat 暴搜结果一致

### 性能验收标准

在 Ascend950 单卡场景下:

- 相比 910B4 平台，平均性能提升 1.25 倍
- 测试场景覆盖多 dim、多 topk 组合
- 基本功能正确执行，无精度回退

## 2.3 DFX 要求

### 可靠性

- 算法计算结果正确性保证
- 异常输入能够检查并输出异常

### 可测试性

- 提供单元测试用例
- 提供性能基准测试用例
- 提供精度对比测试用例

### 兼容性

- 与现有 AscendIndexFlat 接口保持兼容
- 不影响现有其他平台（310P/A2/A3）的功能

# 3. 方案设计

## 3.1 总体方案

### 设计思路

基于 IndexSDK 现有的 AscendIndexFlat 架构，扩展支持 Ascend950 NPU 平台。主要工作包括：

- 算子适配：将现有 910B 平台的 AscendC 算子适配到 Ascend950 平台
- 模型生成：复用 `flat_generate_model.py` 中已有的 Ascend950 分支，生成算子模型
- 性能优化：针对 Ascend950 平台特性进行性能调优

### 技术架构

```text
用户应用层
    ↓
AscendIndexFlat API
    ↓
AscendIndexFlatImpl
    ↓
├── 入库模块 (float 向量直接存储)
├── 检索模块 (距离计算 + TopK)
│   ├── L2 距离算子 (distance_flat_l2)
│   ├── IP 距离算子 (distance_flat_ip)
└── Scale 算子
    ↓
ACL Runtime
    ↓
Ascend950 NPU
```

### 核心流程

#### 入库阶段

1. 输入 float 向量数据
2. 将 float 向量按 block 对齐存储到底库
3. 建立向量 ID 映射

#### 检索阶段

1. 计算查询向量与底库全部 float 向量的距离（L2 或 IP）
2. TopK 选择和结果返回

## 3.2 技术选型

### 方案

复用现有 910B 平台的 AscendC 算子框架，针对 Ascend950 进行适配优化

### 选择理由

Flat 算法已在 910B 平台上有成熟的 AscendC 算子实现，复用现有框架可以：

- 保证功能一致性，降低开发风险
- 减少开发工作量，加快交付速度
- 便于后续维护和多平台统一

### Ascend950 平台特性

Ascend950 NPU 相比 910B4 具有以下优势：

- 更强的 AI Core 计算能力，支持更高吞吐量
- 更高的内存带宽，适合全量检索的密集访存场景
- 更低的时延，适合实时检索场景

## 3.3 功能与性能设计

### 功能实现方案

#### 入库模块

- **存储格式**: float 向量按 block 对齐直接存储，无量化压缩
- **数据流**: float 向量 → block 对齐 → 入库存储

#### 检索模块

- **L2 距离检索**: 计算查询向量与底库全部 float 向量的 L2 距离，返回 TopK
- **IP 距离检索**: 计算查询向量与底库全部 float 向量的内积距离，返回 TopK
- **mask 检索**: 支持 `search_with_masks` 接口，检索时过滤指定向量

### 算子设计

#### 需适配的算子列表

| 算子名称 | 功能 | 对应文件 |
|----------|------|----------|
| `distance_flat_l2` | L2 距离计算 | `distance_flat_l2.cpp` |
| `distance_flat_ip` | IP 距离计算 | `distance_flat_ip.cpp` |

#### 算子适配要点

**1. `pipe_barrier` API 兼容性**:

Ascend950（CANN 9.1.0+）编译器对 C 风格 `pipe_barrier(PIPE_V)` 增加了参数范围校验，要求参数在 [4, 6] 之间，`PIPE_V`（值为 3）不在合法范围内。需改为模板形式 `PipeBarrier<PIPE_V>()`。

**2. 数据格式**:

- 使用 ND 格式存储数据，提高内存访问效率
- 数据按 burst 长度对齐（Ascend950: 64，需根据平台特性调整）

**3. 并行策略**:

- 使用多核并行计算，充分利用 Ascend950 的 AI Core
- 使用向量指令加速 float 距离计算

**4. 内存管理**:

- 使用内存池管理算子内存
- 支持异步执行模式

### 模型生成脚本适配

`flat_generate_model.py` 已包含 Ascend950 分支判断：

```python
if '910' in args.npu_type or '950' in args.npu_type:
    ...
    generate_ascendc_flat_offline_model(map_args, args, _Z_DEFAULT, config_path, soc_version)
```

`generate_ascendc_flat_offline_model` 函数生成以下算子模型：

| 算子 | 说明 |
|------|------|
| `ascendc_distance_flat_ip` | IP 距离计算算子 |
| `ascendc_distance_flat_l2` | L2 距离计算算子 |

### 性能优化策略

1. **内存优化**: 优化数据布局，提高内存访问效率
2. **计算优化**: 利用 Ascend950 NPU 的强大并行计算能力，优化批量处理
3. **流水线优化**: 重叠计算和数据传输，提高吞吐量
4. **多核均衡**: 合理分配计算任务到多个 AI Core

### 性能对比

相比 910B4 平台，Ascend950 平台预期性能提升：

- 平均性能提升约 1.25 倍

### 影响范围

- **修改文件**:
  - `distance_flat_l2.cpp`: 适配 `PipeBarrier<PIPE_V>()`
  - `distance_flat_ip.cpp`: 适配平台特性
  - `ascendc_distance_flat_ip_maxs_with_mask.cpp`: 适配平台特性
  - `flat_generate_model.py`: 确认 950 分支下 mask 算子生成路径
- **新增文件**: 无（复用现有算子框架）
- **不影响**现有其他平台（310P/A2/A3）的实现

## 3.4 安全隐私与DFX设计

### 安全隐私

- 不涉及用户敏感数据处理
- 算子实现遵循安全编码规范

### 兼容性

- API 接口保持向后兼容
- 支持与现有 AscendIndexFlat 接口无缝对接

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

- 硬件平台: Ascend950 NPU（标卡）
- 软件环境: CANN 9.1.0+ 工具链，ACL Runtime

#### 开发约束

- 支持 C++11 及以上标准
- 需要安装 CANN 软件栈
- 需要配置 NPU 驱动和固件

#### 可验收设计

- 功能验收: 通过单元测试和集成测试
- 性能验收: 与 910B4 平台对比，平均 1.25x 性能提升
- 精度验收: 与 CPU 参考实现对比，recall 对齐

### 3.5.2 核心方法说明

**构造函数**：

- `dims`：向量维度
- `metric`：距离度量类型，支持 `METRIC_L2` 和 `METRIC_INNER_PRODUCT`
- `config`：配置参数

**检索方法**：

- `search(nq, x, k, distances, labels)`：检索 TopK 结果（继承自基类）
- `search_with_masks(nq, x, k, distances, labels, mask)`：带 mask 过滤的检索

**数据拷贝方法**：

- `copyFrom(index)`：从 CPU 索引拷贝到 NPU 索引
- `copyTo(index)`：从 NPU 索引拷贝到 CPU 索引

### 3.5.3 编程手册设计

需要在现有《IndexSDK 用户指南》中更新 Flat 算法支持矩阵，新增 Ascend950 平台支持说明。

#### Ascend950 平台 Flat 算法使用指南

1. **环境准备**：
   - 安装 CANN 9.1.0+ 工具链
   - 配置 NPU 驱动和固件
   - 设置环境变量

2. **生成算子模型文件**：

   ```bash
   # 生成 Ascend950 平台 512 维 Flat 算子
   python3 flat_generate_model.py -d 512 -t Ascend950PR

   # 生成 128 维 Flat 算子
   python3 flat_generate_model.py -d 128 -t Ascend950PR
   ```

# 4. 缺点和风险

## 4.1 潜在风险

### Breaking Change

- 无 Breaking Change，完全向后兼容

### 性能风险

- Ascend950 平台性能需要充分测试验证，确保达到 1.25x 910B4 的目标
- 不同 dim/topk 组合下的性能表现可能存在差异

### 复杂度提升

- 增加了平台适配代码，维护成本略有提升
- `pipe_barrier` API 兼容性问题需确保不影响已有平台

## 4.2 负面影响

### 对现有功能的影响

- 不影响现有其他平台（310P/A2/A3）的实现
- 不影响现有其他算法的功能

### 对用户的影响

- 用户需要升级到支持 Ascend950 的版本
- 需要重新生成 Ascend950 平台的算子模型文件

## 4.3 实现成本

### 开发成本

- 预计开发工作量: 2 人周
- 主要工作: 算子适配（`PipeBarrier` API 兼容性修复）、模型生成脚本确认、性能测试验证

### 维护成本

- 长期维护成本低，复用现有框架
- 需要跟进 CANN 版本更新

## 4.4 应对措施

- 充分的单元测试和集成测试
- 性能基准测试和优化
- 与 910B4 平台进行对比验证
- 详细的文档和示例代码

# 5. 现有技术

## IndexSDK 现有实现

- AscendIndexFlat 已在 IndexSDK 中实现，支持 310P/A2/A3 平台
- 910B 平台有成熟的 AscendC 算子实现（`distance_flat_l2`、`distance_flat_ip`、`ascendc_distance_flat_ip_maxs_with_mask`）
- `flat_generate_model.py` 已包含 950 分支判断（`'910' in args.npu_type or '950' in args.npu_type`）
- 本提案复用现有架构和算子框架，扩展支持 Ascend950

# 6. 未解决问题

---

## 附录

### 参考资料

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)
- [IndexSDK 用户指南](../../zh/05_user_guide.md)
- [Ascend NPU 开发文档](https://www.hiascend.com/document)

### 术语介绍

- **Flat**: 全量检索（Brute-force Search）算法
- **dim**: 向量维度
- **topk**: 返回的最近邻数量
- **mask**: 检索过滤掩码
- **IP**: Inner Product，内积距离
- **L2**: 欧几里得距离
- **Ascend950**: 华为新一代高性能 NPU 平台
- **910B4**: Ascend 910B4 NPU 平台
- **PipeBarrier**: AscendC 算子中的流水线同步 API

### 文档更新计划

- RFC 评审通过后，更新《IndexSDK 用户指南》中 Flat 算法的硬件支持矩阵
- 更新《快速开始指南》，添加 Ascend950 平台 Flat 算子生成说明
