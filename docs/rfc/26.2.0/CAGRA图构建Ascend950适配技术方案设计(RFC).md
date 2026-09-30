# CAGRA 图构建 Ascend 950 适配技术方案设计（RFC）

**状态（Status）：** Draft

**作者（Authors）：** @meijq

**创建日期（Created）：** 2026-09-17

**更新日期（Updated）：** 2026-09-17

**相关模块：** `AscendCagraGraphBuilder`、CAGRA AICore 构图算子

---

# 1. 概述

## 1.1 背景

Index SDK 已有 `AscendIndexCagra` 图检索能力，但原有流程依赖外部 CPU 脚本生成图文件。百万级及以上底库在 CPU 侧执行 NN-Descent、剪枝并生成固定出度图，会增加离线构建时间和 Host 内存开销，也无法使用 Ascend 950 的 SIMT 并行资源。

本方案新增 `AscendCagraGraphBuilder`，在 Ascend 950 上完成 CAGRA 构图。输入为 FP32 底库向量，输出为行优先的固定出度邻接矩阵 `uint32_t[dataNum][graphDegree]`。该图可以直接传入 `AscendIndexCagra::Add`，构图和检索保持为两个独立阶段。

## 1.2 目标

- 在单张 Ascend 950 上完成 NN-Descent 近似 KNN 图构建、CAGRA 剪枝、反向边补齐和连通性保障。
- 使用 AICore SIMT 实现距离计算、候选生成、图更新和剪枝等主要计算。
- 提供公共 C++ 接口，以及与静态 shape 匹配的离线模型生成脚本。
- 输出固定出度、无非法节点、无自环、行内无重复节点的邻接矩阵。
- 支持通过更新数阈值提前结束 NN-Descent 迭代。
- 保留 CPU 构图脚本，作为功能参考和不具备 NPU 环境时的替代方案。

## 1.3 非目标

- 本 RFC 不修改 CAGRA 在线检索算法和 RaBitQ 量化格式。
- 本 RFC 不提供多卡或分布式构图。
- 本 RFC 不承诺任意维度和任意 TopK 的端到端检索；相关限制和后续改造见第 6 章。

## 1.4 平台支持

| 项目 | 当前支持情况 |
| --- | --- |
| 硬件 | Ascend 950 系列 |
| Index SDK 安装参数 | `--platform=Ascend950` |
| ATC 模型参数 | `-t Ascend950PR`，映射到 `Ascend950PR_957c` |
| 构图卡数 | 单卡 |
| 输入类型 | FP32 |
| 距离类型 | L2 |
| 默认计算路径 | AICore SIMT |

# 2. 用例与数据流

## 2.1 典型用例

1. 用户准备 `dataNum × dim` 的 FP32 底库向量。
2. 按实际 `dataNum`、`dim`、`intermediateDegree` 和 `graphDegree` 生成构图 OM。
3. 创建 `AscendCagraGraphBuilder` 并调用 `Init`。
4. 调用 `Build` 获得内存中的邻接矩阵，或者调用 `BuildToFile` 直接生成图文件。
5. 对图结构和抽样近邻准确率进行验证。
6. 当参数满足检索约束时，将图传给 `AscendIndexCagra::Add` 完成后续量化和检索。

## 2.2 总体数据流

```text
FP32底库向量
    │
    ▼
CagraNndInit
    │ 初始M度近邻图
    ▼
┌──────────── NN-Descent迭代 ────────────┐
│ CagraNndSampleReverse                  │
│          ▼                             │
│ CagraNndLocalJoin（默认SIMT）           │
│          ▼                             │
│ CagraNndUpdate ── updateCount ── Host  │
└────────────────────────────────────────┘
    │ 收敛或达到maxIterations
    ▼
CagraPruneReverse
    ▼
CagraMerge
    ▼
Host连通性保障（可关闭）
    ▼
uint32_t[dataNum][graphDegree]
```

# 3. 算法与算子设计

## 3.1 算子划分

构图按算法阶段划分为六个粗粒度算子，避免把排序、去重等小步骤拆成大量独立算子。

| 算子 | 功能 | 主要输出 |
| --- | --- | --- |
| `CagraNndInit` | 为每个节点生成不重复的初始邻居，并计算 FP32 L2 距离 | M 度中间图及距离 |
| `CagraNndSampleReverse` | 从 new/old 邻居采样，同时生成采样反向边 | forward/reverse new/old 及各组长度 |
| `CagraNndLocalJoin` | 对共同中心产生的候选点对计算 L2 距离 | 候选 ID、距离和候选计数 |
| `CagraNndUpdate` | 合并原图与候选，去重、排序并保留前 M 条边 | 下一轮中间图、距离和更新数 |
| `CagraPruneReverse` | 按绕路数执行 CAGRA 剪枝，并收集反向边 | 剪枝图、反向图和计数 |
| `CagraMerge` | 保留近邻并用反向边补足固定出度 G | 最终 G 度图 |

其中 `M=intermediateDegree`，`G=graphDegree`，采样度数为 `min(32, M)`。

## 3.2 Host 与 Device 职责

Device 侧负责：

- FP32 L2 距离计算；
- new/old 采样和反向边生成；
- 候选生成、去重、排序与截断；
- CAGRA 绕路剪枝、反向边收集和最终出度合并。

Host 侧负责：

- 创建并复用 Device 中间缓冲区；
- 按固定顺序调度六类算子，并交换图和距离双缓冲；
- 每轮读取一个 `uint32_t updateCount` 判断是否收敛；
- 可选的最终连通性保障；
- 将最终邻接矩阵回传到调用者内存或写入文件。

## 3.3 收敛条件

停止阈值为：

```text
stopUpdates = terminationThreshold × dataNum × intermediateDegree
```

当一轮 `updateCount <= stopUpdates` 时提前结束；否则最多执行 `maxIterations` 轮。达到最大轮数并不表示图已经收敛，性能测试和精度验收必须同时记录最终更新数和 `converged` 状态。

## 3.4 连通性保障

当 `guaranteeConnectivity=true` 时，Host 基于收敛后的 M 度中间图构造受出度约束的双向骨架，将弱连通分量连接到最大分量，再与 NPU 输出的 G 度图合并。该步骤保证最终图弱连通，并优先保护骨架边。

# 4. 接口设计

## 4.1 初始化接口

```cpp
APP_ERROR Init(int dim, int graphDegree, int dataNum,
               const std::vector<int>& deviceList);
```

- `dim`：输入向量维度。
- `graphDegree`：最终图每行的邻居数 G。
- `dataNum`：底库向量数量 N。
- `deviceList`：仅允许包含一个 Device ID。

## 4.2 构图配置

```cpp
struct AscendCagraGraphBuildConfig {
    uint32_t intermediateDegree{128};
    uint32_t maxIterations{20};
    float terminationThreshold{0.0001F};
    bool verbose{false};
    bool guaranteeConnectivity{true};
};
```

| 参数 | 含义 | 影响 |
| --- | --- | --- |
| `intermediateDegree` | NN-Descent 中间图出度 M | 越大通常图质量越高，但显存、计算量增加 |
| `maxIterations` | 最大迭代轮数 | 防止未收敛时无限迭代 |
| `terminationThreshold` | 更新比例阈值 | 越小收敛条件越严格 |
| `verbose` | 输出逐阶段耗时和候选统计 | 用于性能与精度诊断 |
| `guaranteeConnectivity` | 是否执行 Host 连通性保障 | 默认开启；会产生中间图 D2H 拷贝和 Host 处理 |

## 4.3 构图接口

```cpp
APP_ERROR Build(const float* data, uint32_t* graph,
                const AscendCagraGraphBuildConfig& config = {});

APP_ERROR BuildToFile(const float* data,
                      const std::string& graphFilePath,
                      const AscendCagraGraphBuildConfig& config = {});
```

`Build` 要求调用者为 `graph` 分配 `dataNum × graphDegree × sizeof(uint32_t)` 字节。`BuildToFile` 按相同的行优先布局写入二进制文件。

## 4.4 调用示例

```cpp
#include "index/AscendCagraGraphBuilder.h"

faiss::ascend::AscendCagraGraphBuilder builder;
std::vector<int> devices{0};
int ret = builder.Init(128, 64, dataNum, devices);
if (ret != 0) {
    return ret;
}

faiss::ascend::AscendCagraGraphBuildConfig config;
config.intermediateDegree = 128;
config.maxIterations = 20;
config.terminationThreshold = 0.0001F;
config.verbose = true;
config.guaranteeConnectivity = true;

std::vector<uint32_t> graph(static_cast<size_t>(dataNum) * 64);
ret = builder.Build(baseData.data(), graph.data(), config);
```

# 5. 模型生成与部署

## 5.1 构图模型命令

```bash
python3 cagra_build_generate_model.py \
    --cores 56 \
    -n 1000000 \
    -d 128 \
    -i 128 \
    -graph 64 \
    -p 0 \
    -t Ascend950PR
```

构图 OM 为静态 shape。模型生成参数 `N/D/M/G` 必须与 `Init` 和 `Build` 的实际参数完全一致。不同规模或参数组合需要分别生成模型，并共同放在 `MX_INDEX_MODELPATH` 目录下。

## 5.2 构图与检索模型的关系

构图命令没有 `topK` 参数。`topK` 只影响在线检索模型：

```bash
python3 cagra_generate_model.py \
    -data_base 1000000 \
    -d 128 \
    -degree 64 \
    -topK 32 \
    -p 0 \
    -t Ascend950PR
```

如果只需要生成图文件，无需生成检索模型。上面的 `D=128、G=64、topK=32` 是本次构图 POC 采用的端到端联调规格，不改变既有 CAGRA 检索接口的支持范围。

# 6. 参数约束与维度扩展

## 6.1 当前实现约束

| 参数 | NPU 构图 | CAGRA RaBitQ 检索 | 端到端当前可用组合 |
| --- | --- | --- | --- |
| `dim` | SIMT 接受整数 `[1, 3072]`；已验证 64/128/256/384/512/1024/1536/2048/3072 | `{64, 128, 256, 512}` | 取二者共同支持的规格；新增高维检索规格不属于本 RFC 范围 |
| `dataNum` | `[2, INT32_MAX]`，并受 Device/Host 内存限制 | `(0, 1e9]` | OM 与运行值一致 |
| `graphDegree` | `1 ≤ G ≤ M < N`，且 `M ≤ 128`；当前已验证 G=64 | `{64, 128, 256, 512}` | 取二者均支持的规格 |
| `intermediateDegree` | `G ≤ M ≤ 128` | 不适用 | 推荐 128 |
| `topK` | 不适用 | `(0, 4096]` | 由检索模型决定 |
| `cores` | 固定 56 | 由检索模型决定 | 56（构图） |

构图输入必须全部为有限 FP32 数值，不支持 NaN、正无穷和负无穷。非有限值会破坏 L2 距离的排序关系，调用方需要在构图前完成数据清洗。

`dim=129` 等非 128 对齐维度可以进入 SIMT 通用距离路径，但还需要在 A5 上补充模型编译、图结构和图 Recall 验证。既有检索能力不属于本阶段修改范围，本次端到端回归仍使用 `D=128、G=64、topK=32`。

## 6.2 为什么构图不能只修改生成脚本支持 129 维

构图 `NndLocalJoinSimt` 目前按 8 个线程组成一个距离计算组，并为每个线程分配 16 个 FP32 缓存：

```cpp
constexpr uint32_t kDistanceTeamSize = 8;
float lhsValues[16];
```

原缓存容量为 `8 × 16 = 128` 维。当前实现保留 `dim ≤ 128` 的缓存快路径；`128 < dim ≤ 512` 由 8 线程距离组按 128 维分块读取 lhs，每次复用该分块计算 4 个 rhs 候选；`dim > 512` 以两个 rhs 为一组直接流式遍历，每个维度只读取一次 lhs 标量，并避免使用批处理私有数组。模型生成脚本和 Host 接口同步放开到 3072 维。

## 6.3 构图扩展到多维度的设计方向

构图侧扩展到常用维度以及 1024、1536、3072 维，采用以下方案：

1. 保留 `dim ≤ 128` 的现有缓存路径，避免 128 维基线性能回退。
2. `128 < dim ≤ 512` 使用 4 候选批处理的流式 L2 距离路径；`dim > 512` 使用无 lhs 私有数组的双 rhs 流式路径。后者仅保留两个距离累加器，同时把 lhs 元素读取量减半，在访存复用与线程资源占用之间取得平衡。
3. 硬件验证覆盖 `dim ∈ {64, 128, 256, 384, 512, 1024, 1536, 2048, 3072}`，每个规格单独生成静态 shape OM。
4. 对每个规格验证非法边、自环、重复边、连通性、抽样图 Recall 和构图性能。

## 6.4 cuVS 参数边界参考

cuVS 主线 CAGRA 接口没有给 `dim` 设置离散白名单。数据集以二维矩阵传入，维度由矩阵第二维在运行时决定，因此其实现思路也是通用维度加运行时资源选择，而不是为 64/128/256 等维度分别定义算法接口。具体构建后端仍可能因数据类型、显存和内部算法参数产生约束。

cuVS 的默认 `intermediate_graph_degree` 为 128，默认 `graph_degree` 为 64，并要求中间图度数不小于输出图度数；源码建议中间图度数取输出图度数的 1.5 到 2 倍并按 32 对齐。它没有为 CAGRA API 声明一个类似 `{64, 128, 256, 512}` 的 degree 白名单。

`topK` 属于检索输出参数，不参与 CAGRA 构图。cuVS 要求检索 `topK ≤ itopk_size`，`itopk_size` 默认 64 并向 32 对齐；单 CTA 实现还要求 `itopk_size ≤ 512`，AUTO 会在不适合单 CTA 时选择多 CTA 路径。因此 cuVS 的 topK 边界与具体搜索实现和资源配置相关，不能直接作为本项目构图算子的约束。

# 7. 性能与验证

## 7.1 功能验证

构图完成后至少检查：

- 所有邻居 ID 均小于 `dataNum`；
- 每行不存在自环；
- 每行不存在重复邻居；
- 每行邻居数等于 `graphDegree`；
- 弱连通分量数、最大连通分量比例和零入度节点数；
- 抽样节点的图近邻 Recall@1/10/32；
- 在同一张图上使用 CPU 精确距离遍历得到检索 Recall，隔离图质量与量化检索误差。

## 7.2 性能口径

性能报告必须包含：

- 数据规模 `N/D/M/G`；
- 实际迭代轮数、最终更新数、停止阈值和是否收敛；
- Init、SampleReverse、LocalJoin、Update、PruneReverse、Merge 和连通性耗时；
- 端到端构图时间和 vectors/s；
- 图质量指标。

不能用未收敛图的较短耗时作为最终性能结果，也不能把 CPU 精确查询 GT 耗时当成 CPU 构图耗时。

## 7.3 当前 POC 结果

在 SIFT1M 的 `N=1,000,000、D=128、M=128、G=64` 配置下，A5 SIMT 构图耗时约 38.32 秒，吞吐约 2.61 万向量/秒。最终图弱连通分量数为 1、零入度节点数为 0；使用 CPU FP32 精确距离遍历该图时，Recall@1/10/32 分别为 0.9900、0.9950 和 0.9856。该结果用于验证构图功能与质量，不作为不同数据集和硬件版本下的固定 SLA。

# 8. 风险与后续工作

## 8.1 风险

- OM 采用静态 shape，参数不一致会导致模型匹配失败。
- NN-Descent 属于近似构图，迭代数和候选容量会影响图质量。
- `guaranteeConnectivity=true` 包含 D2H 回传和 Host 计算，大规模场景需要关注 Host 内存。
- 已完成常用维度和 1024/1536/2048/3072 维的 A5 功能验证；不同静态 shape 仍需分别生成 OM 并回归。

## 8.2 后续工作

1. 补充更多真实业务数据集上的图 Recall 与端到端检索 Recall。
2. 建立多数据规模、多维度的自动化模型生成和测试矩阵。
3. 将连通性保障迁移到 Device 或降低中间图回传成本。
4. 按第 6.3 节完成多维度 A5 验证和高维 SIMT 距离性能优化。

# 9. 参考资料

- [Index SDK 用户指南](../../zh/05_user_guide.md)
- [AscendIndexCagra API](../../zh/api/02_approximate_retrieval/18_AscendIndexCagra.md#ascendindexcagra)
- [RAPIDS cuVS CAGRA 接口](https://github.com/rapidsai/cuvs/blob/main/cpp/include/cuvs/neighbors/cagra.hpp)
- [RAPIDS cuVS CAGRA 搜索参数检查](https://github.com/rapidsai/cuvs/blob/main/cpp/src/neighbors/detail/cagra/search_plan.cuh)
