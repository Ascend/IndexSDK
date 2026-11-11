# CAGRA 图构建 Ascend 950 PR 适配技术方案设计（RFC）

**状态（Status）：** Draft

**作者（Authors）：** @meijq

**创建日期（Created）：** 2026-09-17

**更新日期（Updated）：** 2026-11-06

**相关模块：** `AscendIndexCagra`、CAGRA AICore 构图与检索算子

---

# 1. 概述

## 1.1 背景

Index SDK 已有 `AscendIndexCagra` 图检索能力，但原有流程依赖外部 CPU 脚本生成图文件。百万级及以上底库在 CPU 侧执行 NN-Descent、剪枝并生成固定出度图，会增加离线构建时间和 Host 内存开销，也无法使用 Ascend 950 PR 的 SIMT 并行资源。

本方案在 Ascend 950 PR 上完成 CAGRA 构图，输入为 FP32 底库向量，构图产物为行优先的固定出度邻接矩阵 `uint32_t[dataNum][graphDegree]`。对外由 `AscendIndexCagra` 统一管理构图、底库量化、查询和图导出，配置统一使用`AscendIndexCagraConfig`。

## 1.2 目标

- 在单张 Ascend 950 PR 上完成 NN-Descent 近似 KNN 图构建、CAGRA 剪枝、反向边补齐和连通性保障。
- 使用 AICore SIMT 实现距离计算、候选生成、图更新和剪枝等主要计算。
- 提供公共 C++ 接口，以及与静态 shape 匹配的离线模型生成脚本。
- 输出固定出度、无非法节点、无自环、行内无重复节点的邻接矩阵。
- 支持通过更新数阈值提前结束 NN-Descent 迭代。
- 保留 CPU 构图脚本，作为功能参考和不具备 NPU 环境时的替代方案。

## 1.3 非目标

- 保持RaBitQ量化格式与距离计算，不引入FP32重排。
- 本 RFC 不提供多卡或分布式构图。
- 不提供增量插入、更新或删除；数据集变化后通过Reset和Train重新建库。

## 1.4 平台支持

| 项目 | 当前支持情况 |
| --- | --- |
| 硬件 | Ascend 950 PR系列产品 |
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
3. 创建`AscendIndexCagra`，配置`AscendIndexCagraConfig`并调用`Init`。
4. 调用`Train(n, data)`或`Add(n, data)`，完成构图、底库装载和底库量化。
5. 调用`GetKnnGraph`读取图，验证结构和抽样近邻准确率；需要文件时调用`SaveGraphToFile`。
6. 通过同一Index调用`Search`完成查询量化和检索。

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
图结构校验
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
APP_ERROR Init(int dim, const std::vector<int>& deviceList,
               const AscendIndexCagraConfig& config = {});
```

- `dim`：支持`{64,128,256,512,1024,1536,3072}`。
- `config`：构图参数；输出度数G为64或128，满足`G<=M<=128`。
- `deviceList`：仅允许包含一个 Device ID。
- Init保存配置，不提前加载检索模型；数据量N在Train/Add传入，须满足`M<N<=1e9`且内存足够。

## 4.2 构图配置

```cpp
struct AscendIndexCagraConfig {
    uint32_t graphDegree{64};
    uint32_t intermediateDegree{128};
    uint32_t maxIterations{20};
    float terminationThreshold{0.0001F};
    bool verbose{false};
    bool guaranteeConnectivity{true};
};
```

| 参数 | 含义 | 影响 |
| --- | --- | --- |
| `graphDegree` | 最终图出度G，支持64/128 | 影响图大小、遍历开销和邻居覆盖 |
| `intermediateDegree` | NN-Descent 中间图出度 M | 越大通常图质量越高，但显存、计算量增加 |
| `maxIterations` | 最大迭代轮数 | 防止未收敛时无限迭代 |
| `terminationThreshold` | 更新比例阈值 | 越小收敛条件越严格 |
| `verbose` | 输出逐阶段耗时和候选统计 | 用于性能与精度诊断 |
| `guaranteeConnectivity` | 是否执行 Host 连通性保障 | 默认开启；会产生中间图 D2H 拷贝和 Host 处理 |

## 4.3 建库、查询和图导出接口

```cpp
APP_ERROR Train(int n, const float* data);
APP_ERROR Add(int n, const float* data);
APP_ERROR Search(int n, const float* queries, int topK, float* distances, uint32_t* labels);
APP_ERROR GetKnnGraph(std::vector<uint32_t>& graph) const;
APP_ERROR SaveGraphToFile(const std::string& filePath) const;
APP_ERROR Reset();
bool IsTrained() const;
int GetSize() const;
```

Train/Add输入连续、有限的FP32底库，两者均完成全量建库；重复Train保持原状态，已建库再次Add返回`APP_ERR_INVALID_PARAM`。Reset清空底库并保留配置，之后可重新Train/Add。Train失败不发布半初始化状态。Search自动量化本次查询，支持Q为1至4096，TopK为`{32,64,128,256,512,1024}`，且TopK≤N，输出缓冲各至少Q×TopK个元素。

GetKnnGraph读取N×G个行优先邻居ID。SaveGraphToFile导出相同内容，文件为本机字节序uint32_t、无文件头，不含向量和量化数据；覆盖已有文件，父目录需存在，路径不得为空或含NUL。未训练返回APP_ERR_INDEX_NOT_INIT，文件操作失败返回非0，可能留下不完整文件，索引保持可用。

## 4.4 调用示例

```cpp
#include <faiss/ascend/AscendIndexCagra.h>

faiss::ascend::AscendIndexCagra index;
std::vector<int> devices{0};
faiss::ascend::AscendIndexCagraConfig config;
config.graphDegree = 64;
config.intermediateDegree = 128;
config.maxIterations = 20;
config.terminationThreshold = 0.0001F;
config.verbose = true;
config.guaranteeConnectivity = true;

int ret = index.Init(128, devices, config);
if (ret != 0) return ret;
ret = index.Train(dataNum, baseData.data());
if (ret != 0) return ret;
std::vector<uint32_t> graph;
ret = index.GetKnnGraph(graph);
if (ret != 0) return ret;
ret = index.SaveGraphToFile("cagra_graph.bin");
if (ret != 0) return ret;
std::vector<float> distances(static_cast<size_t>(queryNum) * topK);
std::vector<uint32_t> labels(static_cast<size_t>(queryNum) * topK);
ret = index.Search(queryNum, queryData.data(), topK, distances.data(), labels.data());
if (ret != 0) return ret;
```

## 4.5 单Index建库与查询入口

构图调度与检索实现统一放入`AscendIndexCagraImpl.cpp`，公开层`AscendIndexCagra.cpp`负责参数检查、锁和调用转发。六个构图算子仍独立实现，由Train内部的私有构图上下文管理临时资源。对外不再提供独立Builder类；原有外部图Add和手动量化接口保留。

- Init保存维度、单卡设备和构图配置，统一支持七档维度及自建图G=64/128；N与M的关系在Train前置校验。
- Train/Add构建整张图，释放构图临时资源后装载底库并执行底库量化。两者不提供增量插入；重复Train保持原状态，重复Add返回`APP_ERR_INVALID_PARAM`，Reset后可重新建库。
- 建库过程在临时实现对象中完成，全部成功后提交，失败不发布部分可用状态。
- Search自动准备当前查询量化数据，允许同一索引按次指定`{32,64,128,256,512,1024}`的TopK（TopK≤N）。算子句柄以TopK和分页大小为键缓存；每个使用的TopK仍需部署对应静态OM。
- 内部容量为64/128/256时沿用bitonic，512/1024接入radix路径；共享本地内存预算，保留量化距离，不做FP32重排。
- radix搜索每轮扩展1个节点并立即将新邻居合入候选池；以无可扩展父节点或达到轮数上限作为结束条件，取消仅观察前26名的连续未改进停止条件。上限为内部容量加图深度估计：从1开始反复乘以`max(2,G/2)`，达到N的次数即深度。N=1000000、G=64时，容量512/1024对应516/1028轮上限。bitonic保留原64轮上限和早停策略。大K的召回与耗时分别测试，轮数上限不等于实际执行轮数。
- 旧Init/Add/QuantizeData/QuantizeQueries/Search保留原手动准备与固定TopK语义。统一Init与旧外部图Add调用模式分开使用。

完整接口、配置默认值与示例见[AscendIndexCagra API](../../zh/api/02_approximate_retrieval/18_AscendIndexCagra.md)。验证覆盖参数白名单、失败状态、Reset重建、分页、多TopK缓存、七档维度与TopK至1024组合，以及旧接口回归。主机模拟与Ascend 950 PR实测结果分别记录。

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

构图OM为静态shape。模型生成参数N/D/M/G必须与Init配置和Train的数据量一致。不同规模或参数组合需要分别生成模型，并共同放在`MX_INDEX_MODELPATH`目录下。

## 5.2 构图与检索模型的关系

`cagra_generate_model.py`默认生成构图和检索模型：用同一组`-data_base/-d/-degree`配置先生成六个构图OM，再生成当前`-topK`的七档检索分页OM。`--intermediate-degree`默认128、`--cores`默认56，支持G=64/128，要求G<=M<=128且M<N；任一阶段失败即停止。独立构图脚本继续可用。算子包仍需在生成模型前编译安装。

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

如果仅调用Train和图导出，无需生成检索模型；Train仍会装载底库并完成量化。调用Search前须部署匹配N/D/G/TopK的检索OM。

# 6. 参数约束与维度扩展

## 6.1 当前实现约束

| 参数 | NPU 构图 | CAGRA RaBitQ 检索 | 端到端当前可用组合 |
| --- | --- | --- | --- |
| `dim` | `{64,128,256,512,1024,1536,3072}` | `{64,128,256,512,1024,1536,3072}` | `{64,128,256,512,1024,1536,3072}` |
| `dataNum` | `M<N<=1e9` | `G<N<=1e9` | `M<N<=1e9` |
| `graphDegree` | `{64,128}`, `G<=M<=128`, `M<N` | 外部图支持`{64,128}` | `{64,128}`, `G<=M<=128` |
| `intermediateDegree` | `G ≤ M ≤ 128` | 不适用 | 推荐 128 |
| `topK` | 不适用 | `{32,64,128,256,512,1024}`, `TopK<=N` | `Search` 按次指定 |
| `cores` | 固定 56 | 由检索模型决定 | 56（构图） |

构图输入必须全部为有限 FP32 数值，不支持 NaN、正无穷和负无穷。非有限值会破坏 L2 距离的排序关系，调用方需要在构图前完成数据清洗。

统一Index在Init阶段校验七档端到端维度；构图和后续查询使用同一D/G配置。SIFT1M精度和性能使用`D=128、G=64、topK=32`，其余支持组合另做功能回归。

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
3. 统一Index的硬件验证覆盖`dim ∈ {64,128,256,512,1024,1536,3072}`，每个规格单独生成静态shape OM。
4. 对每个规格验证非法边、自环、重复边、连通性、抽样图 Recall 和构图性能。

## 6.4 cuVS 参数边界参考

cuVS 主线 CAGRA 接口没有给 `dim` 设置离散白名单。数据集以二维矩阵传入，维度由矩阵第二维在运行时决定，因此其实现思路也是通用维度加运行时资源选择，而不是为 64/128 等维度分别定义算法接口。具体构建后端仍可能因数据类型、显存和内部算法参数产生约束。

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

在 SIFT1M 的 `N=1,000,000、D=128、M=128、G=64` 配置下，Ascend 950 PR SIMT 构图耗时约 38.32 秒，吞吐约 2.61 万向量/秒。最终图弱连通分量数为 1、零入度节点数为 0；使用 CPU FP32 精确距离遍历该图时，Recall@1/10/32 分别为 0.9900、0.9950 和 0.9856。该结果用于验证构图功能与质量，不作为不同数据集和硬件版本下的固定 SLA。

# 8. 风险与后续工作

## 8.1 风险

- OM 采用静态 shape，参数不一致会导致模型匹配失败。
- NN-Descent 属于近似构图，迭代数和候选容量会影响图质量。
- `guaranteeConnectivity=true` 包含 D2H 回传和 Host 计算，大规模场景需要关注 Host 内存。
- 独立Builder接口移除后，调用方需改用Index的Init、Train/Add和GetKnnGraph/SaveGraphToFile，并使用同版本头文件和库重新编译。
- 已完成常用维度和 1024/1536/2048/3072 维的 Ascend 950 PR 功能验证；不同静态 shape 仍需分别生成 OM 并回归。

## 8.2 后续工作

1. 补充更多真实业务数据集上的图 Recall 与端到端检索 Recall。
2. 建立多数据规模、多维度的自动化模型生成和测试矩阵。
3. 将连通性保障迁移到 Device 或降低中间图回传成本。
4. 按第 6.3 节完成多维度 Ascend 950 PR 验证和高维 SIMT 距离性能优化。

# 9. 参考资料

- [Index SDK 用户指南](../../zh/05_user_guide.md)
- [AscendIndexCagra API](../../zh/api/02_approximate_retrieval/18_AscendIndexCagra.md#ascendindexcagra)
- [AscendIndexCagra API与配置](../../zh/api/02_approximate_retrieval/18_AscendIndexCagra.md)
- [RAPIDS cuVS CAGRA 接口](https://github.com/rapidsai/cuvs/blob/main/cpp/include/cuvs/neighbors/cagra.hpp)
- [RAPIDS cuVS CAGRA 搜索参数检查](https://github.com/rapidsai/cuvs/blob/main/cpp/src/neighbors/detail/cagra/search_plan.cuh)
