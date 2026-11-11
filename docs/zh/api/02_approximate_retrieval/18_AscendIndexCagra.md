# AscendIndexCagra<a name="ZH-CN_TOPIC_0000002513157730"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002544797645"></a>

AscendIndexCagra在单张Ascend 950 PR系列设备上完成FP32/L2构图和RaBitQ近似最近邻检索。调用统一Init后，Train或Add完成全量构图、底库装载及量化，Search自动完成本次查询量化并检索，无需单独创建Builder。

构图配置统一使用AscendIndexCagraConfig。已建库的图可通过GetKnnGraph读取，或通过SaveGraphToFile写入文件。原有外部图Init、Add和手动量化接口保留兼容。

## 单Index接口

头文件为`faiss/ascend/AscendIndexCagra.h`，命名空间为`faiss::ascend`。返回`APP_ERROR`的方法以0表示成功、非0表示失败；输入、输出向量均为Host缓冲。

| 接口 | 作用与约束 |
| --- | --- |
| `Init(int dim, const std::vector<int>& deviceList, const AscendIndexCagraConfig& config = {})` | 保存维度、单卡设备和构图配置。对象只能初始化一次；此阶段不要求底库数量和TopK，不加载检索OM。 |
| `Train(int n, const float* data)` | 从连续FP32的`data[n][dim]`构建全量图，装入底库并完成底库量化。要求`M<n<=1e9`，实际规模受内存限制。成功后即可Search。失败保留原索引状态。 |
| `Add(int n, const float* data)` | 空索引上完成全量建库，不提供增量插入。已建库时返回`APP_ERR_INVALID_PARAM`并保留原数据，重新建库需先Reset；重复Train仍为无操作。 |
| `Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels)` | 对每次调用自动量化查询；`n`为1至4096，TopK支持`{32,64,128,256,512,1024}`，且TopK≤N。输出缓冲分别至少有`n*topK`个元素。可在同一索引上改变TopK。 |
| `Reset()` | 清空图、底库、量化数据和查询算子缓存，保留统一Init配置，可重新Train/Add；空索引Reset成功。 |
| `GetKnnGraph(std::vector<uint32_t>& graph) const` | 已建库时导出行优先的`N*G`个邻居ID；未建库返回`APP_ERR_INDEX_NOT_INIT`，失败不替换调用方的向量。 |
| `SaveGraphToFile(const std::string& filePath) const` | 已建库时将图写入指定文件，覆盖已有内容。文件为行优先的`N*G`个本机字节序uint32_t，无文件头；不包含向量或量化数据。路径不能为空或包含NUL，父目录需已存在。未建库返回`APP_ERR_INDEX_NOT_INIT`；打开、写入或关闭失败返回非0。写入失败时目标文件可能不完整，索引状态不变。 |
| `IsTrained() const` | 返回是否已完成建库及底库量化。 |
| `GetSize() const` | 返回已建库的底库数量；未建库或Reset后为0。 |

### AscendIndexCagraConfig

| 成员 | 默认值 | 约束与含义 |
| --- | --- | --- |
| `graphDegree` | 64 | 自建图输出度数G，支持64、128。 |
| `intermediateDegree` | 128 | 中间图度数M，满足`G<=M<=128`且`M<N`。 |
| `maxIterations` | 20 | NN-Descent最大迭代次数，大于0。 |
| `terminationThreshold` | 0.0001 | 有限浮点数，范围[0,1]。 |
| `guaranteeConnectivity` | true | 开启Host连通性补边。 |
| `verbose` | false | 打印构图阶段、迭代及收敛信息。 |

构图和查询均支持维度`{64,128,256,512,1024,1536,3072}`，仅接受有限FP32值。默认使用RaBitQ量化距离，不进行FP32重排。

Train和Add都是全量建库入口，选择其中一个完成建库。构图临时资源在底库装载前释放；查询算子按TopK缓存，首次使用某TopK时加载对应OM。需提前生成匹配N、D、G、TopK的七档分页OM；切换TopK不要求重新构图，但仍需该TopK的OM已存在。模型生成说明见[用户指南](../../05_user_guide.md)。

```cpp
faiss::ascend::AscendIndexCagra index;
faiss::ascend::AscendIndexCagraConfig config;
config.graphDegree = 64;
config.intermediateDegree = 128;
int ret = index.Init(128, {deviceId}, config);
if (ret != 0) return ret;
ret = index.Train(dataNum, baseData); // 也可调用Add(dataNum, baseData)
if (ret != 0) return ret;
std::vector<float> distances(static_cast<size_t>(queryNum) * topK);
std::vector<uint32_t> labels(static_cast<size_t>(queryNum) * topK);
ret = index.Search(queryNum, queryData, topK, distances.data(), labels.data());
if (ret != 0) return ret;
// 需要导出图时调用；导出不触发重新构图。
ret = index.SaveGraphToFile("cagra_graph.bin");
if (ret != 0) return ret;
```

以下为兼容外部图调用接口，与上述统一Init调用模式分开使用。

## AscendIndexCagra接口<a name="ZH-CN_TOPIC_0000002513317664"></a>

<a name="table_cagra_ctor"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexCagra();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexCagra的默认构造函数，创建一个Cagra检索Index实例。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table_cagra_delete"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexCagra(const AscendIndexCagra&amp;) = delete; AscendIndexCagra&amp; operator=(const AscendIndexCagra&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">禁用拷贝构造和拷贝赋值，AscendIndexCagra不可拷贝。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Init接口<a name="ZH-CN_TOPIC_0000002513317665"></a>

<a name="table_cagra_init"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int graphDegree, int dataNum, int topK, const std::vector&lt;int&gt;&amp; deviceList);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">初始化AscendIndexCagra，配置向量维度、图度数、底库数量、检索topK值及设备列表。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dim</code></strong>：特征向量维度。<br><strong><code>int graphDegree</code></strong>：近邻图的度数，即每个节点的邻居数量。<br><strong><code>int dataNum</code></strong>：底库中特征向量的数量。<br><strong><code>int topK</code></strong>：检索返回的最近邻数量。<br><strong><code>const std::vector&lt;int&gt;&amp; deviceList</code></strong>：Device侧设备ID列表。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：返回0表示成功，非0表示失败。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {64, 128, 256, 512, 1024, 1536, 3072}。<br>● graphDegree ∈ {64, 128}。<br>● graphDegree &lt; dataNum ≤ 1e9，实际可用规模受Host和Device内存限制。<br>● topK∈{32,64,128,256,512,1024}, topK≤dataNum。<br>● deviceList暂只支持单卡。<br>● 调用其他接口前必须先调用Init进行初始化。</td></tr>
</tbody></table>

## Add接口<a name="ZH-CN_TOPIC_0000002513317666"></a>

<a name="table_cagra_add"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向Index中添加近邻图和底库特征数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const uint32_t* graph</code></strong>：近邻图数据，大小为dataNum * graph_degree。<br><strong><code>const uint32_t* hash</code></strong>：兼容保留参数，必须非空；当前实现不读取其内容。访问位图由SDK内部分配，每次查询由kernel清零；不能通过本参数预置访问标记或过滤条件。<br><strong><code>const float* data</code></strong>：底库特征向量数据，大小为dataNum * dim。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：返回0表示成功，非0表示失败。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">调用Add前必须先调用Init。</td></tr>
</tbody></table>

### 显存需求

设N=dataNum、D=dim、G=graphDegree。主要持久设备缓冲估算为：FP32底库 `4ND` 字节、近邻图 `4NG` 字节、底库量化编码 `N×(6×ceil(D/8)+16)` 字节，以及visited位图 `64×ceil(N/32)×4` 字节。量化编码在统一接口Train/Add阶段或兼容接口QuantizeData阶段建立。

N=1000000时visited约8MB；N=1000000000时约8GB（7.45GiB）。Init的数值上限不代表设备可容纳该规模，调用方还须考虑查询缓冲、workspace和临时分配的峰值占用，按实际可用内存确定底库规模。

## QuantizeData接口<a name="ZH-CN_TOPIC_0000002513317667"></a>

<a name="table_cagra_quantize"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR QuantizeData(int n, const float* queryData, int ntotal, const float* baseData);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对查询向量和底库向量进行量化编码，包括随机正交变换、质心计算和比特量化。量化后的数据将用于后续检索。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：查询向量的数量。<br><strong><code>const float* queryData</code></strong>：查询特征向量数据，大小为n * dim。<br><strong><code>int ntotal</code></strong>：底库向量的数量。<br><strong><code>const float* baseData</code></strong>：底库特征向量数据，大小为ntotal * dim。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：返回0表示成功，非0表示失败。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 调用 QuantizeData 前必须先调用 Init 和 Add。<br>● n∈(0, 4096]。<br>● ntotal 必须为正数，且与 Init 的 dataNum 一致。<br>● 底库不变时，后续查询批次可调用 QuantizeQueries，复用底库量化结果。</td></tr>
</tbody></table>

## QuantizeQueries接口

<a name="table_cagra_quantize_queries"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR QuantizeQueries(int n, const float* queryData);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">复用已保存的底库质心、正交变换和量化编码，为当前查询批次生成量化码及预计算数据，替换上一批查询缓存。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：当前查询向量数量。<br><strong><code>const float* queryData</code></strong>：连续的查询特征数组，大小为 n * dim。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：返回 0 表示成功，非 0 表示失败。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 先成功调用 Init、Add 和 QuantizeData。<br>● n∈(0, 4096]，queryData 不能为空。<br>● 查询数据或顺序改变后，先调用本接口更新查询缓存，再调用 Search。<br>● 再次调用 Add 后，需重新调用 QuantizeData 完成底库与查询量化。</td></tr>
</tbody></table>

## Search接口<a name="ZH-CN_TOPIC_0000002513317668"></a>

<a name="table_cagra_search"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">执行近似最近邻检索，返回每个查询向量的topK个最近邻的距离和标签。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：查询向量的数量。<br><strong><code>const float* queryData</code></strong>：查询特征向量数据，大小为n * dim。<br><strong><code>int topK</code></strong>：检索返回的最近邻数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float* dists</code></strong>：检索结果距离值，大小为n * topK，按查询顺序排列。<br><strong><code>uint32_t* labels</code></strong>：检索结果标签，大小为n * topK，按查询顺序排列。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：返回0表示成功，非0表示失败。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 调用 Search 前必须先调用 Init、Add 和 QuantizeData。<br>● 使用最近一次 QuantizeData 或 QuantizeQueries 生成的查询缓存，queryData 的内容和顺序须与缓存对应。n 不得超过最近量化的查询数；检索子集时使用该批次的前 n 条查询。<br>● topK 应与 Init 时设置的 topK 一致。<br>● n∈(0, 4096]。<br>● topK∈{32,64,128,256,512,1024}, topK≤dataNum。</td></tr>
</tbody></table>

同一底库处理多个查询批次时，首次执行 `Init → Add → QuantizeData → Search`，
后续每批执行 `QuantizeQueries → Search`。完整查询耗时包含当前批次的 `QuantizeQueries` 和 `Search`；
底库量化在首次准备时完成。

检索 TopK 支持32、64、128、256、512、1024，且不得超过dataNum。内部候选容量为不小于max(64,TopK)的最小2的幂，容量64/128/256使用bitonic，512/1024使用radix。量化距离保持不变，不进行FP32重排。兼容接口的Init和Search保持固定TopK；统一接口允许Search按次指定TopK，均须有匹配top_k属性的OM。更新后须重新编译Host库、安装算子包并重新生成检索OM。
