# AscendIndexCagra<a name="ZH-CN_TOPIC_0000002513157730"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002544797645"></a>

AscendIndexCagra是基于Cagra的图检索算法，通过构建近邻图实现高效近似最近邻搜索。

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
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {64, 128, 256, 512}。<br>● graphDegree ∈ {64, 128, 256, 512}。<br>● dataNum ∈(0, 1e9]。<br>● topK∈(0, 4096]。<br>● deviceList暂只支持单卡。<br>● 调用其他接口前必须先调用Init进行初始化。</td></tr>
</tbody></table>

## Add接口<a name="ZH-CN_TOPIC_0000002513317666"></a>

<a name="table_cagra_add"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向Index中添加近邻图、哈希表和底库特征数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const uint32_t* graph</code></strong>：近邻图数据，大小为dataNum * graph_degree。<br><strong><code>const uint32_t* hash</code></strong>：哈希表数据，用于检索过程中的访问标记，大小为dataNum * 2。<br><strong><code>const float* data</code></strong>：底库特征向量数据，大小为dataNum * dim。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：返回0表示成功，非0表示失败。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">调用Add前必须先调用Init。</td></tr>
</tbody></table>

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
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 调用 Search 前必须先调用 Init、Add 和 QuantizeData。<br>● 使用最近一次 QuantizeData 或 QuantizeQueries 生成的查询缓存，queryData 的内容和顺序须与缓存对应。n 不得超过最近量化的查询数；检索子集时使用该批次的前 n 条查询。<br>● topK 应与 Init 时设置的 topK 一致。<br>● n∈(0, 4096]。<br>● topK∈(0, 4096]。</td></tr>
</tbody></table>

同一底库处理多个查询批次时，首次执行 `Init → Add → QuantizeData → Search`，
后续每批执行 `QuantizeQueries → Search`。完整查询耗时包含当前批次的 `QuantizeQueries` 和 `Search`；
底库量化在首次准备时完成。
