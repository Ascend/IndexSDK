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
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 调用QuantizeData前必须先调用Init和Add。n∈(0, 4096]。</td></tr>
</tbody></table>

## Search接口<a name="ZH-CN_TOPIC_0000002513317668"></a>

<a name="table_cagra_search"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">执行近似最近邻检索，返回每个查询向量的topK个最近邻的距离和标签。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：查询向量的数量。<br><strong><code>const float* queryData</code></strong>：查询特征向量数据，大小为n * dim。<br><strong><code>int topK</code></strong>：检索返回的最近邻数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float* dists</code></strong>：检索结果距离值，大小为n * topK，按查询顺序排列。<br><strong><code>uint32_t* labels</code></strong>：检索结果标签，大小为n * topK，按查询顺序排列。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：返回0表示成功，非0表示失败。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 调用Search前必须先调用Init、Add和QuantizeData。<br>● topK应与Init时设置的topK一致。<br>● n∈(0, 4096]。<br>● topK∈(0, 4096]。</td></tr>
</tbody></table>
