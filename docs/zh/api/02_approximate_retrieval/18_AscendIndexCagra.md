# AscendIndexCagra<a name="ZH-CN_TOPIC_0000002513157730"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002544797645"></a>

AscendIndexCagra是基于Cagra的图检索算法，通过构建近邻图实现高效近似最近邻搜索。

## AscendIndexCagra接口<a name="ZH-CN_TOPIC_0000002513317664"></a>

<a name="table_cagra_ctor"></a>
<table><tbody><tr id="row_cagra_ctor_1"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p_cagra_ctor_1"><a name="p_cagra_ctor_1"></a><a name="p_cagra_ctor_1"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p_cagra_ctor_2"><a name="p_cagra_ctor_2"></a><a name="p_cagra_ctor_2"></a>AscendIndexCagra();</p>
</td>
</tr>
<tr id="row_cagra_ctor_3"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p_cagra_ctor_3"><a name="p_cagra_ctor_3"></a><a name="p_cagra_ctor_3"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p_cagra_ctor_4"><a name="p_cagra_ctor_4"></a><a name="p_cagra_ctor_4"></a>AscendIndexCagra的默认构造函数，创建一个Cagra检索Index实例。</p>
</td>
</tr>
<tr id="row_cagra_ctor_5"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p_cagra_ctor_5"><a name="p_cagra_ctor_5"></a><a name="p_cagra_ctor_5"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p_cagra_ctor_6"><a name="p_cagra_ctor_6"></a><a name="p_cagra_ctor_6"></a>无</p>
</td>
</tr>
<tr id="row_cagra_ctor_7"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p_cagra_ctor_7"><a name="p_cagra_ctor_7"></a><a name="p_cagra_ctor_7"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p_cagra_ctor_8"><a name="p_cagra_ctor_8"></a><a name="p_cagra_ctor_8"></a>无</p>
</td>
</tr>
<tr id="row_cagra_ctor_9"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p_cagra_ctor_9"><a name="p_cagra_ctor_9"></a><a name="p_cagra_ctor_9"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p_cagra_ctor_10"><a name="p_cagra_ctor_10"></a><a name="p_cagra_ctor_10"></a>无</p>
</td>
</tr>
<tr id="row_cagra_ctor_11"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p_cagra_ctor_11"><a name="p_cagra_ctor_11"></a><a name="p_cagra_ctor_11"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p_cagra_ctor_12"><a name="p_cagra_ctor_12"></a><a name="p_cagra_ctor_12"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table_cagra_delete"></a>
<table><tbody><tr id="row_cagra_delete_1"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p_cagra_delete_1"><a name="p_cagra_delete_1"></a><a name="p_cagra_delete_1"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p_cagra_delete_2"><a name="p_cagra_delete_2"></a><a name="p_cagra_delete_2"></a>AscendIndexCagra(const AscendIndexCagra&amp;) = delete;</p>
<p id="p_cagra_delete_2b"><a name="p_cagra_delete_2b"></a><a name="p_cagra_delete_2b"></a>AscendIndexCagra&amp; operator=(const AscendIndexCagra&amp;) = delete;</p>
</td>
</tr>
<tr id="row_cagra_delete_3"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p_cagra_delete_3"><a name="p_cagra_delete_3"></a><a name="p_cagra_delete_3"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p_cagra_delete_4"><a name="p_cagra_delete_4"></a><a name="p_cagra_delete_4"></a>禁用拷贝构造和拷贝赋值，AscendIndexCagra不可拷贝。</p>
</td>
</tr>
<tr id="row_cagra_delete_5"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p_cagra_delete_5"><a name="p_cagra_delete_5"></a><a name="p_cagra_delete_5"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p_cagra_delete_6"><a name="p_cagra_delete_6"></a><a name="p_cagra_delete_6"></a>无</p>
</td>
</tr>
<tr id="row_cagra_delete_7"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p_cagra_delete_7"><a name="p_cagra_delete_7"></a><a name="p_cagra_delete_7"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p_cagra_delete_8"><a name="p_cagra_delete_8"></a><a name="p_cagra_delete_8"></a>无</p>
</td>
</tr>
<tr id="row_cagra_delete_9"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p_cagra_delete_9"><a name="p_cagra_delete_9"></a><a name="p_cagra_delete_9"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p_cagra_delete_10"><a name="p_cagra_delete_10"></a><a name="p_cagra_delete_10"></a>无</p>
</td>
</tr>
<tr id="row_cagra_delete_11"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p_cagra_delete_11"><a name="p_cagra_delete_11"></a><a name="p_cagra_delete_11"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p_cagra_delete_12"><a name="p_cagra_delete_12"></a><a name="p_cagra_delete_12"></a>无</p>
</td>
</tr>
</tbody>
</table>

## Init接口<a name="ZH-CN_TOPIC_0000002513317665"></a>

<a name="table_cagra_init"></a>
<table><tbody><tr id="row_cagra_init_1"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p_cagra_init_1"><a name="p_cagra_init_1"></a><a name="p_cagra_init_1"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p_cagra_init_2"><a name="p_cagra_init_2"></a><a name="p_cagra_init_2"></a>APP_ERROR Init(int dim, int graphDegree, int dataNum, int topK, const std::vector&lt;int&gt;&amp; deviceList);</p>
</td>
</tr>
<tr id="row_cagra_init_3"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p_cagra_init_3"><a name="p_cagra_init_3"></a><a name="p_cagra_init_3"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p_cagra_init_4"><a name="p_cagra_init_4"></a><a name="p_cagra_init_4"></a>初始化AscendIndexCagra，配置向量维度、图度数、底库数量、检索topK值及设备列表。</p>
</td>
</tr>
<tr id="row_cagra_init_5"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p_cagra_init_5"><a name="p_cagra_init_5"></a><a name="p_cagra_init_5"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p_cagra_init_6"><a name="p_cagra_init_6"></a><a name="p_cagra_init_6"></a><strong id="b_cagra_init_1"><a name="b_cagra_init_1"></a><a name="b_cagra_init_1"></a>int dim</strong>：特征向量维度。</p>
<p id="p_cagra_init_7"><a name="p_cagra_init_7"></a><a name="p_cagra_init_7"></a><strong id="b_cagra_init_2"><a name="b_cagra_init_2"></a><a name="b_cagra_init_2"></a>int graphDegree</strong>：近邻图的度数，即每个节点的邻居数量。</p>
<p id="p_cagra_init_8"><a name="p_cagra_init_8"></a><a name="p_cagra_init_8"></a><strong id="b_cagra_init_3"><a name="b_cagra_init_3"></a><a name="b_cagra_init_3"></a>int dataNum</strong>：底库中特征向量的数量。</p>
<p id="p_cagra_init_9"><a name="p_cagra_init_9"></a><a name="p_cagra_init_9"></a><strong id="b_cagra_init_4"><a name="b_cagra_init_4"></a><a name="b_cagra_init_4"></a>int topK</strong>：检索返回的最近邻数量。</p>
<p id="p_cagra_init_10"><a name="p_cagra_init_10"></a><a name="p_cagra_init_10"></a><strong id="b_cagra_init_5"><a name="b_cagra_init_5"></a><a name="b_cagra_init_5"></a>const std::vector&lt;int&gt;&amp; deviceList</strong>：Device侧设备ID列表。</p>
</td>
</tr>
<tr id="row_cagra_init_7"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p_cagra_init_11"><a name="p_cagra_init_11"></a><a name="p_cagra_init_11"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p_cagra_init_12"><a name="p_cagra_init_12"></a><a name="p_cagra_init_12"></a>无</p>
</td>
</tr>
<tr id="row_cagra_init_9"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p_cagra_init_13"><a name="p_cagra_init_13"></a><a name="p_cagra_init_13"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p_cagra_init_14"><a name="p_cagra_init_14"></a><a name="p_cagra_init_14"></a><strong id="b_cagra_init_6"><a name="b_cagra_init_6"></a><a name="b_cagra_init_6"></a>APP_ERROR</strong>：返回0表示成功，非0表示失败。</p>
</td>
</tr>
<tr id="row_cagra_init_11"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p_cagra_init_15"><a name="p_cagra_init_15"></a><a name="p_cagra_init_15"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul_cagra_init_1"></a><a name="ul_cagra_init_1"></a><ul id="ul_cagra_init_1"><li>dim ∈ {64, 128, 256, 512}。</li><li>graphDegree ∈ {64, 128, 256, 512}。</li><li>dataNum ∈(0, 1e9]。</li><li>topK∈(0, 4096]。</li><li>deviceList暂只支持单卡。</li><li>调用其他接口前必须先调用Init进行初始化。</li></ul>
</td>
</tr>
</tbody>
</table>

## Add接口<a name="ZH-CN_TOPIC_0000002513317666"></a>

<a name="table_cagra_add"></a>
<table><tbody><tr id="row_cagra_add_1"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p_cagra_add_1"><a name="p_cagra_add_1"></a><a name="p_cagra_add_1"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p_cagra_add_2"><a name="p_cagra_add_2"></a><a name="p_cagra_add_2"></a>APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);</p>
</td>
</tr>
<tr id="row_cagra_add_3"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p_cagra_add_3"><a name="p_cagra_add_3"></a><a name="p_cagra_add_3"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p_cagra_add_4"><a name="p_cagra_add_4"></a><a name="p_cagra_add_4"></a>向Index中添加近邻图、哈希表和底库特征数据。</p>
</td>
</tr>
<tr id="row_cagra_add_5"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p_cagra_add_5"><a name="p_cagra_add_5"></a><a name="p_cagra_add_5"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p_cagra_add_6"><a name="p_cagra_add_6"></a><a name="p_cagra_add_6"></a><strong id="b_cagra_add_1"><a name="b_cagra_add_1"></a><a name="b_cagra_add_1"></a>const uint32_t* graph</strong>：近邻图数据，大小为dataNum * graph_degree。</p>
<p id="p_cagra_add_7"><a name="p_cagra_add_7"></a><a name="p_cagra_add_7"></a><strong id="b_cagra_add_2"><a name="b_cagra_add_2"></a><a name="b_cagra_add_2"></a>const uint32_t* hash</strong>：哈希表数据，用于检索过程中的访问标记，大小为dataNum * 2。</p>
<p id="p_cagra_add_8"><a name="p_cagra_add_8"></a><a name="p_cagra_add_8"></a><strong id="b_cagra_add_3"><a name="b_cagra_add_3"></a><a name="b_cagra_add_3"></a>const float* data</strong>：底库特征向量数据，大小为dataNum * dim。</p>
</td>
</tr>
<tr id="row_cagra_add_7"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p_cagra_add_9"><a name="p_cagra_add_9"></a><a name="p_cagra_add_9"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p_cagra_add_10"><a name="p_cagra_add_10"></a><a name="p_cagra_add_10"></a>无</p>
</td>
</tr>
<tr id="row_cagra_add_9"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p_cagra_add_11"><a name="p_cagra_add_11"></a><a name="p_cagra_add_11"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p_cagra_add_12"><a name="p_cagra_add_12"></a><a name="p_cagra_add_12"></a><strong id="b_cagra_add_4"><a name="b_cagra_add_4"></a><a name="b_cagra_add_4"></a>APP_ERROR</strong>：返回0表示成功，非0表示失败。</p>
</td>
</tr>
<tr id="row_cagra_add_11"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p_cagra_add_13"><a name="p_cagra_add_13"></a><a name="p_cagra_add_13"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p_cagra_add_14"><a name="p_cagra_add_14"></a><a name="p_cagra_add_14"></a>调用Add前必须先调用Init。</p>
</td>
</tr>
</tbody>
</table>

## QuantizeData接口<a name="ZH-CN_TOPIC_0000002513317667"></a>

<a name="table_cagra_quantize"></a>
<table><tbody><tr id="row_cagra_quantize_1"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p_cagra_quantize_1"><a name="p_cagra_quantize_1"></a><a name="p_cagra_quantize_1"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p_cagra_quantize_2"><a name="p_cagra_quantize_2"></a><a name="p_cagra_quantize_2"></a>APP_ERROR QuantizeData(int n, const float* queryData, int ntotal, const float* baseData);</p>
</td>
</tr>
<tr id="row_cagra_quantize_3"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p_cagra_quantize_3"><a name="p_cagra_quantize_3"></a><a name="p_cagra_quantize_3"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p_cagra_quantize_4"><a name="p_cagra_quantize_4"></a><a name="p_cagra_quantize_4"></a>对查询向量和底库向量进行量化编码，包括随机正交变换、质心计算和比特量化。量化后的数据将用于后续检索。</p>
</td>
</tr>
<tr id="row_cagra_quantize_5"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p_cagra_quantize_5"><a name="p_cagra_quantize_5"></a><a name="p_cagra_quantize_5"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p_cagra_quantize_6"><a name="p_cagra_quantize_6"></a><a name="p_cagra_quantize_6"></a><strong id="b_cagra_quantize_1"><a name="b_cagra_quantize_1"></a><a name="b_cagra_quantize_1"></a>int n</strong>：查询向量的数量。</p>
<p id="p_cagra_quantize_7"><a name="p_cagra_quantize_7"></a><a name="p_cagra_quantize_7"></a><strong id="b_cagra_quantize_2"><a name="b_cagra_quantize_2"></a><a name="b_cagra_quantize_2"></a>const float* queryData</strong>：查询特征向量数据，大小为n * dim。</p>
<p id="p_cagra_quantize_8"><a name="p_cagra_quantize_8"></a><a name="p_cagra_quantize_8"></a><strong id="b_cagra_quantize_3"><a name="b_cagra_quantize_3"></a><a name="b_cagra_quantize_3"></a>int ntotal</strong>：底库向量的数量。</p>
<p id="p_cagra_quantize_9"><a name="p_cagra_quantize_9"></a><a name="p_cagra_quantize_9"></a><strong id="b_cagra_quantize_4"><a name="b_cagra_quantize_4"></a><a name="b_cagra_quantize_4"></a>const float* baseData</strong>：底库特征向量数据，大小为ntotal * dim。</p>
</td>
</tr>
<tr id="row_cagra_quantize_7"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p_cagra_quantize_11"><a name="p_cagra_quantize_11"></a><a name="p_cagra_quantize_11"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p_cagra_quantize_12"><a name="p_cagra_quantize_12"></a><a name="p_cagra_quantize_12"></a>无</p>
</td>
</tr>
<tr id="row_cagra_quantize_9"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p_cagra_quantize_13"><a name="p_cagra_quantize_13"></a><a name="p_cagra_quantize_13"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p_cagra_quantize_14"><a name="p_cagra_quantize_14"></a><a name="p_cagra_quantize_14"></a><strong id="b_cagra_quantize_5"><a name="b_cagra_quantize_5"></a><a name="b_cagra_quantize_5"></a>APP_ERROR</strong>：返回0表示成功，非0表示失败。</p>
</td>
</tr>
<tr id="row_cagra_quantize_11"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p_cagra_quantize_15"><a name="p_cagra_quantize_15"></a><a name="p_cagra_quantize_15"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul_cagra_quantize_1"></a><a name="ul_cagra_quantize_1"></a><ul id="ul_cagra_quantize_1"><li>调用QuantizeData前必须先调用Init和Add。n∈(0, 4096]。</li></ul>
</td>
</tr>
</tbody>
</table>

## Search接口<a name="ZH-CN_TOPIC_0000002513317668"></a>

<a name="table_cagra_search"></a>
<table><tbody><tr id="row_cagra_search_1"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p_cagra_search_1"><a name="p_cagra_search_1"></a><a name="p_cagra_search_1"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p_cagra_search_2"><a name="p_cagra_search_2"></a><a name="p_cagra_search_2"></a>APP_ERROR Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels);</p>
</td>
</tr>
<tr id="row_cagra_search_3"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p_cagra_search_3"><a name="p_cagra_search_3"></a><a name="p_cagra_search_3"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p_cagra_search_4"><a name="p_cagra_search_4"></a><a name="p_cagra_search_4"></a>执行近似最近邻检索，返回每个查询向量的topK个最近邻的距离和标签。</p>
</td>
</tr>
<tr id="row_cagra_search_5"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p_cagra_search_5"><a name="p_cagra_search_5"></a><a name="p_cagra_search_5"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p_cagra_search_6"><a name="p_cagra_search_6"></a><a name="p_cagra_search_6"></a><strong id="b_cagra_search_1"><a name="b_cagra_search_1"></a><a name="b_cagra_search_1"></a>int n</strong>：查询向量的数量。</p>
<p id="p_cagra_search_7"><a name="p_cagra_search_7"></a><a name="p_cagra_search_7"></a><strong id="b_cagra_search_2"><a name="b_cagra_search_2"></a><a name="b_cagra_search_2"></a>const float* queryData</strong>：查询特征向量数据，大小为n * dim。</p>
<p id="p_cagra_search_8"><a name="p_cagra_search_8"></a><a name="p_cagra_search_8"></a><strong id="b_cagra_search_3"><a name="b_cagra_search_3"></a><a name="b_cagra_search_3"></a>int topK</strong>：检索返回的最近邻数量。</p>
</td>
</tr>
<tr id="row_cagra_search_7"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p_cagra_search_9"><a name="p_cagra_search_9"></a><a name="p_cagra_search_9"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p_cagra_search_10"><a name="p_cagra_search_10"></a><a name="p_cagra_search_10"></a><strong id="b_cagra_search_4"><a name="b_cagra_search_4"></a><a name="b_cagra_search_4"></a>float* dists</strong>：检索结果距离值，大小为n * topK，按查询顺序排列。</p>
<p id="p_cagra_search_11"><a name="p_cagra_search_11"></a><a name="p_cagra_search_11"></a><strong id="b_cagra_search_5"><a name="b_cagra_search_5"></a><a name="b_cagra_search_5"></a>uint32_t* labels</strong>：检索结果标签，大小为n * topK，按查询顺序排列。</p>
</td>
</tr>
<tr id="row_cagra_search_9"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p_cagra_search_13"><a name="p_cagra_search_13"></a><a name="p_cagra_search_13"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p_cagra_search_14"><a name="p_cagra_search_14"></a><a name="p_cagra_search_14"></a><strong id="b_cagra_search_6"><a name="b_cagra_search_6"></a><a name="b_cagra_search_6"></a>APP_ERROR</strong>：返回0表示成功，非0表示失败。</p>
</td>
</tr>
<tr id="row_cagra_search_11"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p_cagra_search_15"><a name="p_cagra_search_15"></a><a name="p_cagra_search_15"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul_cagra_search_1"></a><a name="ul_cagra_search_1"></a><ul id="ul_cagra_search_1"><li>调用Search前必须先调用Init、Add和QuantizeData。</li><li>topK应与Init时设置的topK一致。</li><li>n∈(0, 4096]。</li><li>topK∈(0, 4096]。</li></ul>
</td>
</tr>
</tbody>
</table>
