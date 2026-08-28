# 多Index批量检索<a name="ZH-CN_TOPIC_0000001456535132"></a>

在检索距离（distances）值相同的情况下，相较于使用单Index检索功能，多Index批量检索使用的TopK排序算法不同，最终呈现的结果标签会存在一些差异，导致返回的TopK值的标签（label）存在差异。

## Search（AscendIndex）接口<a name="ZH-CN_TOPIC_0000001456854904"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void Search(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k,float *distances, idx_t *labels, bool merged);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现从多个AscendIndex库执行特征向量查询的接口，根据输入的特征向量返回最相似的“k”条特征的距离及ID。<br>当前支持以下算法：<br>● 由Index派生而来的子类型AscendIndexSQ（QuantizerType为QT_8bit）。<br>● 由Index派生而来的子类型AscendIndexFlat（FlatIP、FlatL2）。<br>● 由Index派生而来的子类型AscendIndexIVFSP。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;AscendIndex *&gt; indexes</code></strong>：待执行检索的多个index。<br><strong><code>idx_t n</code></strong>：执行检索的query数。<br><strong><code>const float *x</code></strong>：执行检索的query特征向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>bool merged</code></strong>：是否要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 参与检索的Index必须创建在同一张卡上。<br>● 当前“indexes”支持类型参见如下。“indexes”为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。<br>● “indexes”为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。<br>● “indexes”为AscendIndexFlat的指针并且需满足0 &lt; indexes.size() ≤ 10000。<br>此处“n”不超过1024。此处“k”不超过1024。此处“x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。“distances”/“labels”需要为非空指针，且满足：<br>● 当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## Search（AscendIndexInt8）接口<a name="ZH-CN_TOPIC_0000001533044201"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void Search(std::vector&lt;AscendIndexInt8 *&gt; indexes, idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, bool merged);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现从多个AscendIndexInt8库执行特征向量查询的接口，根据输入的特征向量返回最相似的“k”条特征的距离及ID。<br>当前仅支持由AscendIndexInt8派生而来的子类型AscendIndexInt8Flat。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;AscendIndexInt8 *&gt; indexes</code></strong>：待执行检索的多个index。<br><strong><code>idx_t n</code></strong>：执行检索的query数。<br><strong><code>const int8_t *x</code></strong>：执行检索的query特征向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>bool merged</code></strong>：是否要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 参与检索的Index必须创建在同一张卡上。<br>● 当前“indexes”仅支持类型为AscendIndexInt8并且需满足<strong><code>0 &lt; indexes.size() ≤ 10000</code></strong>。<br>● 此处“n”不超过1024。<br>● 此处“k”不超过1024。<br>● 此处“x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”/“labels”需要为非空指针，且满足：当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## Search（FaissIndex）接口<a name="ZH-CN_TOPIC_0000001506334841"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void Search(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k,float *distances, idx_t *labels, bool merged);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现从多个Index库执行特征向量查询的接口，根据输入的特征向量返回最相似的“k”条特征的距离及ID。<br>当前支持以下算法：<br>● 由Index派生而来的子类型AscendIndexSQ（QuantizerType为QT_8bit）。<br>● 由Index派生而来的子类型AscendIndexFlat（FlatIP、FlatL2）。<br>● 由Index派生而来的子类型AscendIndexIVFSP。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;Index *&gt; indexes</code></strong>：待执行检索的多个index。<br><strong><code>idx_t n</code></strong>：执行检索的query数。<br><strong><code>const float *x</code></strong>：执行检索的query特征向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>bool merged</code></strong>：是否要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 参与检索的Index必须创建在同一张卡上。<br>● 当前“indexes”支持类型参见如下。“indexes”为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。<br>● “indexes”为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。<br>● “indexes”为AscendIndexFlat的指针并且需满足0 &lt; indexes.size() ≤ 10000。<br>此处“n”不超过1024。此处“k”不超过1024。此处“x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。“distances”/“labels”需要为非空指针，且满足：<br>● 当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchWithFilter（FaissIndex单filter）接口<a name="ZH-CN_TOPIC_0000001521615937"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void SearchWithFilter(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters, bool merged);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">多“indexes”执行检索，根据输入的特征向量返回最相似的<strong><code>k</code></strong>条特征的ID。提供基于CID过滤的功能，“filters”为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;Index *&gt; indexes</code></strong>：待执行检索的多个index。<br><strong><code>idx_t n</code></strong>：执行检索的query数。<br><strong><code>const float *x</code></strong>：执行检索的query特征向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const void *filters</code></strong>：过滤条件。<br><strong><code>bool merged</code></strong>：是否要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 参与检索的Index必须创建在同一张卡上。<br>● 当前“indexes”支持类型参见如下。“indexes”为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足<strong><code>0 &lt; indexes.size() ≤ 10000</code></strong>。<br>● “indexes”为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足<strong><code>0 &lt; indexes.size() ≤ 10000</code></strong>。<br>此处“n”不超过1024。此处“k”不超过1024。此处“x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。“distances”/“labels”需要为非空指针，且满足：<br>● 当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。<br>“filters”需要为长度为n * 6的uint32_t的数组，否则可能出现越界读取的错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchWithFilter（AscendIndex单filter）接口<a name="ZH-CN_TOPIC_0000001521894949"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void SearchWithFilter(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters, bool merged);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">多“indexes”执行检索，根据输入的特征向量返回最相似的<strong><code>k</code></strong>条特征的ID。提供基于CID过滤的功能，“filters”为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;AscendIndex *&gt; indexes</code></strong>：待执行检索的多个index。<br><strong><code>idx_t n</code></strong>：执行检索的query数。<br><strong><code>const float *x</code></strong>：执行检索的query特征向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const void *filters</code></strong>：过滤条件。<br><strong><code>bool merged</code></strong>：是否要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 参与检索的Index必须创建在同一张卡上。<br>● 当前“indexes”支持类型参见如下。“indexes”为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足<strong><code>0 &lt; indexes.size() ≤ 10000</code></strong>。<br>● “indexes”为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足<strong><code>0 &lt; indexes.size() ≤ 10000</code></strong>。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。<br>● 此处“n”不超过1024。<br>● 此处“k”不超过1024。<br>● 此处“x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”/“labels”需要为非空指针，且满足：当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。<br>“filters”需要为长度为n * 6的uint32_t的数组，否则可能出现越界读取的错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchWithFilter（FaissIndex多filter）接口<a name="ZH-CN_TOPIC_0000001635576093"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void SearchWithFilter(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, void *filters[], bool merged);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">多“indexes”执行检索，根据输入的特征向量返回最相似的“k”条特征的ID。<br>提供基于CID过滤的功能，“filters”为大小为“n”的指针数组，“filters”数组中的每个指针指向indexes.size() * 6 个uint32_t的数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;Index *&gt; indexes</code></strong>：待执行检索的多个index。<br><strong><code>idx_t n</code></strong>：执行检索的query数。<br><strong><code>const float *x</code></strong>：执行检索的query特征向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>void *filters[]</code></strong>：过滤条件。<br><strong><code>bool merged</code></strong>：是否要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 参与检索的Index必须创建在同一张卡上。<br>● 当前“indexes”支持类型参见如下。“indexes”为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。<br>● “indexes”为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。<br>● “n”不超过1024。<br>● “k”不超过1024。<br>● “x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”/“labels”需要为非空指针，且满足：当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。<br>“filters”需要为长度为n的指针数组，且数组中每个指针指向长度为indexes.size() * 6的uint32_t的数组，否则可能出现越界读取的错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchWithFilter（AscendIndex多filter）接口<a name="ZH-CN_TOPIC_0000001635815493"></a>

<a name="table20177631161415"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void SearchWithFilter(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, void *filters[], bool merged);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">多“indexes”执行检索，根据输入的特征向量返回最相似的“k”条特征的ID。<br>提供基于CID过滤的功能，“filters”为大小为“n”的指针数组，“filters”数组中的每个指针指向indexes.size() * 6 个uint32_t的数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;Index *&gt; indexes</code></strong>：待执行检索的多个index。<br><strong><code>idx_t n</code></strong>：执行检索的query数。<br><strong><code>const float *x</code></strong>：执行检索的query特征向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>void *filters[]</code></strong>：过滤条件。<br><strong><code>bool merged</code></strong>：是否要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 参与检索的Index必须创建在同一张卡上。<br>● 当前“indexes”支持类型参见如下。“indexes”为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。<br>● “indexes”为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。<br>● “n”不超过1024。<br>● “k”不超过1024。<br>● “x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”/“labels”需要为非空指针，且满足：当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。<br>“filters”需要为长度为n的指针数组，且数组中每个指针指向长度为indexes.size() * 6的uint32_t的数组，否则可能出现越界读取的错误并引起程序崩溃。</td></tr>
</tbody></table>
