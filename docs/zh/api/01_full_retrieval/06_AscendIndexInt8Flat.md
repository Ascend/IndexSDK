# AscendIndexInt8Flat<a name="ZH-CN_TOPIC_0000001506334741"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506615033"></a>

AscendIndexInt8Flat存储INT8类型特征向量并执行暴力搜索。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AscendIndexInt8Flat接口<a name="ZH-CN_TOPIC_0000001456375168"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Flat的构造函数，生成维度为dims的AscendIndexInt8（单个Index管理的一组向量的维度是唯一的），此时根据config中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexInt8管理的一组特征向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。<br><strong><code>AscendIndexInt8FlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dims ∈ {64, 128, 256, 384, 512, 768, 1024}。<br>● metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

<a name="table08035919302"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Flat的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexScalarQuantizer *index</code></strong>：CPU侧Index资源。<br><strong><code>AscendIndexInt8FlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，必须为<strong><code>AscendIndexInt8Flat</code></strong>执行<strong><code>copyTo</code></strong>接口生成的faiss::IndexScalarQuantizer类型指针。</td></tr>
</tbody></table>

<a name="table11312020103012"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Flat(const faiss::IndexIDMap *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Flat的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIDMap *index</code></strong>：CPU侧Index资源。<br><strong><code>AscendIndexInt8FlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，必须为<strong><code>AscendIndexInt8Flat</code></strong>执行<strong><code>copyTo</code></strong>接口生成的faiss::IndexIDMap类型指针。</td></tr>
</tbody></table>

<a name="table186285584308"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Flat(const AscendIndexInt8Flat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexInt8Flat&amp;</code></strong>：常量AscendIndexInt8Flat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table206471151315"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexInt8Flat();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Flat的析构函数，销毁AscendIndexInt8Flat对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456375340"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIDMap* index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Flat基于一个已有的“index”拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIDMap *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的IndexIDMap指针，该Index的成员索引维度d参数取值范围为{64, 128, 256, 384, 512, 768, 1024}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

<a name="table862731073217"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexScalarQuantizer* index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Flat基于一个已有的“index”拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexScalarQuantizer* index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index的维度d参数取值范围为{64, 128, 256, 384, 512, 768, 1024}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001506334805"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexScalarQuantizer* index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexInt8Flat的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexScalarQuantizer* index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</td></tr>
</tbody></table>

<a name="table1981952413329"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIDMap* index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexInt8Flat的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexIDMap *index</code></strong>：CPU侧index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的IndexIDMap指针，index占用的资源由用户释放内存。</td></tr>
</tbody></table>

## getBase接口<a name="ZH-CN_TOPIC_0000001506334753"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void getBase(int deviceId, std::vector&lt;int8_t&gt; &amp;xb) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该AscendIndexInt8Flat在特定“deviceId”上管理的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int deviceId</code></strong>：Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;int8_t&gt; &amp;xb</code></strong>：AscendIndexInt8Flat在“deviceId”上存储的底库特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“deviceId”需要为合法的设备ID。</td></tr>
</tbody></table>

## getBaseSize接口<a name="ZH-CN_TOPIC_0000001506414709"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t getBaseSize(int deviceId) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该AscendIndexInt8Flat在特定“deviceId”上管理的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int deviceId</code></strong>：Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">在特定“deviceId”上的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“deviceId”需要为合法的设备ID。</td></tr>
</tbody></table>

## getIdxMap接口<a name="ZH-CN_TOPIC_0000001506495853"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该AscendIndexInt8Flat在特定“deviceId”上管理的特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int deviceId</code></strong>：Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; &amp;idxMap</code></strong>：AscendIndexInt8Flat在“deviceId”上存储的底库特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“deviceId”需要为合法的设备ID。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506414909"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Flat&amp; operator=(const AscendIndexInt8Flat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexInt8Flat&amp;</code></strong>：常量AscendIndexInt8Flat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reset接口<a name="ZH-CN_TOPIC_0000001506495889"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void reset();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">清空该AscendIndexInt8Flat的底库向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001456694912"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexInt8特征向量查询接口，根据输入的特征向量以及“mask”掩码返回最相似的“k”条特征的距离及ID。mask为<strong><code>0</code></strong>、<strong><code>1</code></strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong><code>1</code></strong>参与，<strong><code>0</code></strong>不参与。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const int8_t* x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const void* mask</code></strong>：底库的过滤掩码。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “n”的取值范围：0 &lt; n &lt; 1e9。<br>● “k”通常不允许超过4096。<br>● 指针“x”需要为非空指针，且长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 指针“distances”/“labels”需要为非空指针，且长度应该为<strong><code>k</code></strong> * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 指针“mask”需要为非空指针，需保证传入的掩码长度为⌈ntotal / 8⌉ * n（“ntotal”为底库向量的条数）。<br>● mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。<br>● 使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</td></tr>
</tbody></table>

## setPageSize接口<a name="ZH-CN_TOPIC_0000002007453769"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setPageSize(uint16_t pageBlockNum);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置该AscendIndexInt8Flat在search时一次性连续计算底库的block数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint16_t pageBlockNum</code></strong>：一次性连续计算底库的block数量。不设置时，默认一次性连续计算16个block。一个block存储向量的大小由AscendIndexInt8FlatConfig中的blockSize决定。该值越大，search时占用的内存越大。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “pageBlockNum”的取值范围：0 &lt; pageBlockNum ≤ 144<br>● 该接口主要用于大底库场景，search接口性能调优使用。该值越大，占用AscendIndexInt8FlatConfig中配置的resourceSize预置内存越大。建议申请足够大的预置内存，再利用该接口进行参数调优。</td></tr>
</tbody></table>
