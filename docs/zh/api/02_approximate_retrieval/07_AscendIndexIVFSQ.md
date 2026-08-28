# AscendIndexIVFSQ<a name="ZH-CN_TOPIC_0000001506334625"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456694964"></a>

AscendIndexIVFSQ利用IVF来进行加速，是二级近似检索算法。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AscendIndexIVFSQ接口<a name="ZH-CN_TOPIC_0000001506414893"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQ的构造函数，基于一个已有的index创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>：CPU侧资源配置。<br><strong><code>AscendIndexIVFSQConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针。</td></tr>
</tbody></table>

<a name="table1823217151014"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQ(int dims, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQ的构造函数，生成AscendIndexIVFSQ，此时根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexIVFSQ管理的一组特征向量的维度。<br><strong><code>int nlist</code></strong>：聚类中心的个数，与算子生成脚本中的“coarse_centroid_num”参数对应。<br><strong><code>faiss::ScalarQuantizer::QuantizerType qtype</code></strong>：AscendIndexIVFSQ的量化器类型。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。<br><strong><code>bool encodeResidual</code></strong>：表示是否对残差编码。<br><strong><code>AscendIndexIVFSQConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dims ∈ {64, 128, 256, 384, 512}<br>● nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}<br>● qtype = ScalarQuantizer::QuantizerType::QT_8bit，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。<br>● metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</td></tr>
</tbody></table>

> [!NOTE]
>
>- 当前“encodeResidual”在“metric=faiss::MetricType::METRIC_INNER_PRODUCT”下，仅支持“false”取值，即当前并不支持对残差编码的IVFSQ方法，当取值为“true”时能够运行成功但存在精度问题。

<a name="table134501935171012"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQ的构造函数，生成AscendIndexIVFSQ，此时根据“config”中配置的值设置Device侧资源。此接口不执行初始化，由子类执行初始化相关功能，后续会废弃此接口，请勿使用。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexIVFSQ管理的一组特征向量的维度。<br><strong><code>int nlist</code></strong>：聚类中心的个数，与算子生成脚本中的“coarse_centroid_num”参数对应。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。<br><strong><code>AscendIndexIVFSQConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dims ∈ {64, 128, 256, 384, 512}<br>● nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}<br>● metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQ(const AscendIndexIVFSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFSQ&amp;</code></strong>：常量AscendIndexIVFSQ。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexIVFSQ接口<a name="ZH-CN_TOPIC_0000001456534936"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSQ();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQ的析构函数，销毁AscendIndexIVFSQ对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456375244"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQ基于一个已有的index拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>：CPU侧index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index的维度d参数取值范围为{64, 128, 256, 384, 512}，<br>Index的聚类中心的个数nlist参数取值范围{1024, 2048, 4096, 8192, 16384, 32768}<br>总的候选桶数量nprobe的取值范围0 &lt; nprobe ≤ nlist<br>底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。<br>sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001506334649"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFScalarQuantizer *index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexIVFSQ的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexIVFScalarQuantizer *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001456854860"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQ&amp; operator=(const AscendIndexIVFSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFSQ&amp;</code></strong>：常量AscendIndexIVFSQ。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## train接口<a name="ZH-CN_TOPIC_0000001456854976"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对AscendIndexIVFSQ执行训练，继承AscendIndex中的相关接口并提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 训练采用k-means进行聚类，训练集比较小可能会影响查询精度。<br>● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“x”需要为非空指针，且长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>
