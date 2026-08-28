# AscendIndexIVFSQT<a name="ZH-CN_TOPIC_0000001456375224"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506615005"></a>

AscendIndexIVFSQT类，包含降维算法的三级检索IVFSQ算法，需要传入两个参数指明降维前后的维度信息，要求降维后维度能整除降维前的维度。适用于1000万级底库的场景。

需要按照IVFSQT算子生成方式，生成三级检索所需算子。

该类型带有模糊聚类功能：入桶前，使用threshold参数控制模糊程度。请根据底库容量和可用内存大小设置threshold参数值，过大的threshold会引起内存不足，导致失败。<term>Atlas 200/300/500 推理产品</term>环境建议设置\[1.0, 1.1\]，<term>Atlas 推理系列产品</term>环境建议设置\[1.0, 1.5\]。搜索时建议使用**batch size = 65536**。

使用流程为：1.构建index对象；2.train数据；3.add数据；4.update数据；5.search检索数据；6.析构index对象。update后不支持继续add数据。有新数据需要进行检索时，请将原来的index对象析构后，重新按照流程使用。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AscendIndexIVFSQT接口<a name="ZH-CN_TOPIC_0000001506495685"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQT的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>：CPU侧的Index资源。<br><strong><code>AscendIndexIVFSQTConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “index”需要为合法有效的CPU Index指针。<br>● index-&gt;d ∈ {256}。<br>● index-&gt;sq.d ∈ {32, 64, 128}。<br>● “index”的维度必须大于index-&gt;sq的维度且可以被index-&gt;sq的维度整除。</td></tr>
</tbody></table>

<a name="table124585216195"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQT(int dimIn, int dimOut, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQT的构造函数，生成AscendIndexIVFSQT，此时根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dimIn</code></strong>：AscendIndexIVFSQT管理的一组原始特征向量的维度。<br><strong><code>int dimOut</code></strong>：AscendIndexIVFSQT管理的一组降维目标特征向量的维度。<br><strong><code>int nlist</code></strong>：聚类中心的个数，与算子生成脚本中的“coarse_centroid_num”参数对应。<br><strong><code>faiss::ScalarQuantizer::QuantizerType qtype</code></strong>：AscendIndexIVFSQT的量化器类型。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。<br><strong><code>AscendIndexIVFSQTConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dimIn ∈ {256}。<br>● dimOut ∈ {32, 64, 128}。<br>● nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}。<br>● qtype = ScalarQuantizer::QuantizerType::QT_8bit，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”量化器类型。<br>● metric = faiss::MetricType::METRIC_INNER_PRODUCT （当前仅支持 “faiss::MetricType::METRIC_INNER_PRODUCT”。）</td></tr>
</tbody></table>

<a name="table68594118203"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQT(const AscendIndexIVFSQT&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFSQT&amp;</code></strong>：AscendIndexIVFSQT对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexIVFSQT接口<a name="ZH-CN_TOPIC_0000001456854984"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSQT();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQT的析构函数，销毁AscendIndexIVFSQT对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456695060"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVSQT基于一个已有的“index”拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针。<br>● index-&gt;d ∈ {256}。<br>● index-&gt;sq.d ∈ {32, 64, 128}。<br>● “index”的维度必须大于index-&gt;sq的维度，且可以被index-&gt;sq的维度整除。<br>● update过的对象请勿调用该接口。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001506495825"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexIVFSQT的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexIVFScalarQuantizer *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</td></tr>
</tbody></table>

## fineTune接口<a name="ZH-CN_TOPIC_0000001456694860"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void fineTune(size_t n, const float *x);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对中心进行微调和优化，避免分桶不均匀的问题。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t n</code></strong>：特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## getFuzzyK接口<a name="ZH-CN_TOPIC_0000001456855008"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getFuzzyK() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取入桶时每个向量的最大值。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>int</code></strong>：每个向量入桶时的最大值。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getListCodesAndIds接口<a name="ZH-CN_TOPIC_0000001687739112"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回当前的AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量和对应ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int listId</code></strong>：AscendIndexIVFSQT的nlist中的特定nlistId。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;uint8_t&gt;&amp; codes</code></strong>：AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量。<br><strong><code>std::vector&lt;ascend_idx_t&gt;&amp; ids</code></strong>：AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## getListLength接口<a name="ZH-CN_TOPIC_0000001735977797"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>uint32_t getListLength(int listId) const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回当前的AscendIndexIVFSQT的nlist中的特定nlistId上的长度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int listId</code></strong>：AscendIndexIVFSQT的nlist中的特定nlistId。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">AscendIndexIVFSQT的nlist中的特定nlistId上的长度。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## getLowerBound接口<a name="ZH-CN_TOPIC_0000001506614885"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getLowerBound() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回二级分簇的阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">二级分簇的阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getMergeThres接口<a name="ZH-CN_TOPIC_0000001506615073"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getMergeThres() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取合并子桶阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">合并子桶阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getQMax接口<a name="ZH-CN_TOPIC_0000001456535208"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>float getQMax() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回特征向量的最大值。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">特征向量的最大值。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getQMin接口<a name="ZH-CN_TOPIC_0000001506615029"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>float getQMin() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回特征向量的最小值。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">特征向量的最小值。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getThreshold接口<a name="ZH-CN_TOPIC_0000001506334633"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>float getThreshold() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取判断向量是否入多个桶的阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>float</code></strong>：判断向量是否入多个桶的阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506615085"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSQT&amp; operator=(const AscendIndexIVFSQT&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFSQT&amp;</code></strong>：AscendIndexIVFSQT对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001506615053"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据ID删除底库特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IDSelector &amp;sel</code></strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">返回被删除的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口。</td></tr>
</tbody></table>

## reset接口<a name="ZH-CN_TOPIC_0000001506334789"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">重置索引，特征数据清零。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">调用后请勿继续使用该对象。</td></tr>
</tbody></table>

## setAddTotal接口<a name="ZH-CN_TOPIC_0000001456375316"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setAddTotal(size_t addTotal);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置待添加的底库向量总数，默认值“100000000”。需要先设置“PreciseMemControl”为“true”。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t addTotal</code></strong>：待添加的底库向量总数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## setFuzzyK接口<a name="ZH-CN_TOPIC_0000001456534940"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setFuzzyK(int value);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置入桶时每个向量的最大值。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int value</code></strong>：每个向量入桶时的最大值，建议固定为默认值“3”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">value的取值范围是（0,10]。</td></tr>
</tbody></table>

## setLowerBound接口<a name="ZH-CN_TOPIC_0000001506334777"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setLowerBound(int lowerBound);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置二级分簇的阈值，默认值为“32”。<br>若一级分簇桶中元素大于lowerBound则进行二次分簇，否则保留原状。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int lowerBound</code></strong>：二级分簇的阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## setMemoryLimit接口<a name="ZH-CN_TOPIC_0000001506614917"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setMemoryLimit(float memoryLimit);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置Host内存限制，默认值为“32”，单位“GB”。需要先设置“PreciseMemControl”为“true”。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>float memoryLimit</code></strong>：内存限制。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## setMergeThres接口<a name="ZH-CN_TOPIC_0000001456694900"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setMergeThres(int mergeThres);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置合并子桶阈值，默认值为“5”。<br>若二级分簇后某子桶中元素小于mergeThres，则合并该子桶元素至其他子桶中。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int mergeThres</code></strong>：合并子桶阈值。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## setNumProbes接口<a name="ZH-CN_TOPIC_0000001736410013"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setNumProbes(int nprobes) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置当前的AscendIndexIVFSQT的nprobe数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int nprobes</code></strong>：AscendIndexIVFSQT的nprobe数。建议保持为默认值“64”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● nprobes ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }<br>● l2Probe ≥ nprobes, l2Probe≤ l3SegmentNum, l2Probe≤ nprobes * 64<br>● l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }<br>● l2Probe和l3SegmentNum的设置可参见<a href="#setsearchparams接口">setSearchParams</a>。<br>● setNumProbes接口预计2025年9月废除，请使用<a href="#setsearchparams接口">setSearchParams</a>。</td></tr>
</tbody></table>

## setPreciseMemControl接口<a name="ZH-CN_TOPIC_0000001506334681"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setPreciseMemControl(bool preciseMemControl);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">是否精确限制Host侧的内存大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>bool preciseMemControl</code></strong>：默认为“false”，表示停用对Host侧内存大小精确限制；为“true”时表示启用。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">当前版本暂不支持该接口，请勿调用。</td></tr>
</tbody></table>

## setSearchParams接口<a name="ZH-CN_TOPIC_0000002052679693"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置影响检索精度和性能的参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">int nprobe：AscendIndexIVFSQT的nprobe数。建议保持为默认值“64”。<br>int l2Probe：二级检索选择子桶的数量，默认值为“48”。<br>int l3SegmentNum：L3算子处理的段数，影响查找的base总数，默认值为“96”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● nprobe ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }<br>● l2Probe ≥ nprobe, l2Probe≤ l3SegmentNum, l2Probe≤ nprobe * 64<br>● l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }</td></tr>
</tbody></table>

## setSortMode接口<a name="ZH-CN_TOPIC_0000002165943965"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setSortMode(int mode);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置topk排序模式。模式0为近似排序；模式1为精确排序。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">int mode：topk排序模式。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 该接口需要在Search接口之前使用。<br>● “mode”支持模式0或模式1，默认为模式0。模式0：近似排序会截断部分topk结果，提升性能。<br>● 模式1：精确排序，会提升检索精度，牺牲部分性能。</td></tr>
</tbody></table>

## setThreshold接口<a name="ZH-CN_TOPIC_0000001456854808"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setThreshold(float value);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置判断向量是否入多个桶的阈值，默认值为“1.0”。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>float value</code></strong>：判断向量是否入多个桶的阈值，建议设置[1.0, 1.5]。由于Device侧内存存在限额，当使用内存达到限额后，会触发OOM机制，导致进程被杀死。用户可先查看Device侧的内存限额数据。（/sys/fs/cgroup/memory/usermemory/memory.limit_in_bytes），来评估添加底库的大小，若内存不充裕时，参数值建议在[1.0, 1.1]范围。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">value的取值范围是[0, fuzzyK- 1]，fuzzyK的取值请参见<a href="#getfuzzyk接口">getFuzzyK接口</a>。</td></tr>
</tbody></table>

## setUseCpuUpdate接口<a name="ZH-CN_TOPIC_0000002167379329"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>setUseCpuUpdate(int numThreads);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">是否使用CPU进行<a href="#update接口">update</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int numThreads</code></strong>：用于进行update的CPU核数，默认值为当前CPU的核数。<br>● 若当前CPU的核数&gt;96：当前CPU核数＜输入的numThreads，<strong><code>numThreads</code></strong> =96；<br>● 96＜输入的numThreads≤当前CPU核数，<strong><code>numThreads</code></strong>=96；<br>● 输入的numThreads≤96，numThreads为输入值。<br>若当前CPU的核数≤96：<br>● 当前CPU核数＜输入的numThreads ≤ 96，<strong><code>numThreads</code></strong>为当前CPU核数；<br>● 0＜输入的numThreads≤当前CPU核数，<strong><code>numThreads</code></strong>为输入值<strong><code>。</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>numThreads</code></strong>取值需大于0。<br>● 需要在使用<a href="#update接口">update</a>前配置。</td></tr>
</tbody></table>

## train接口<a name="ZH-CN_TOPIC_0000001456375352"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对AscendIndexIVFSQT执行训练，继承AscendIndexIVFSQ中的相关接口并提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 训练采用k-means进行聚类，训练集比较小可能会影响查询精度。<br>● 此处“n”的取值范围：nlist ≤ n ≤ 7,000,000。<br>● 此处指针“x”需要为非空指针，且长度应该为dimIn * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## update接口<a name="ZH-CN_TOPIC_0000001506414869"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void update(bool cleanData = true);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">三级检索的第二级，在add完毕全部的底库数据后，执行search前，用于训练子桶中心并根据子桶中心入桶。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">cleanData：是否清除中间数据，默认为“true”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">一次检索全流程中该接口只需要调用一次。</td></tr>
</tbody></table>

## updateTParams接口<a name="ZH-CN_TOPIC_0000001456854936"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void updateTParams(int l2Probe, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">测试时传入三级检索所需参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int l2Probe</code></strong>：二级检索选择子桶的数量，默认值为“48”。<br><strong><code>int l3SegmentNum</code></strong>：L3算子处理的段数，影响查找的base总数，默认值为“96”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● nprobe ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }<br>● l2Probe ≥ nprobe, l2Probe≤ l3SegmentNum, l2Probe≤ nprobe * 64<br>● l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }<br>● nprobe的设置可参见<a href="#setsearchparams接口">setSearchParams</a>。<br>● updateTParams接口预计2026年9月废除，请使用<a href="#setsearchparams接口">setSearchParams</a>。</td></tr>
</tbody></table>
