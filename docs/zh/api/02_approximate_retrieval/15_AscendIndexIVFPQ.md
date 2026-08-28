# AscendIndexIVFPQ<a name="ZH-CN_TOPIC_0000002478095516"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002510095475"></a>

AscendIndexIVFPQ利用IVF进行加速，是二级近似检索算法。当前仅支持L2距离，且出于性能考量，仅支持检索320以内的topk。

## AscendIndexIVFPQ接口<a name="ZH-CN_TOPIC_0000002509975505"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFPQ(int dims, faiss::MetricType metric, int nlist, int msubs, int nbits, AscendIndexIVFPQConfig config)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFPQ的构造函数，创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：底库检索向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：距离类型，当前只支持faiss::METRIC_L2。<br><strong><code>int nlist</code></strong>：IVF分桶数。<br><strong><code>int msubs</code></strong>：划分子空间数。<br><strong><code>int nbits</code></strong>：PQ编码的长度比特数，例如nbits=8，PQ编码的序号为0~255。<br><strong><code>AscendIndexIVFPQConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dims目前仅支持128。<br>● nlist ∈ {1024, 2048, 4096, 8192, 16384, 262144, 524288}。<br>● msubs ∈ {2, 4, 8, 16, 32}。<br>● nbits目前支持8。<br>● config.useKmeansPP代表是否启用NPU聚类：设置为true时使用NPU K-Means训练粗聚类中心，设置为false时使用CPU聚类。大nlist场景建议resourceSize≥512MB（nlist=262144）或≥1GB（nlist=524288），训练样本量建议≥nlist×40。</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFPQ&amp; operator=(const AscendIndexIVFPQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFPQ&amp;</code></strong>：常量AscendIndexIVFPQ。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexIVFPQ接口<a name="ZH-CN_TOPIC_0000002477935546"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>~AscendIndexIVFPQ()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFPQ的析构函数，销毁AscendIndexIVFPQ对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## operate = 接口<a name="ZH-CN_TOPIC_0000002484264062"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFPQ&amp; operator=(const AscendIndexIVFPQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFPQ&amp;</code></strong>：常量AscendIndexIVFPQ。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## train接口<a name="ZH-CN_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对AscendIndexIVFPQ执行训练，继承AscendIndex中的相关接口并提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 训练采用k-means进行聚类，训练集比较小可能会影响查询精度。<br>● 此处“n”的取值范围：0 &lt; n &lt; 1e9；且应满足n ≥ nlist，否则粗聚类无法初始化。<br>● 此处指针“x”需要为非空指针，且长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “useKmeansPP”参数设置为“true”代表启用NPU聚类，否则采用CPU聚类；nlist=262144/524288时均可使用NPU粗聚类。<br>● 当nlist &gt; 16384时，训练采样默认按min(nlist×40, n, 10M)截断，以控制Host/Device内存占用。</td></tr>
</tbody></table>

## remove_ids接口<a name="ZH-CN_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void remove_ids(size_t n, const idx_t *ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对提供的索引序号对应在AscendIndexIVFPQ中的已训练向量进行删除，调用AscendIndexIVFPQImpl中的相关接口实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t n</code></strong>：删除集中特征向量的条数。<br><strong><code>const idx_t *ids</code></strong>：准备删除的特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“ids”需要为非空指针，且长度应该为<strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFPQ *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">从IndexIVFPQ的index索引中读取已训练的数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFPQ *index</code></strong>：IVFPQ索引，faiss库中的一种索引类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 调用前确保“index”中数据已有训练完成的聚类中心和倒排列表且参数完整。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(const faiss::IndexIVFPQ *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将已训练的数据保存到IndexIVFPQ的index索引中。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFPQ *index</code></strong>：IVFPQ索引，faiss库中的一种索引类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 调用前确保原始向量已完成训练和入库，避免接口读取空聚类中心、码本和倒排列表到“index”时产生错误。</td></tr>
</tbody></table>

## update接口<a name="ZH-CN_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; update(idx_t n, const float *x, idx_t *ids)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexIVFPQ底库中ids对应的向量批量更新为x，对于不存在于底库的id不做更新处理，并返回不存在的id列表。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：准备更新的集中特征向量的条数。<br><strong><code>float *x</code></strong>：准备更新的特征向量列表。<br><strong><code>idx_t *ids</code></strong>：准备更新的特征向量ID列表。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; noExistIds</code></strong>：返回不存在的向量ID列表。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“x”需要为非空指针，且长度应该为n，大小为<strong><code>dims * n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 此处指针“ids”需要为非空指针，且长度应该为<strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>
