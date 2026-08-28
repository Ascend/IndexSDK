# AscendIndexIVF<a name="ZH-CN_TOPIC_0000001456375220"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506334721"></a>

AscendIndexIVF作为特征检索组件中的采用IVF的Index的基类，为特征检索中的其他的IVF的Index定义接口。

对于IVF系列算法，在Atlas 300I Duo 推理卡上的线性增长取决于距离计算的运算量在整个search过程的占比。相较于其他计算类型，只有距离计算的运算量可以均分到多个运算单元，所以在大batch和nprobe较大的场景下，线性增长度更好，而小batch和nprobe较小的场景下线性增长度则较差。

> [!NOTE]
> IVF系列算法，应遵循nlist \* 2MB +  **resourceSize**  < NPU侧内存的规则，避免程序运行时申请内存失败，例如：npu卡上内存为64GB，则nlist应小于32768，32768 \* 2MB = 64GB，程序运行可能超出NPU内存大小。造成该限制的原因是目前检索业务申请内存的规则为大页内存优先，大页内存申请粒度为2MB。当nlist个桶内都有数据时，向硬件申请内存时，硬件分配的内存按照2MB的粒度对齐。（其中**resourceSize**是AscendIndexIVFConfig中用户指定的共享内存大小，默认128MB）

## AscendIndexIVF接口<a name="ZH-CN_TOPIC_0000001506414821"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVF(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFConfig config = AscendIndexIVFConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVF的构造函数，生成AscendIndexIVF，此时根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexIVF管理的一组特征向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型，当前支持“faiss::MetricType::METRIC_L2”以及“faiss::MetricType::METRIC_INNER_PRODUCT”。<br><strong><code>int nlist</code></strong>：聚类中心的个数，与算子生成脚本中的“coarse_centroid_num”参数对应。<br><strong><code>AscendIndexIVFConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}</td></tr>
</tbody></table>

<a name="table9624174810199"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVF(const AscendIndexIVF&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVF&amp;</code></strong>：常量AscendIndexIVF。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexIVF接口<a name="ZH-CN_TOPIC_0000001506334765"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexIVF();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVF的析构函数，销毁AscendIndexIVF对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001506334601"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVF* index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVF基于一个已有的“index”拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVF* index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “index”需要为合法有效的CPU Index指针。<br>● 该“index”的probe必须大于0且小于或等于nlist。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001506615113"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVF* index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexIVF的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexIVF* index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</td></tr>
</tbody></table>

## getNumLists接口<a name="ZH-CN_TOPIC_0000001506614893"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getNumLists() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回当前的AscendIndexIVF的nlist数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">AscendIndexIVF的nlist数。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getNumProbes接口<a name="ZH-CN_TOPIC_0000001456534948"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getNumProbes() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回当前的AscendIndexIVF的nprobe数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">AscendIndexIVF的nprobe数。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getListCodesAndIds接口<a name="ZH-CN_TOPIC_0000001456854940"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回当前的AscendIndexIVF的nlist中的特定nlistId上的特征向量和对应ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int listId</code></strong>：AscendIndexIVF的nlist中的特定nlistId。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;uint8_t&gt;&amp; codes</code></strong>：AscendIndexIVF的nlist中的特定nlistId上的特征向量。<br><strong><code>std::vector&lt;ascend_idx_t&gt;&amp; ids</code></strong>：AscendIndexIVF的nlist中的特定nlistId上的特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">0 ≤ listId＜nlist</td></tr>
</tbody></table>

## getListLength接口<a name="ZH-CN_TOPIC_0000001506614973"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual uint32_t getListLength(int listId) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回当前的AscendIndexIVF的nlist中的特定nlistId上的长度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int listId</code></strong>：AscendIndexIVF的nlist中的特定nlistId。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">AscendIndexIVF的nlist中的特定nlistId上的长度。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">0 ≤ listId＜nlist</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506495837"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVF&amp; operator=(const AscendIndexIVF&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVF&amp;</code></strong>：常量AscendIndexIVF。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reclaimMemory接口<a name="ZH-CN_TOPIC_0000001506615049"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t reclaimMemory() override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在保证底库数量不变的情况下，缩减底库占用的内存，继承AscendIndex中的接口并提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">缩减的内存大小，单位为Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reserveMemory接口<a name="ZH-CN_TOPIC_0000001506334617"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void reserveMemory(size_t numVecs) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在建立底库前为底库申请预留内存的抽象接口，继承AscendIndex中的接口并提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t numVecs</code></strong>：申请预留内存的底库数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">单卡环境时：0&lt;“numVecs”≤ “2e8”；多卡环境时：0 &lt; “numVecs”≤ “1e9”(“numVecs” ÷ 卡的数量需小于“2e8”)；超出限制会抛出异常，停止运行。</td></tr>
</tbody></table>

## reset接口<a name="ZH-CN_TOPIC_0000001506414685"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">清空该AscendIndexIVF的底库向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## setNumProbes接口<a name="ZH-CN_TOPIC_0000001506614937"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void setNumProbes(int nprobes);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置当前的AscendIndexIVF的nprobe数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int nprobes</code></strong>：AscendIndexIVF的nprobe数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">0 &lt; nprobes ≤ nlist</td></tr>
</tbody></table>
