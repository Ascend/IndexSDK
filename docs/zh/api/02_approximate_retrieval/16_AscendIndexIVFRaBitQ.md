# AscendIndexIVFRaBitQ<a name="ZH-CN_TOPIC_0000002513157720"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002544797635"></a>

AscendIndexIVFRaBitQ利用IVF进行加速，是二级近似检索算法。当前支持L2距离计算。

## AscendIndexIVFRaBitQ接口<a name="ZH-CN_TOPIC_0000002513317654"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFRaBitQConfig config)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFRaBitQ的构造函数，创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：底库检索向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：距离类型，支持faiss::METRIC_L2和faiss::METRIC_IP。<br><strong><code>int nlist</code></strong>：IVF分桶数。<br><strong><code>AscendIndexIVFRaBitQConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dims目前仅支持128。<br>● nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}。</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFRaBitQ&amp; operator=(const AscendIndexIVFRaBitQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFRaBitQ&amp;</code></strong>：常量AscendIndexIVFRaBitQ。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexIVFRaBitQ接口<a name="ZH-CN_TOPIC_0000002544837623"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>~AscendIndexIVFRaBitQ()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFRaBitQ的析构函数，销毁AscendIndexIVFRaBitQ对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## operate = 接口<a name="ZH-CN_TOPIC_0000002513157724"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFRaBitQ&amp; operator=(const AscendIndexIVFRaBitQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFRaBitQ&amp;</code></strong>：常量AscendIndexIVFRaBitQ。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## train接口<a name="ZH-CN_TOPIC_0000002544797639"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对AscendIndexIVFRaBitQ执行训练，继承AscendIndex中的相关接口并提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 训练采用k-means进行聚类，训练集比较小可能会影响查询精度。<br>● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“x”需要为非空指针，且长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “useKmeansPP”参数设置为“true”代表启用NPU聚类，否则采用CPU聚类。精度问题参考<a href="../../07_faq.md#浮点数计算精度问题">浮点数计算精度问题</a>。</td></tr>
</tbody></table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000002513157728"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void remove_ids(size_t n, const idx_t* ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对提供的索引序号对应在AscendIndexIVFRaBitQ中的已训练向量进行删除，调用AscendIndexIVFRaBitQImpl中的相关接口实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t n</code></strong>：准备删除的 集中特征向量的条数。<br><strong><code>const idx_t *ids</code></strong>：准备删除的特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“ids”需要为非空指针，且长度应该为<strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000002557609263"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFRaBitQ *index)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">提供一个cpu侧IndexIVFRaBitQ索引，从训练好的索引中加载数据到device侧供后续检索，调用AscendIndexIVFRaBitQImpl中的相关接口实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFRaBitQ *index</code></strong>：训练好的cpu侧IndexIVFRaBitQ索引。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处指针“index”需要为非空指针，且应为已训练好的IndexIVFRaBitQ索引。<br>● 调用此接口读取数据前应按照正常流程配置AscendIndexIVFRaBitQConfig并创建AscendIndexIVFRaBitQ对象。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000002557689209"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFRaBitQ *index) const</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">提供一个cpu侧IndexIVFRaBitQ索引，将device侧已训练好的数据下载到cpu索引中以持久化，调用AscendIndexIVFRaBitQImpl中的相关接口实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIVFRaBitQ *index</code></strong>：训练好的cpu侧IndexIVFRaBitQ索引。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处指针“index”需要为非空指针。<br>● 调用此接口持久化数据前应按照正常流程创建AscendIndexIVFRaBitQ对象并训练入库。</td></tr>
</tbody></table>

## search接口<a name="ZH-CN_TOPIC_IVFRABITQ_SEARCH"></a>

<a name="table_ivfrabitq_search"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexIVFRaBitQ特征向量查询。支持通过Faiss SearchParameters.sel传入IDSelector，在查询阶段按ID过滤（多卡场景下各卡对本地候选做同一全局ID语义过滤后合并）。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询特征向量条数。<br><strong><code>const float *x</code></strong>：查询特征向量，长度 n*dim。<br><strong><code>idx_t k</code></strong>：返回最相似结果个数。<br><strong><code>const SearchParameters *params</code></strong>：可选。params-&gt;sel 支持 IDSelectorRange / IDSelectorBatch / IDSelectorArray / IDSelectorBitmap / faiss::ascend::IDSelectorRoaring，以及上述类型的 IDSelectorNot。params 为 nullptr 或 sel 为 nullptr 时不做ID过滤。IDSelectorArray.ids、IDSelectorBitmap.bitmap 与 IDSelectorRoaring 的 frozen/live 指针必须在 search 返回前保持有效（Bitmap 与 Roaring frozen 为零拷贝）。Array/Bitmap/Roaring 缓存按 payload 指针和长度命中，不比对内容；原地改写同一 buffer 不会使缓存失效，换过滤集请换指针。IDSelectorBitmap 为稠密字节位图，不会自动转成 Roaring。IDSelectorRoaring 仅接受 CRoaring frozen 序列化（与 SDK 同版本 roaring_bitmap_frozen_serialize）或 live roaring_bitmap_t*（首次 miss 时 freeze）。传入 SearchParametersIVF 时，nprobe 仅对本次查询生效（须 &gt; 0 且 &lt;= nlist），不修改 index 上的 nprobe；基类 SearchParameters 只过滤时仍使用 index 上的 nprobe。Faiss SearchParametersIVF.nprobe 默认值为 1，只过滤时请继续用基类，或显式把 nprobe 设成 index.getNumProbes()。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：距离结果，长度 n*k。<br><strong><code>idx_t *labels</code></strong>：ID结果，长度 n*k；不足 k 个时剩余用 -1 填充。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● IDSelector过滤阶段可通过环境变量 IVFRABITQ_SELECTOR_FILTER_STAGE 配置，取值为 auto、pre、post、both；默认 auto，在支持L2前置过滤的普通kernel路径使用pre，在SIMT路径（Ascend 950）自动使用post。<br>● pre表示仅在L2距离阶段执行前置过滤；post表示仅在AICPU TopK阶段执行后置过滤；both表示前后两阶段均执行，主要用于对比验证。<br>● SIMT路径（Ascend 950）暂不支持L2前置过滤，设置为pre时会返回参数错误；如需在SIMT路径使用IDSelector，请设置为auto或post。<br>● IDSelectorRoaring 不走L2前置过滤，默认 auto 时在AICPU TopK阶段后置过滤。<br>● 多卡检索时同一 IDSelector 按全局ID语义应用到每张卡，再在Host侧合并 top-k。<br>● Array/Bitmap/Roaring 查询期缓存按 buffer 指针与长度命中，不扫描内容；原地修改 ids/bitmap/frozen 后仍传入同一指针会复用旧过滤结果。</td></tr>
</tbody></table>

## update接口<a name="ZH-CN_TOPIC_0000002566242121"></a>

<a name="table962730101715"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; update(idx_t n, float* x, idx_t* ids)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexIVFRaBitQ底库中ids对应的向量批量更新为x，对于不存在于底库的id不做更新处理，并返回不存在的id列表</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：准备更新的集中特征向量的条数。<br><strong><code>idx_t *x</code></strong>：准备更新的特征向量列表。<br><strong><code>idx_t *ids</code></strong>：准备更新的特征向量ID列表。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; noExistIds</code></strong>：返回不存在的向量ID列表。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“x”需要为非空指针，且长度应该为<strong><code>n</code></strong>，大小为<strong><code>n*dim，</code></strong>否则可能出现越界读写错误并引起程序崩溃。<br>● 此处指针“ids”需要为非空指针，且长度应该为<strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>
