# AscendIndexBinaryFlat<a name="ZH-CN_TOPIC_0000001506334701"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456694988"></a>

AscendIndexBinaryFlat类继承自Faiss的IndexBinary类，用于二值化特征检索。

仅支持<term>Atlas 推理系列产品</term>。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## add接口<a name="ZH-CN_TOPIC_0000001456854896"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add(idx_t n, const uint8_t *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向底库中添加特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const uint8_t *x</code></strong>：添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">指针“x”的长度应该为dims/8 * <strong><code>n</code></strong>，否则可能出现越界读写的错误或程序崩溃。<br>n &gt; 0，add操作需要保证最终底库大小ntotal取芯片内存实际容量与“1e9”之间的较小值。</td></tr>
</tbody></table>

> [!NOTE]
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。

## add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001506414809"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向底库中添加特征向量并指定对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const uint8_t *x</code></strong>：添加进底库的特征向量。<br><strong><code>const idx_t *xids</code></strong>：添加进底库的特征向量对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">0 &lt; n，add操作需要保证最终底库大小n取芯片内存实际容量与“1e9”之间的较小值。<br>指针“x”的长度应该为dims/8 * n，指针“xids”的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。用户需要根据自己的业务场景，保证xids的合法性，如底库中存在重复的ID，search结果中的label将无法对应具体的底库向量。</td></tr>
</tbody></table>

## AscendIndexBinaryFlat接口<a name="ZH-CN_TOPIC_0000001456535056"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(int dims, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexBinaryFlat的构造函数，生成维度为dims的AscendIndexBinaryFlat，根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexBinaryFlat管理的一组特征向量的维度。<br><strong><code>AscendIndexBinaryFlatConfig config</code></strong>：Device侧资源配置。<br><strong><code>bool usedFloat</code></strong>：用于入库为二进制、检索特征为float类型的检索方式（<a href="#ZH-CN_TOPIC_0000001456375288">search接口</a>）的性能提升，默认为“false”；设置为“true”时表示进行性能提升。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“dims”∈ { 256, 512, 1024 }</td></tr>
</tbody></table>

<a name="table191641015539"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(const faiss::IndexBinaryFlat *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexBinaryFlat的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexBinaryFlat *index</code></strong>：CPU侧index资源。<br><strong><code>AscendIndexBinaryFlatConfig config</code></strong>：Device侧资源配置。<br><strong><code>bool usedFloat</code></strong>：用于入库为二进制、检索特征为float类型的检索方式（<a href="#ZH-CN_TOPIC_0000001456375288">search接口</a>）的性能提升，默认为“false”；设置为“true”时表示进行性能提升。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU index指针，index-&gt;d ∈ {256, 512, 1024}，index-&gt;ntotal取芯片内存实际容量与“1e9”之间的较小值。</td></tr>
</tbody></table>

<a name="table142022518319"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(const faiss::IndexBinaryIDMap *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexBinaryFlat的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexBinaryIDMap *index</code></strong>：CPU侧index资源。<br><strong><code>AscendIndexBinaryFlatConfig config</code></strong>：Device侧资源配置。<br><strong><code>bool usedFloat</code></strong>：用于入库为二进制、检索特征为float类型的检索方式（<a href="#ZH-CN_TOPIC_0000001456375288">search接口</a>）的性能提升，默认为“false”；设置为“true”时表示进行性能提升。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的faiss::IndexBinaryIDMap指针，index-&gt;index为合法有效的IndexBinaryFlat指针，index-&gt;index-&gt;d ∈ {256, 512, 1024}，index-&gt;index-&gt;ntotal取芯片内存实际容量与“1e9”之间的较小值。</td></tr>
</tbody></table>

<a name="table145324411437"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(const AscendIndexBinaryFlat &amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明AscendIndexBinaryFlat拷贝构造函数为空，即AscendIndexBinaryFlat为不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexBinaryFlat &amp;</code></strong>：常量AscendIndexBinaryFlat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexBinaryFlat接口<a name="ZH-CN_TOPIC_0000001506495917"></a>

<a name="table13115573310"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexBinaryFlat() = default;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexBinaryFlat的析构函数，销毁AscendIndexBinaryFlat对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001506414941"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexBinaryFlat *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基于一个已有的Index拷贝到AscendIndexBinaryFlat，清空当前的AscendIndexBinaryFlat底库，并保持原有的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexBinaryFlat *index</code></strong>：faiss::IndexBinaryFlat指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的IndexBinaryFlat指针，index-&gt;d ∈ {256, 512, 1024}，index-&gt;ntotal取芯片内存实际容量与“1e9”之间的较小值。</td></tr>
</tbody></table>

<a name="table1570816514419"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexBinaryIDMap *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基于一个已有的“index”拷贝到AscendIndexBinaryFlat，清空当前的AscendIndexBinaryFlat底库，并保持原有的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexBinaryIDMap *index</code></strong>：faiss::IndexBinaryIDMap指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的faiss::IndexBinaryIDMap指针，index-&gt;index为合法有效的IndexBinaryFlat指针，index-&gt;index-&gt;d ∈ {256, 512, 1024}，index-&gt;index-&gt;ntotal取芯片内存实际容量与“1e9”之间的较小值。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001456855048"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexBinaryFlat *index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基于一个已有的AscendIndexBinaryFlat拷贝到faiss::IndexBinaryFlat index, index原有资源被清空。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexBinaryFlat *index</code></strong>：faiss::IndexBinaryFlat指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的IndexBinaryFlat指针，拷贝后的“index”资源由用户释放。</td></tr>
</tbody></table>

<a name="table19831553111512"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexBinaryIDMap *index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基于一个已有的AscendIndexBinaryFlat拷贝到faiss::IndexBinaryIDMap index, index原有资源被清空。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexBinaryIDMap *index</code></strong>：faiss::IndexBinaryIDMap指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的IndexBinaryIDMap指针，拷贝后的Index资源由用户释放。</td></tr>
</tbody></table>

## operator= 接口<a name="ZH-CN_TOPIC_0000001456535072"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlat &amp;operator = (const AscendIndexBinaryFlat &amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明AscendIndexBinaryFlat赋值构造函数为空，即AscendIndexBinaryFlat为不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexBinaryFlat &amp;</code></strong>：常量AscendIndexBinaryFlat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001506495769"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">删除底库中指定的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IDSelector &amp;sel</code></strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">返回成功删除（忽略非法ID）的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reset接口<a name="ZH-CN_TOPIC_0000001456855028"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">清空该AscendIndexBinaryFlat的底库向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## search接口<a id="ZH-CN_TOPIC_0000001456375288"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels, const SearchParameters *params) const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID和对应距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询向量个数。<br><strong><code>const uint8_t *x</code></strong>：查询向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const SearchParameters *params：</code></strong>Faiss的可选参数，默认为“nullptr”，暂不支持该参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int32_t *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：“k”个最近向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 查询的特征向量数据“x”的长度应该为dims/8 * <strong><code>n</code></strong>，“distances”以及“labels”的长度应该为<strong><code>k</code></strong> * <strong><code>n</code></strong>，否则可能会出现越界读写的情况，引起程序的崩溃。<br>● 0 &lt; n ≤ 1e9，0 &lt; k ≤1e5（n ≤ 1e9的限制远超过实际可用资源，请用户根据业务场景选择合适的查询向量个数）。</td></tr>
</tbody></table>

<a name="table1659211341612"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID和对应距离。用于入库特征为二进制特征，检索特征为float类型的检索方式。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询向量个数。<br><strong><code>const float *x</code></strong>：查询向量。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：“k”个最近向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 查询的特征向量数据“x”的长度应该为dims * n，“distances”以及“labels”的长度应该为k * n，否则可能会出现越界读写的情况，引起程序的崩溃。<br>● 0 &lt; n ≤ 1e9，0 &lt; k ≤1e5（n ≤ 1e9的限制远超过实际可用资源，请用户根据业务场景选择合适的查询向量个数）。</td></tr>
</tbody></table>

## setRemoveFast接口<a name="ZH-CN_TOPIC_0000002024780673"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>static void setRemoveFast(bool removeFast);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置是否快速删除底库中的向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>bool removeFast</code></strong>：设置为“true”表示使用快速删除；“false”表示不使用。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">快速删除会提高删除底库的性能，但是会稍微降低添加底库的性能。不调用该接口时默认不使用快速删除。该接口只能调用一次，且需要在构造index对象前使用。</td></tr>
</tbody></table>
