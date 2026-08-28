# AscendIndex<a id="ZH-CN_TOPIC_0000001456375304"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506414937"></a>

AscendIndex作为特征检索组件中的大部分检索的Index的基类，向上承接Faiss，向下为特征检索中的其他Index定义接口。

## add接口<a id="ZH-CN_TOPIC_0000001506614985"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndex建库和向底库中添加新的特征向量的功能。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：待添加进底库的特征向量数量。<br><strong><code>const float *x</code></strong>：待添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">指针“x”的长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>“n”的取值范围：0 &lt; n &lt; 1e9。</td></tr>
</tbody></table>

> [!NOTE]
>
>- add接口不能与`add_with_ids`接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用`add_with_ids`接口。

<a name="table17254342193617"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add(idx_t n, const uint16_t *x);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndex建库和向底库中添加新的特征向量的功能。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：待添加进底库的特征向量数量。<br><strong><code>const uint16_t *x</code></strong>：待添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">指针“x”的长度应该为dims * n，否则可能出现越界读写错误并引起程序崩溃。<br>“n”的取值范围：0 &lt; n &lt; 1e9。</td></tr>
</tbody></table>

## add\_with\_ids接口<a id="ZH-CN_TOPIC_0000001456694864"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndex建库和向底库中添加新的特征向量的功能，添加时底库特征都有对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：待添加进底库的特征向量数量。<br><strong><code>const float *x</code></strong>：待添加进底库的特征向量。<br><strong><code>const idx_t *ids</code></strong>：待添加进底库的特征向量对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 指针“x”的长度应该为dims * <strong><code>n</code></strong>，指针“ids”的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 当filter开关<a href="./03_AscendIndexConfig.md#ZH-CN_TOPIC_0000001506414705">filterable</a>为“true”时，需要保证“ids”中的时间戳为正。“ids”（类型为uint64_t）中包含了timestamp（时间戳，类型为int32_t）和cid（camera id，类型为uint8_t），如下所示： <pre>-----| cid | timestamp | -----
 14  |  8  |    32     |  10</pre></td></tr>
</tbody></table>

<a name="table562574920111"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const uint16_t *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndex建库和向底库中添加新的特征向量的功能，添加时底库特征都有对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：待添加进底库的特征向量数量。<br><strong><code>const uint16_t *x</code></strong>：待添加进底库的特征向量。<br><strong><code>const idx_t *ids</code></strong>：待添加进底库的特征向量对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 指针“x”的长度应该为dims * n，指针“ids”的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 当filter开关<a href="./03_AscendIndexConfig.md#ZH-CN_TOPIC_0000001506414705">filterable</a>为“true”时，需要保证“ids”中的时间戳为正。“ids”（类型为uint64_t）中包含了timestamp（时间戳，类型为int32_t）和cid（camera id，类型为uint8_t），如下所示： <pre>-----| cid | timestamp | -----
 14  |  8  |    32     |  10</pre></td></tr>
</tbody></table>

## AscendIndex接口<a name="ZH-CN_TOPIC_0000001456695048"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndex(int dims, faiss::MetricType metric, AscendIndexConfig config)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndex的构造函数，生成维度为dims的AscendIndex（单个Index管理的一组向量的维度是唯一的），此时根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndex管理的一组特征向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型，当前支持“faiss::MetricType::METRIC_L2”以及“faiss::MetricType::METRIC_INNER_PRODUCT”。<br><strong><code>AscendIndexConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“dims”为(0, 4096]的整数且需要能被16整除。</td></tr>
</tbody></table>

<a name="table161511529133912"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndex(const AscendIndex&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明AscendIndex拷贝构造函数为空，即AscendIndex为不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndex&amp;</code></strong>：常量AscendIndex。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table62621513124018"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndex();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndex的析构函数，销毁AscendIndex对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getDeviceList接口<a name="ZH-CN_TOPIC_0000001506495857"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>std::vector&lt;int&gt; getDeviceList();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回Index中管理的Device昇腾AI处理器设置，交由子类继承并实现，在本类中不提供相应的实现，仅会返回一个空<strong><code>vector&lt;int&gt;</code></strong>。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">Index中管理的Device昇腾AI处理器设置。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## operator= 接口<a name="ZH-CN_TOPIC_0000001506334661"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndex&amp; operator=(const AscendIndex&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明AscendIndex赋值构造函数为空，即AscendIndex为不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndex&amp;</code></strong>：常量AscendIndex。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reclaimMemory接口<a name="ZH-CN_TOPIC_0000001456695092"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual size_t reclaimMemory();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在保证底库数量不变的情况下，缩减底库占用的内存，交由子类继承并实现，在本类中不提供相应的实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">缩减的内存大小，单位为Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001456535000"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndex删除底库中指定的特征向量的接口。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IDSelector &amp;sel</code></strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">返回被删除的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reserveMemory接口<a name="ZH-CN_TOPIC_0000001456375348"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void reserveMemory(size_t numVecs);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在建立底库前为底库申请预留内存的抽象接口，交由子类继承并实现，在本类中不提供相应的实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t numVecs</code></strong>：申请预留内存的底库数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reset接口<a name="ZH-CN_TOPIC_0000001506414901"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">清空该AscendIndex的底库向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## search接口<a name="ZH-CN_TOPIC_0000001506334641"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndex特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const SearchParameters *params：</code></strong>Faiss的可选参数，默认为“nullptr”，暂不支持该参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。当有效的检索结果不足“k”个时，剩余无效距离用65504或-65504填充（因metric而异）。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。当有效的检索结果不足“k”个时，剩余无效label用-1填充。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">查询的特征向量数据“x”的长度应该为dims * <strong><code>n</code></strong>，“distances”以及“labels”的长度应该为<strong><code>k</code></strong> * <strong><code>n</code></strong>，否则可能会出现越界读写的情况，引起程序的崩溃。其中，“n”的取值范围：0 &lt; n &lt; 1e9；“k”通常不允许超过4096。</td></tr>
<tr><td width="140" align="center" valign="middle">注意事项</td><td valign="middle">使用小库暴搜算法的场景中，如果在底库和batch数较大时出现性能下降现象，需要增大AscendIndexConfig中的“resources”参数值（暴搜算法默认值为128MB）。</td></tr>
</tbody></table>

<a name="table03178548130"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndex特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const uint16_t *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。当有效的检索结果不足“k”个时，剩余无效距离用65504或-65504填充（因metric而异）。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。当有效的检索结果不足“k”个时，剩余无效label用-1填充。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">查询的特征向量数据“x”的长度应该为dims * n，“distances”以及“labels”的长度应该为k * n，否则可能会出现越界读写的情况，引起程序的崩溃。其中，“n”的取值范围：0 &lt; n &lt; 1e9；“k”通常不允许超过4096。</td></tr>
<tr><td width="140" align="center" valign="middle">注意事项</td><td valign="middle">使用小库暴搜算法的场景中，如果在底库和batch数较大时出现性能下降现象，需要增大AscendIndexConfig中的“resources”参数值（暴搜算法默认值为128MB）。</td></tr>
</tbody></table>
