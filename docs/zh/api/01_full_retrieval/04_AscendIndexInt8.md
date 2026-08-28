# AscendIndexInt8<a id="ZH-CN_TOPIC_0000001506495841"></a>

## 功能介绍<a id="ZH-CN_TOPIC_0000001506495913"></a>

AscendIndexInt8作为特征检索组件中的采用INT8特征向量的Index的基类，为特征检索中的其他INT8的Index定义接口。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## add接口<a name="ZH-CN_TOPIC_0000001506334825"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向AscendIndexInt8底库中添加新的特征向量。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const int8_t *x</code></strong>：添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处指针“x”的长度应该为dims * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 底库向量总数的取值范围：0 &lt; n &lt; 1e9。</td></tr>
</tbody></table>

<a name="table6211414109"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add(idx_t n, const char *x);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向AscendIndexInt8底库中添加新的特征向量。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const char *x</code></strong>：添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处指针“x”的长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 底库向量总数的取值范围：0 &lt; n &lt; 1e9。</td></tr>
</tbody></table>

> [!NOTE]
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。

## add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001506614905"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const int8_t *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向AscendIndexInt8底库中添加新的特征向量，且指定特征ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const int8_t *x</code></strong>：添加进底库的特征向量。<br><strong><code>const idx_t *ids</code></strong>：添加进底库的特征向量ID。ID在Index实例中需唯一。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处指针“x”的长度应该为dims * <strong><code>n</code></strong>，指针ids的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。<br>● 底库向量总数的取值范围：0 &lt; n &lt; 1e9。</td></tr>
</tbody></table>

<a name="table38814511704"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const char *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向AscendIndexInt8底库中添加新的特征向量，且指定特征ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const char *x</code></strong>：添加进底库的特征向量。<br><strong><code>const idx_t *ids</code></strong>：添加进底库的特征向量对应的ID。ID在Index实例中需唯一。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处指针“x”的长度应该为dims * <strong><code>n</code></strong>，指针“ids”的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。<br>● 底库向量总数的取值范围：0 &lt; n &lt; 1e9。</td></tr>
</tbody></table>

## assign接口<a name="ZH-CN_TOPIC_0000001506495721"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexInt8特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const int8_t *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 查询的特征向量数据“x”的长度应符合dims * <strong><code>n</code></strong>，“labels”的长度应符合<strong><code>k</code></strong> * <strong><code>n</code></strong>，否则可能会出现越界读写的情况，引起程序的崩溃。<br>● 此处“n”大于0且小于1e9。<br>● 此处“k”大于0且小于等于4096。<br>● 此处<strong><code>n</code></strong> * <strong><code>k</code></strong>小于1e10。</td></tr>
</tbody></table>

## AscendIndexInt8接口<a name="ZH-CN_TOPIC_0000001506614993"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8的构造函数，生成维度为dims的AscendIndexInt8（单个Index管理的一组向量的维度是唯一的），此时根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexInt8管理的一组特征向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndexInt8在执行特征向量相似度检索的时候使用的距离度量类型，当前支持“faiss::MetricType::METRIC_L2”和“faiss::MetricType::METRIC_INNER_PRODUCT”。<br><strong><code>AscendIndexInt8Config config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“dims”为不小于64，不大于1024的整数，且需要能被64整除。</td></tr>
</tbody></table>

<a name="table103312407520"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8(const AscendIndexInt8&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexInt8&amp;</code></strong>：AscendIndexInt8对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table1882220715614"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexInt8();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8的析构函数，销毁AscendIndexInt8对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getDeviceList接口<a name="ZH-CN_TOPIC_0000001672982421"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>std::vector&lt;int&gt; getDeviceList() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">返回Index中管理的Device昇腾AI处理器设置，交由子类继承并实现，在本类中不提供相应的实现，仅会返回一个空<strong><code>vector&lt;int&gt;</code></strong>。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">Index中管理的Device昇腾AI处理器设置。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getDim接口<a name="ZH-CN_TOPIC_0000001690599922"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getDim() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取AscendIndexInt8管理的一组特征向量的维度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">AscendIndexInt8管理的一组特征向量的维度。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getNTotal接口<a name="ZH-CN_TOPIC_0000001738718517"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>faiss::idx_t getNTotal() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取AscendIndexInt8已添加进底库的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">AscendIndexInt8已添加进底库的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getMetricType接口<a name="ZH-CN_TOPIC_0000001738678653"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>faiss::MetricType getMetricType() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取AscendIndexInt8执行特征向量相似度检索的时候使用的距离度量类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">返回AscendIndexInt8执行特征向量相似度检索的时候使用的距离度量类型。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## isTrained接口<a name="ZH-CN_TOPIC_0000001690759666"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>bool isTrained() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">判断AscendIndexInt8是否已训练。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">AscendIndexInt8已训练状态，“true”表示已训练，“false”表示未训练。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506414841"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8&amp; operator=(const AscendIndexInt8&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexInt8&amp;</code></strong>：常量AscendIndexInt8。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reclaimMemory接口<a name="ZH-CN_TOPIC_0000001506615133"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual size_t reclaimMemory();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基类中定义的虚函数，具体描述参考子类。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001456695088"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexInt8删除底库中指定的特征向量的接口。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IDSelector &amp;sel</code></strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">返回被删除的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reserveMemory接口<a name="ZH-CN_TOPIC_0000001506615065"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void reserveMemory(size_t numVecs);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基类中定义的虚函数，具体描述参考子类。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t numVecs</code></strong>：申请预留内存的底库数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## search接口<a name="ZH-CN_TOPIC_0000001506414889"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexInt8特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的距离及ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const int8_t *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。当有效的检索结果不足“k”个时，剩余无效距离用65504或-65504填充（因metric而异）。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。当有效的检索结果不足“k”个时，剩余无效label用-1填充。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 查询的特征向量数据“x”的长度应该为dims * <strong><code>n</code></strong>，“distances”以及“labels”的长度应该为k * <strong><code>n</code></strong>，否则可能会出现越界读写的情况，引起程序的崩溃。<br>● 此处“n”大于0且小于1e9。<br>● 此处“k”大于0且小于等于4096。</td></tr>
</tbody></table>

<a name="table88671631181418"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexInt8特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的距离及ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const char *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 查询的特征向量数据“x”的长度应该为dims * <strong><code>n</code></strong>，“distances”以及“labels”的长度应该为k * <strong><code>n</code></strong>，否则可能会出现越界读写的情况，引起程序的崩溃。<br>● 此处“n”大于0且小于1e9。<br>● 此处“k”大于0且小于等于4096。</td></tr>
</tbody></table>

## train接口<a name="ZH-CN_TOPIC_0000001456534956"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void train(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基类中定义的虚函数，具体描述参考子类。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const int8_t *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## updateCentroids接口<a name="ZH-CN_TOPIC_0000001506414833"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void updateCentroids(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基类中定义的虚函数，具体描述参考子类。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const int8_t *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table2023134918146"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void updateCentroids(idx_t n, const char *x);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基类中定义的虚函数，具体描述参考子类。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const char *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
