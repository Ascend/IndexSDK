
# AscendIndexIVFSP<a name="ZH-CN_TOPIC_0000001635576081"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001635815481"></a>

昇腾原生IVFSP检索算法，使用自研矩阵近似策略，压缩特征向量后存底库，并使用自研倒排链策略选取出最可能包含Ground Truth（真实）的底库，最后使用自研检索策略在倒排链过滤后的底库进行检索得到Top K向量结果。

AscendIndexIVFSP只支持标准态场景，且只支持<term>Atlas 推理系列产品</term>。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## add接口<a name="ZH-CN_TOPIC_0000001585895568"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向底库中添加特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const float *x</code></strong>：添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 指针“x”的长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 底库向量总数“n”通常大于0且小于1e9。<br>● 一次性add的数据量应该小于等于特征底库数据大小。</td></tr>
</tbody></table>

> [!NOTE]
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。
>- add接口在小batch添加场景进行了性能优化，此场景根据数据集不同，精度会有所降低，建议在已有底库场景下用小batch添加。

## add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001586055512"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向底库中添加特征向量并指定对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：添加进底库的特征向量数量。<br><strong><code>const float *x</code></strong>：添加进底库的特征向量。<br><strong><code>const idx_t *ids</code></strong>：添加进底库的特征向量对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">指针“x”的长度应该为dims * <strong><code>n</code></strong>，指针“ids”的长度应为“n”，否则可能出现越界读写错误并引起程序崩溃。用户需要根据自己的业务场景，保证“ids”的合法性，如底库中存在重复的ID，检索结果中的“label”将无法对应具体的底库向量。<br>“n”的取值范围：0 &lt; n &lt; 1e9。</td></tr>
</tbody></table>

> [!NOTE]
> add\_with\_ids接口在小batch添加场景进行了性能优化，此场景根据数据集不同，精度会有所降低，建议在已有底库场景下用小batch添加。

## AscendIndexIVFSP接口<a name="ZH-CN_TOPIC_0000001585736168"></a>

> [!NOTE]
>将参数“config”传递给函数前，请根据实际情况先设置conf.handleBatch、conf.nprobe、conf.searchListSize的值（字段描述参考[公共参数](./06_AscendIndexIVFSPConfig.md#ZH-CN_TOPIC_0000001635696057)）。
>其中conf.handleBatch、conf.searchListSize值需与[IVFSP](../../05_user_guide.md#ivfsp)业务算子模型文件生成中的nprobe handle batch、search list size保持一致。
>conf.filterable（继承自[AscendIndexConfig](../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig) ）默认为“false”，如果要使用search\_with\_filter\(\)接口，需设置**conf.filterable = true**。“conf.filterable”设置为“true”将在NPU卡上存储额外的信息，消耗更多的NPU卡上内存。

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSP的构造函数，根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexIVFSP管理的一组特征向量的维度。<br><strong><code>int nonzeroNum</code></strong>：特征向量压缩降维后非零维度个数。<br><strong><code>int nlist</code></strong>：聚类中心的个数，与<a href="../../05_user_guide.md#ivfsp">IVFSP业务算子模型文件生成</a>中的&lt;centroid num&gt;参数值对应。<br><strong><code>const char *codeBookPath</code></strong>：IVFSP使用的码本文件路径。<br><strong><code>faiss::ScalarQuantizer::QuantizerType qType</code></strong>：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。当前“faiss::MetricType metric”仅支持“METRIC_L2”。<br><strong><code>AscendIndexIVFSPConfig</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 训练生成码本时的&lt;dim&gt;、&lt;nonzero num&gt;、&lt;centroid num&gt; 值应该与此函数的参数“dims”、“nonzeroNum”、“nlist”对应。<br>● “codeBookPath”加载的码本应该与此函数的参数“dims”、“nonzeroNum”、“nlist”对应，且程序的执行用户是码本文件的属主；且码本文件不能为软链接。<br>● 当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。<br>● “nonzeroNum”需为16的倍数且小于等于min(128, dims)。<br>● metric ∈ {faiss::MetricType::METRIC_L2}。</td></tr>
</tbody></table>

<a name="table49022324218"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &amp;codeBookSharedIdx, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSP的构造函数，根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexIVFSP管理的一组特征向量的维度。<br><strong><code>int nonzeroNum</code></strong>：特征向量压缩降维后非零维度个数。<br><strong><code>int nlist</code></strong>：聚类中心的个数，与<a href="../../05_user_guide.md#ivfsp">IVFSP业务算子模型文件生成</a>中的&lt;centroid num&gt;参数值对应。<br><strong><code>const AscendIndexIVFSP &amp;codeBookSharedIdx</code></strong>：共享码本的AscendIndexIVFSP对象。<br><strong><code>faiss::ScalarQuantizer::QuantizerType qType</code></strong>：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。当前“faiss::MetricType metric”仅支持“METRIC_L2”。<br><strong><code>AscendIndexIVFSPConfig</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 训练生成码本时的&lt;dim&gt;、&lt;nonzero num&gt;、&lt;centroid num&gt; 值应该与此函数的参数“dims”、“nonzeroNum”、“nlist”对应。<br>● codeBookSharedIdx共享码本的码本配置要与当前Index的码本配置相同，且配置相同的Device资源。<br>● 当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。<br>● “nonzeroNum”需为16的倍数且小于等于min(128, dims)。<br>● metric ∈ {faiss::MetricType::METRIC_L2}。</td></tr>
</tbody></table>

<a name="table8581162710235"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSP (const AscendIndexIVFSP&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFSP&amp;</code></strong>：常量AscendIndexIVFSP。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table186918413239"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSP();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSP的析构函数，销毁AscendIndexIVFSP对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table241282321712"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSP的构造函数，根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">● int dims：AscendIndexIVFSP管理的一组特征向量的维度。<br>● int nonzeroNum：特征向量压缩降维后非零维度个数。<br>● int nlist：聚类中心的个数，与<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP业务算子模型文件生成”中的&lt;centroid num&gt;参数值对应。<br>● faiss::ScalarQuantizer::QuantizerType qType：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。<br>● faiss::MetricType metric：AscendIndex在执行特征向量相似度检索时使用的距离度量类型。当前“faiss::MetricType metric”仅支持“METRIC_L2”。<br>● AscendIndexIVFSPConfig：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。<br>● “nonzeroNum”需为16的倍数且小于等于min(128, dims)。<br>● metric ∈ {faiss::MetricType::METRIC_L2}。</td></tr>
</tbody></table>

## loadAllData接口<a id="ZH-CN_TOPIC_0000001585736172"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void loadAllData(const char *dataPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将Index结构从磁盘读入Device，包括压缩降维后的特征向量和码本数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const char *dataPath：</code></strong>数据文件路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“dataPath”对应的文件应该是调用saveAllData方法得到的落盘文件，程序执行用户对其有读权限；且文件不能为软链接。<br>该接口无法共享码本，如需共享码本，建议使用loadAllData。</td></tr>
</tbody></table>

<a name="table115591219131513"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>static std::shared_ptr&lt;AscendIndexIVFSP&gt; loadAllData(const AscendIndexIVFSPConfig &amp;config, const uint8_t *data, size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">从内存中恢复AscendIndexIVFSP对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">● <strong><code>const AscendIndexIVFSPConfig &amp;config</code></strong>：Device侧资源配置，当前只需设置config.deviceList以及config.resourceSize即可，其他配置参数会从内存中恢复。<br>● <strong><code>const uint8_t *data</code></strong>：由saveAllData方法得到的内存指针。<br>● <strong><code>size_t dataLen</code></strong>：data指针的真实长度。<br>● <strong><code>const AscendIndexIVFSP *codeBookSharedIdx</code></strong>：共享码本的AscendIndexIVFSP指针，默认为nullptr，即不共享码本。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">从内存中恢复的AscendIndexIVFSP智能指针对象。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● data需要为非空的合法指针。<br>● dataLen为指针data的真实长度，否则可能出现越界读写错误并引起程序崩溃。<br>● codeBookSharedIdx共享码本的码本配置要与当前Index的码本配置相同，且配置相同的Device资源。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001635975413"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFSP&amp; operator=(const AscendIndexIVFSP&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFSP&amp;</code></strong>：常量AscendIndexIVFSP。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001635576085"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexIVFSP删除底库中指定的特征向量的接口。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IDSelector &amp;sel</code></strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">返回被删除的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## reset接口<a name="ZH-CN_TOPIC_0000001635815485"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">清空该AscendIndexIVFSP的底库向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## saveAllData接口<a name="ZH-CN_TOPIC_0000001635696053"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void saveAllData(const char *dataPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将Index结构从Device侧写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const char *dataPath</code></strong>：保存的数据文件路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">用户应该保证“dataPath”文件路径所在的目录存在，且执行用户对目录具有写权限；出于安全加固的考虑，目录层级中不能含有软链接。<br>当“dataPath”对应的文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</td></tr>
</tbody></table>

<a name="table11876949141314"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void saveAllData(uint8_t *&amp;data, size_t &amp;dataLen) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexIVFSP对象存储至内存中。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>uint8_t *&amp;data</code></strong>：存储AscendIndexIVFSP数据的内存指针。<br><strong><code>size_t &amp;dataLen</code></strong>：data指针的真实长度。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">传入的data需要为空指针，且接口返回后需要用户使用完data后通过delete来释放其内存，否则会造成内存泄漏。</td></tr>
</tbody></table>

## search接口<a name="ZH-CN_TOPIC_0000001635815489"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexIVFSP特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const SearchParameters *params：</code></strong>Faiss的可选参数，默认为“nullptr”，暂不支持该参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。当有效的检索结果不足“k”个时，剩余无效距离用65504或-65504填充。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。当有效的检索结果不足“k”个时，剩余无效label用“-1”填充。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">查询的特征向量数据“x”的长度应该为dims * <strong><code>n</code></strong>，“distances”以及“labels”的长度应该为<strong><code>k</code></strong> * <strong><code>n</code></strong>，否则可能会出现越界读写的情况，引起程序的崩溃。此处“n”的取值范围：0 &lt; n &lt; 1e9；“k”通常不允许超过4096。</td></tr>
</tbody></table>

## search\_with\_filter接口<a name="ZH-CN_TOPIC_0000001585736176"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSP的特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。提供基于CID过滤的功能，“filters”为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const void *filters</code></strong>：过滤条件。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “n”的取值范围：0 &lt; n &lt; 1e9。<br>● <strong><code>k</code></strong>通常不允许超过4096。<br>● “x”需要为非空指针，且长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”、“labels”需要为非空指针，且长度应该为<strong><code>k</code></strong> * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● “filters”需要为非空指针，且长度为n * 6的uint32_t的数组，否则可能出现越界读取的错误并引起程序崩溃。</td></tr>
</tbody></table>

## setNumProbes接口<a name="ZH-CN_TOPIC_0000001635576089"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setNumProbes(int nprobes);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置检索时总的候选桶数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int nprobes</code></strong>：AscendIndexIVFSP的nprobe数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“nprobes”为16的倍数且符合0 &lt; nprobes ≤ nlist。</td></tr>
</tbody></table>

## setVerbose接口<a name="ZH-CN_TOPIC_0000001586055516"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void setVerbose(bool verbose);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置是否显式添加特征向量到底库的进度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>bool verbose</code></strong>：是否显式添加特征向量到底库的进度。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## trainCodeBook接口<a name="ZH-CN_TOPIC_0000002148530670"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void trainCodeBook(const AscendIndexCodeBookInitParams &amp;codeBookInitParams) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IVFSP码本训练接口。如果训练速度较慢，可能是安装OpenBLAS时限制了使用单线程，可以设置环境变量export OMP_NUM_THREADS=4 进行加速</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">const AscendIndexCodeBookInitParams &amp;codeBookInitParams：训练码本所需的初始化参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">参考<a href="../02_approximate_retrieval/13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexCodeBookInitParams接口</a>。</td></tr>
</tbody></table>

## addCodeBook接口<a name="ZH-CN_TOPIC_0000002148372594"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void addCodeBook(const char *codeBookPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">添加训练好的码本。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">const char *codeBookPath：码本路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“codeBookPath”对应的文件是调用trainCodeBook方法得到的码本文件，程序执行用户对其有读权限；且文件不能为软链接。</td></tr>
</tbody></table>

## AscendIndexCodeBookInitParams接口<a name="ZH-CN_TOPIC_0000002183731529"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum, std::string codeBookOutputDir, std::string learnDataPath, bool verbose);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IVFSP训练码本的初始化结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数值</td><td valign="middle"><strong><code>int numIter</code></strong>：训练迭代次数参数，默认为“1”。<br><strong><code>int device</code></strong>：设备逻辑ID，默认为“0”。<br><strong><code>float ratio</code></strong>：训练用原始样本的采样率，默认为“1.0”。<br><strong><code>int batchSize</code></strong>：训练时以batchSize大小执行训练。与<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP训练算子模型文件生成”中的&lt;batch_size&gt;保持一致，默认值为“32768”。<br><strong><code>int codeNum</code></strong>：每次最大按codeNum样本数量操作码本，必须为2的幂次。与<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP训练算子模型文件生成”中的&lt;codebook_batch_size&gt;保持一致，默认为“32768”。<br><strong><code>std::string codeBookOutputDir</code></strong>：生成的码本文件输出到的目录，用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。<br><strong><code>std::string learnDataPath</code></strong>：训练用的原始特征文件路径，支持bin、npy格式，bin存储方式为行优先，数据类型为“float32”。<br><strong><code>bool verbose</code></strong>：是否开启额外打印信息，默认为“true”。</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● numIter∈ (0, 20]。<br>● ratio∈ (0, 1.0]。<br>● batchSize∈ (0, 32768]。<br>● codeNum∈ (0, 32768]。<br>● 当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。<br>● 在执行训练生成码本前，请先参考<a href="../../05_user_guide.md#ivfsp">IVFSP</a>生成训练算子模型文件。</td></tr>
</tbody></table>

## trainCodeBookFromMem接口<a name="ZH-CN_TOPIC_0000002257319034"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IVFSP码本训练接口。训练数据从内存中加载，如果训练速度较慢，可能是安装OpenBLAS时限制了使用单线程，可以设置环境变量export OMP_NUM_THREADS=4进行加速。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams：训练码本所需的初始化参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">了解AscendIndexCodeBookInitFromMemParams相关说明，请参见<a href="#ascendindexcodebookinitfrommemparams接口">AscendIndexCodeBookInitFromMemParams</a>。</td></tr>
</tbody></table>

## AscendIndexCodeBookInitFromMemParams接口<a name="ZH-CN_TOPIC_0000002291969193"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexCodeBookInitFromMemParams (int numIter, int device, float ratio, int batchSize, int codeNum,bool verbose,std::string codeBookOutputDir,const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IVFSP训练码本的初始化结构体。从内存中加载训练数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数值</td><td valign="middle"><strong><code>int numIter：</code></strong>训练迭代次数参数，默认为“1”。<br><strong><code>int device：</code></strong>设备逻辑ID，默认为“0”。<br><strong><code>float ratio：</code></strong>训练用原始样本的采样率，默认为“1.0”。<br><strong><code>int batchSize：</code></strong>训练时以batchSize大小执行训练。与<a href="../../05_user_guide.md#ivfsp">IVFSP训练算子模型文件生成</a>中的&lt;batch_size&gt;保持一致，要求大于“0”，默认值为“32768”。<br><strong><code>int codeNum：</code></strong>每次最大按codeNum样本数量操作码本，必须为2的幂次。与<a href="../../05_user_guide.md#ivfsp">IVFSP训练算子模型文件生成</a>中的&lt;codebook_batch_size&gt;保持一致，要求大于0，默认为“32768”。<br><strong><code>std::string codeBookOutputDir：</code></strong>生成的码本文件输出到的目录。用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。<br><strong><code>bool verbose：</code></strong>是否开启额外打印信息，默认为“true”。<br><strong><code>const float *memLearnData：</code></strong>内存中数据指针，默认为空指针。<br><strong><code>size_t memLearnDataSize：</code></strong>内存中数据长度，默认为0。<br><strong><code>bool isTrainAndAdd：</code></strong>是否训练码本后直接添加到Index开关，默认为false。</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● numIter∈ (0, 20]<br>● ratio∈ (0, 1.0]<br>● memLearnDataSize % dim == 0<br>● memLearnDataSize≤25G<br>● 当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。<br>● 在执行训练生成码本前，请先参考<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节生成训练算子模型文件。<br>● 当isTrainAndAdd为true时，码本训练好之后直接添加到Index中，不会进行落盘；<br>● 当isTrainAndAdd为false时，码本会保存到codeBookOutputDir路径下，需调用addCodeBook手动添加。<br>● memLearnDataSize为指针memLearnData的真实长度，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>
