# IndexILFlat<a name="ZH-CN_TOPIC_0000001506614925"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506414785"></a>

IndexILFlat继承自IndexIL，为纯Device侧检索方案，利用昇腾AI处理器和AI Core等资源进行各个接口的使能。程序需要在Host侧编译生成二进制文件，然后将二进制文件和相关运行时依赖部署到Device侧执行。IndexILFlat需要使用[Init](#init接口)指定对应资源的初始化，初始化完之后会申请一段完整空间用于存储底库。在使用完之后，需要调用[Finalize](#finalize接口)接口对资源进行释放。

IndexILFlat方案当前只在<term>Atlas 推理系列产品</term>上进行功能和性能的维护，底库和query向量由用户保证归一化，接口当前仅支持向量内积距离，具体使用方法请参见[IndexILFlat](#indexilflat)。（该算法运行成功依赖TIK算子的om文件，纯Device场景需要用户确保部署的是基于Index SDK交付件生成的om文件，需要确保om文件不被篡改。）

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AddFeatures接口<a name="ZH-CN_TOPIC_0000001456854852"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向特征库插入“n”个指定下标索引的特征向量，如果在下标处已存在特征向量，则修改。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：插入特征向量数目。<br><strong><code>const float16_t *features</code></strong>：待插入的特征向量，长度为n * 向量维度dim。<br><strong><code>const idx_t *indices</code></strong>：待插入特征向量对应的下标索引，有效长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, capacity)之间。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## ComputeDistance接口<a name="ZH-CN_TOPIC_0000001456535116"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0 且table为非空指针），则输出经过映射后的距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，长度为n * 向量维度dim。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（“ntotalPad”为 (ntotal + 15) / 16 * 16，即“ntotal”对16补齐）。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：合理的n值应该在[0, capacity]之间。<br>● <strong><code>distances</code></strong>：需要提供的空间长度为n * ntotalPad（“ntotalPad”为(ntotal + 15) / 16 * 16，即“ntotal”对16补齐的结果，每个query的有效比对距离存储在前“ntotal”的空间，补齐部分数据没有实际意义）。推荐使用<strong><code>aclrtmalloc</code></strong>接口，可以申请到全量的物理内存来使用，优化处理时延。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “queries”和“distances”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000001456694920"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">与ComputeDistance类似，区别在于ComputeDistance计算待查询向量与所有底库向量的距离，而该接口ComputeDistanceByIdx只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则返回映射后的topk结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，有效长度为n * dim，“dim”需与初始化时指定的dim保持一致。<br><strong><code>const int *num</code></strong>： 给定每个query要比对的底库特征向量数目，长度为n。<br><strong><code>const idx_t *indices</code></strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大“num”补齐空间占用，“indices”长度为n * max(num)。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“*table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大“num”补齐空间占用，空间长度为n * max(num)。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● <strong><code>num</code></strong>：由用户指定，长度为n，每个query的num值应该在[0， ntotal]之间。<br>● <strong><code>indices</code></strong>：每个特征的索引应该在[0, ntotal)之间。<br>● 接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”、“distances”和“num”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## ComputeDistanceByThreshold接口<a name="ZH-CN_TOPIC_0000001506615117"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在ComputeDistance的基础上增加了阈值筛选，只返回满足阈值条件的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则distances为映射后再进行阈值过滤的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>float16_t *queries</code></strong>：待查询特征向量，长度为n * 向量维度dim。<br><strong><code>float threshold</code></strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照“threshold”进行过滤。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即*table指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int *num</code></strong>：每条待查询特征向量满足阈值条件的底库向量数量长度为n。<br><strong><code>idx_t *indices</code></strong>：满足阈值条件的底库向量下标索引，每个query符合条件的底库数量不同，当从前往后记录所有有效的index之后，按“ntotalPad”补齐占用的空间，“indices”的总长度应该为n * nTotalPad（“ntotalPad”为 (ntotal + 15) / 16 * 16，即“ntotal”对16补齐）。<br><strong><code>float *distances</code></strong>：满足阈值条件的底库向量与待查向量距离，有效值记录方式和空间size与“indices”相同。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● <strong><code>indices</code></strong>：需要提供的空间长度为n * ntotalPad（“ntotalPad”为 (ntotal + 15) / 16 * 16，即“ntotal”对16补齐的结果，第<strong><code>i</code></strong>个query比对过滤后，有效底库的索引存储在“ntotalPad”的前*(num + i) 的空间，补齐部分数据没有实际意义）。<br>● <strong><code>distances</code></strong>：需要提供的空间长度为n * ntotalPad。<br>● “indices”和“distances”推荐使用<strong><code>aclrtmalloc</code></strong>接口，可以申请到全量的物理内存来使用，优化处理时延。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”、“distances”和“num”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## Finalize接口<a name="ZH-CN_TOPIC_0000001506414845"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Finalize() override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">释放特征库管理资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## GetFeatures接口<a name="ZH-CN_TOPIC_0000001456854992"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条指定下标索引的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：获取底库向量的个数。<br><strong><code>const idx_t *indices</code></strong>：需要获取的n个底库向量对应的索引值。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float16_t *features</code></strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间，ntotal可以通过GetNTotal接口获取。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000001456375336"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int GetNTotal() const override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询当前特征库特征向量数目的理论最大值。如果插入特征向量indices连续，则ntotal等于特征向量数目。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int ntotal</code></strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>int ntotal</code></strong>：请参见功能描述。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## IndexILFlat接口<a name="ZH-CN_TOPIC_0000001456694872"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>IndexILFlat();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IndexILFlat的构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table194381755582"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>IndexILFlat(const IndexILFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const IndexILFlat&amp;：</code></strong>IndexILFlat对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~IndexILFlat接口<a name="ZH-CN_TOPIC_0000001456375172"></a>

<a name="table11904175418"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~IndexILFlat();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IndexILFlat的析构函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Init接口<a name="ZH-CN_TOPIC_0000001456375212"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize = -1) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">初始化特征库参数，申请底库内存资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dim</code></strong>：特征向量的维度。<br><strong><code>AscendMetricType metricType</code></strong>： 特征距离类别（向量内积、欧氏距离、余弦相似度）。<br><strong><code>int capacity</code></strong>：底库最大容量，接口会根据“capacity”值申请capacity * dim * sizeof(fp16) 字节内存数据。<br><strong><code>int64_t resourceSize</code></strong>：提前申请Device的缓存资源，检索接口被调用时可以直接使用这里的资源，而不必调用<strong><code>aclrtmalloc</code></strong>接口去申请内存，达到优化加速。<br>默认取值“-1”，代表按默认size申请缓存资源（128MB），可以根据检索业务的数据量和Device上的资源使用情况来更精确地配置实际需要使用的size大小。<br>例如：query的“batch”为“64”，底库总量为100万，而一个FP32数值占用4个字节，那么这里的“resourceSize”可以设置为：64 * 1000000 * 4 = 256,000,000 Byte，注意接口内部支持申请的最大缓存资源为4GB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {32, 64, 128, 256, 384, 512, 1024}<br>● metricType：IndexILFlat目前只实现了向量内积距离，即只支持“AscendMetricType::ASCEND_METRIC_INNER_PRODUCT”。<br>● capacity：接口允许为底库申请的内存上限设为12,288,000,000Byte，同时capacity的值域约束为(0, 12000000]。以512维、FP16类型的底库向量为例，最大支持的“capacity”为1200万(12288000000 / (512 * sizeof(fp_16)) )。<br>● 对于256维、FP16类型的底库向量，尽管内存约束支持更大的capacity，capacity最大也只能设为1200万。<br>resourceSize：可以配置为-1或[134217728，4294967296]之间的值，数值的单位为Byte，相当于[128MB，4096MB]。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001897140809"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>IndexILFlat&amp; operator=(const IndexILFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const IndexILFlat&amp;：</code></strong>IndexILFlat对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## RemoveFeatures接口<a name="ZH-CN_TOPIC_0000001506414837"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR RemoveFeatures(int n, const idx_t *indices) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">删除向量库中“n”个指定下标索引的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：删除特征向量数目。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间，ntotal可以通过GetNTotal接口获取。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## Search接口<a name="ZH-CN_TOPIC_0000001456854856"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询与query向量距离最近的“topk”个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0 且table为非空指针），则输出映射后的距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，长度为n * 向量维度dim。<br><strong><code>int topk</code></strong>：查询向量和底库的比对距离进行排序，返回“topk”条结果。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：外部内存，与query相似度最高的<strong><code>topk</code></strong>* <strong><code>n</code></strong>个底库特征向量所对应的余弦距离，长度为n * topk。<br><strong><code>idx_t *indices</code></strong>：外部内存，返回与query相似度最高的“topk”个底库向量对应的下标索引，长度为n * topk。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● <strong><code>topk</code></strong>：取值应在[0, 1024]之间。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”和“distances”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchByThreshold接口<a name="ZH-CN_TOPIC_0000001456694892"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且table为非空指针），则返回映射后的topk结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，长度为n * dim。<br><strong><code>float threshold</code></strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照“threshold”进行过滤。<br><strong><code>int topk</code></strong>：query和底库的比对距离进行排序，返回“topk”条结果。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“*table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int *num</code></strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。<br><strong><code>idx_t *indices</code></strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按“topk”补齐占用空间，“indices”总长度为n * topk。<br><strong><code>float *distances</code></strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与“indices”相同。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● <strong><code>topk</code></strong>：取值应在[0, 1024]之间。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”、“distances”和“num”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SetNTotal接口<a name="ZH-CN_TOPIC_0000001456854892"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SetNTotal(int n) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">为外部提供调整“ntotal”计数。<br>每次增加底库向量后，Index内部尽管会根据最大插入下标更新“ntotal”值，但并没有记录[0, ntotal]范围内哪些区域是无效的空间，因此<strong><code>RemoveFeatures</code></strong>操作没有改变“ntotal”的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置“ntotal”，这样可以在可控范围内减少算子的计算量，以提高接口性能。<br>例如：当前插入100条向量，底库索引为0~99 时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部“ntotal”保持不变，只能设为[ntotal, capacity]之间的值，再次执行删除索引为90~99的底库，此时可以手动把“ntotal”设置为[80, capacity]之间的值，设置为“80”时，可以使参与比对的底库数据量有效减少20条。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：由用户在业务面管理的最大底库的索引加1。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle"><strong><code>n</code></strong>：取值应在[0, capacity]之间。</td></tr>
</tbody></table>
