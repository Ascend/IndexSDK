# AscendIndexILFlat<a name="ZH-CN_TOPIC_0000002514896041"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002482656058"></a>

AscendIndexILFlat为ILFlat标准态场景，需要使用Init指定对应资源的初始化，初始化完成之后会申请一段完整空间用于存储底库。在使用完成之后，需要调用Finalize接口对资源进行释放。

AscendIndexILFlat仅支持使用<term>Atlas 推理系列产品</term>，在标准态部署方式下的向量内积距离类型。AscendIndexILFlat在使用时依赖Flat和AICPU算子，具体请参见[Flat](../../05_user_guide.md#flat)和[AICPU](../../05_user_guide.md#aicpu)。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AddFeatures接口<a name="ZH-CN_TOPIC_0000002514776041"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const float *features);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向特征库追加“n”个特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：插入特征向量数目。<br><strong><code>const float *features</code></strong>：待插入的特征向量，长度为n * 向量维度dim。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● “features”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table392463914228"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const float16_t *features);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向特征库追加“n”个特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：插入特征向量数目。<br><strong><code>const float16_t *features</code></strong>：待插入的特征向量，长度为n * 向量维度dim。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● “features”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## AscendIndexILFlat接口<a name="ZH-CN_TOPIC_0000002516511133"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexILFlat();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexILFlat的构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table161511529133912"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexILFlat(const AscendIndexILFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index拷贝函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexILFlat&amp;</code></strong>：AscendIndexILFlat对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table62621513124018"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexILFlat();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexILFlat的析构函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## ComputeDistance接口<a name="ZH-CN_TOPIC_0000002482736032"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出经过映射后的距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，长度为n * 向量维度dim。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（“ntotalPad”为 (ntotal + 15) / 16 * 16，即“ntotal”对16补齐）。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：合理的n值应该在(0, capacity]之间。<br>● <strong><code>distances</code></strong>：需要提供的空间长度为n * ntotalPad（“ntotalPad”为(ntotal + 15) / 16 * 16，即“ntotal”对16补齐的结果，每个query的有效比对距离存储在前“ntotal”的空间，补齐部分数据没有实际意义）。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “queries”和“distances”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table17574555124816"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR ComputeDistance(int n, const float *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出经过映射后的距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float *queries</code></strong>：待查询特征向量，长度为n * 向量维度dim。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（“ntotalPad”为 (ntotal + 15) / 16 * 16，即“ntotal”对16补齐）。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：合理的n值应该在(0, capacity]之间。<br>● <strong><code>distances</code></strong>：需要提供的空间长度为n * ntotalPad（“ntotalPad”为(ntotal + 15) / 16 * 16，即“ntotal”对16补齐的结果，每个query的有效比对距离存储在前“ntotal”的空间，补齐部分数据没有实际意义）。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “queries”和“distances”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000002514896043"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const float *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">ComputeDistance计算待查询向量与所有底库向量的距离，而ComputeDistanceByIdx接口只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则返回映射后的topk结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float *queries</code></strong>：待查询特征向量，有效长度为n * dim，“dim”需与初始化时指定的dim保持一致。<br><strong><code>const int *num</code></strong>： 给定每个query要比对的底库特征向量数目，长度为n。<br><strong><code>const idx_t *indices</code></strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大“num”补齐空间占用，“indices”长度为n * max(num)。输入在host，indices为host指针；输入在device，indices为device指针。<br><strong><code>MEMORY_TYPE memoryType</code></strong>：输入输出存放位置策略，默认为MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，可选策略如下：<br>● MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST：输入在host，输出在host。<br>● MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE：输入在device，输出在device。<br>● MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST：输入在device，输出在host。<br>● MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE：输入在host，输出在device。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“*table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大“num”补齐空间占用，空间长度为n * max(num)。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● <strong><code>num</code></strong>：由用户指定，长度为n，每个query的num值应该在[0, ntotal]之间。<br>● <strong><code>indices</code></strong>：每个特征的索引应该在[0, ntotal)之间。<br>● 接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”、“distances”和“num”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。<br>● 选择memoryType存放策略时，“queries”、“distances”需要为对应位置指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table93703718308"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">ComputeDistance计算待查询向量与所有底库向量的距离，而ComputeDistanceByIdx接口只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则返回映射后的topk结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，有效长度为n * dim，“dim”需与初始化时指定的dim保持一致。<br><strong><code>const int *num</code></strong>： 给定每个query要比对的底库特征向量数目，长度为n。<br><strong><code>const idx_t *indices</code></strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大“num”补齐空间占用，“indices”长度为n * max(num)。输入在host，indices为host指针；输入在device，indices为device指针。<br><strong><code>MEMORY_TYPE memoryType</code></strong>：输入输出存放位置策略，默认为MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，可选策略如下：<br>● MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST：输入在host，输出在host。<br>● MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE：输入在device，输出在device。<br>● MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST：输入在device，输出在host。<br>● MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE：输入在host，输出在device。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“*table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大“num”补齐空间占用，空间长度为n * max(num)。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● <strong><code>num</code></strong>：由用户指定，长度为n，每个query的num值应该在[0, ntotal]之间。<br>● <strong><code>indices</code></strong>：每个特征的索引应该在[0, ntotal)之间。<br>● 接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”、“distances”和“num”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## Finalize接口<a name="ZH-CN_TOPIC_0000002482656060"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void Finalize();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">释放特征库管理资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## GetFeatures接口<a name="ZH-CN_TOPIC_0000002484074790"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, float *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条指定下标索引的特征向量。输出在host。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：获取底库向量的个数。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *features</code></strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间，ntotal可以通过GetNTotal接口获取。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table018415716495"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条指定下标索引的特征向量。输出在host。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：获取底库向量的个数。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float16_t *features</code></strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间，ntotal可以通过GetNTotal接口获取。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetFeaturesOnDevice<a name="ZH-CN_TOPIC_0000002516516843"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeaturesOnDevice (int n, float16_t *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条指定下标索引的特征向量。输出在Device。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：获取底库向量的个数。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float16_t *features</code></strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。Device侧指针。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间，ntotal可以通过GetNTotal接口获取。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table15312115612410"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeaturesOnDevice (int n, float *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条指定下标索引的特征向量。输出在Device。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：获取底库向量的个数。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *features</code></strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。Device侧指针。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间，ntotal可以通过GetNTotal接口获取。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000002514776043"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int GetNTotal() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询当前特征库特征向量数目的理论最大值。如果插入特征向量indices连续，则ntotal等于特征向量数目。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int ntotal</code></strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>int</code></strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Init接口<a name="ZH-CN_TOPIC_0000002482736034"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexILFlat的初始化函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dim</code></strong>：AscendIndexILFlat管理的特征向量的维度。<br><strong><code>int capacity</code></strong>：底库最大容量，接口会根据“capacity”值申请capacity * dim * sizeof(fp16) 字节内存数据。<br><strong><code>faiss::MetricType metricType</code></strong>： 特征距离类别（向量内积、欧氏距离、余弦相似度）。<br><strong><code>const std::vector&lt;int&gt; &amp;deviceList</code></strong>：Device侧资源配置。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为“-1”，表示设置为“128MB”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {32, 64, 128, 256, 384, 512}<br>● metricType：AscendIndexILFlat目前只实现了向量内积距离，即只支持“faiss::MetricType::METRIC_INNER_PRODUCT”。<br>● capacity：接口允许为底库申请的内存上限设为12,288,000,000Byte，同时“capacity”的值域约束为[0, 12000000]。以512维、FP16类型的底库向量为例，最大支持的“capacity”为1200万(12288000000 / (512 * sizeof(fp_16)) )。<br>● 对于256维、FP16类型的底库向量，尽管内存约束支持更大的“capacity”，“capacity”最大也只能设为1200万。<br>仅支持配置单卡，暂不支持配置多卡，需满足<strong><code>deviceList.size() == 1</code></strong>。resourceSize：可以配置为-1或[134217728，4294967296]之间的值，相当于[128MB，4096MB]。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置为“1024MB”。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000002482794858"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexILFlat&amp; operator=(const AscendIndexILFlat &amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexILFlat &amp;</code></strong>：AscendIndexILFlat对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## RemoveFeatures接口<a name="ZH-CN_TOPIC_0000002482917750"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR RemoveFeatures(int n, const idx_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">删除向量库中“n”个指定下标索引的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：删除特征向量数目。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间，ntotal可以通过GetNTotal接口获取。<br>● <strong><code>n</code></strong>：取值应在[0, capacity]之间。<br>● “indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## Search接口<a name="ZH-CN_TOPIC_0000002514896045"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询与query向量距离最近的“topk”个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出映射后的距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，长度为n * 向量维度dim。<br><strong><code>int topk</code></strong>：查询向量和底库的比对距离进行排序，返回“topk”条结果。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：外部内存，与query相似度最高的<strong><code>topk</code></strong>* <strong><code>n</code></strong>个底库特征向量所对应的余弦距离，长度为n * topk。<br><strong><code>idx_t *indices</code></strong>：外部内存，返回与query相似度最高的“topk”个底库向量对应的下标索引，长度为n * topk。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● <strong><code>topk</code></strong>：取值应在(0, 1024]之间。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”和“distances”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table838713119461"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询与query向量距离最近的“topk”个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出映射后的距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float *queries</code></strong>：待查询特征向量，长度为n * 向量维度dim。<br><strong><code>int topk</code></strong>：查询向量和底库的比对距离进行排序，返回“topk”条结果。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：外部内存，与query相似度最高的<strong><code>topk</code></strong>* <strong><code>n</code></strong>个底库特征向量所对应的余弦距离，长度为n * topk。<br><strong><code>idx_t *indices</code></strong>：外部内存，返回与query相似度最高的“topk”个底库向量对应的下标索引，长度为n * topk。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● <strong><code>topk</code></strong>：取值应在(0, 1024]之间。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”和“distances”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchByThreshold接口<a name="ZH-CN_TOPIC_0000002482656062"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且table为非空指针），则返回映射后的topk结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float *queries</code></strong>：待查询特征向量，长度为n * dim。<br><strong><code>float threshold</code></strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照“threshold”进行过滤。<br><strong><code>int topk</code></strong>：query和底库的比对距离进行排序，返回“topk”条结果。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“*table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int *num</code></strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。<br><strong><code>idx_t * indices</code></strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按“topk”补齐占用空间，“indices”总长度为n * topk。<br><strong><code>float *distances</code></strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与“indices”相同。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● <strong><code>topk</code></strong>：取值应在(0, 1024]之间。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”、“distances”和“num”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table910711421721"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且table为非空指针），则返回映射后的topk结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：待查询特征向量的数目。<br><strong><code>const float16_t *queries</code></strong>：待查询特征向量，长度为n * dim。<br><strong><code>float threshold</code></strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照“threshold”进行过滤。<br><strong><code>int topk</code></strong>：query和底库的比对距离进行排序，返回“topk”条结果。<br><strong><code>unsigned int tableLen</code></strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为“10000”。<br><strong><code>const float *table</code></strong>：映射表指针，指向“tableLen”长度的有效映射值存储空间，目前支持的冗余长度为“48”，即“*table”指向的空间长度为10048 * sizeof(float) Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int *num</code></strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。<br><strong><code>idx_t* indices</code></strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按“topk”补齐占用空间，“indices”总长度为n * topk。<br><strong><code>float *distances</code></strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与“indices”相同。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● <strong><code>topk</code></strong>：取值应在(0, 1024]之间。<br>● 传递“tableLen”和“table”参数同时满足要求时，接口会对计算出来的<strong><code>distance</code></strong>进行映射：首先将<strong><code>distance</code></strong>值归一化为 [0, 1]之间的浮点数<strong><code>f1</code></strong>，然后用<strong><code>f1</code></strong>乘上“tableLen”并取整，这样得到[0, <strong><code>tableLen</code></strong>]之间的整数索引，再利用该整数索引作为偏移，去“table”指向的内存空间取出对应的<strong><code>score</code></strong>，即完成映射，将<strong><code>score</code></strong>存入“distance” 。 索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。<br>● “indices”、“queries”、“distances”和“num”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SetNTotal接口<a name="ZH-CN_TOPIC_0000002514776045"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SetNTotal(int n);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">为外部提供调整“ntotal”计数。<br>每次增加底库向量后，Index内部尽管会根据最大插入下标更新“ntotal”值，但并没有记录[0, ntotal]范围内哪些区域是无效的空间，因此<strong><code>RemoveFeatures</code></strong>操作没有改变“ntotal”的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置“ntotal”，这样可以在可控范围内减少算子的计算量，以提高接口性能。<br>例如：当前插入100条向量，底库索引为0~99 时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部“ntotal”保持不变，只能设为[ntotal, capacity]之间的值，再次执行删除索引为90~99的底库，此时可以手动把“ntotal”设置为[80, capacity]之间的值，设置为“80”时，可以使参与比对的底库数据量有效减少20条。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：由用户在业务面管理的最大底库的索引加1。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle"><strong><code>n</code></strong>：取值应在[0, capacity]之间。</td></tr>
</tbody></table>

## UpdateFeatures接口<a name="ZH-CN_TOPIC_0000002516314733"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR UpdateFeatures (int n, const float16_t *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向特征库更新“n”个指定下标索引的特征向量，如果在下标处不存在特征向量，则添加；如果在下标处已存在特征向量，则修改。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：插入特征向量数目。<br><strong><code>const float16_t *features</code></strong>：待插入的特征向量，长度为n * 向量维度dim。<br><strong><code>const idx_t *indices</code></strong>：待插入特征向量对应的下标索引，有效长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间。<br>● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

<a name="table19567183517113"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR UpdateFeatures(int n, const float *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向特征库更新“n”个指定下标索引的特征向量，如果在下标处不存在特征向量，则添加；如果在下标处已存在特征向量，则修改。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：插入特征向量数目。<br><strong><code>const float *features</code></strong>：待插入的特征向量，长度为n * 向量维度dim。<br><strong><code>const idx_t *indices</code></strong>：待插入特征向量对应的下标索引，有效长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● <strong><code>indices</code></strong>：每个特征的索引应在[0, ntotal)之间。<br>● <strong><code>n</code></strong>：取值应在(0, capacity]之间。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>
