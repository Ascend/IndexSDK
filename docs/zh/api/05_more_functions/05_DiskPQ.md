# DiskPQ<a name="ZH-CN_TOPIC_0000002382802364"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002382647580"></a>

Index SDK提供PQ（Product Quantization）量化的训练和检索功能。PQ接口不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则可能导致功能异常。

## DiskPQParams接口<a name="ZH-CN_TOPIC_0000002382807444"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>DiskPQParams { int pqChunks = 512; int funcType = 1; int dim = 1; char *pqTable = nullptr; uint32_t *offsets = nullptr; char *tablesTransposed = nullptr; char *centroids = nullptr; }</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">PQ量化结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数值</td><td valign="middle"><strong><code>int pqChunks：</code></strong>表示将原始向量维度dim切分为pqChunks块。<br><strong><code>int funcType：</code></strong>表示进行PQ查表距离计算时使用的计算标准。<br><strong><code>int dim：</code></strong>表示原始数据维度。<br><strong><code>char *pqTable：</code></strong>表示存储码本数据的指针。默认值为nullptr。<br><strong><code>uint32_t *offsets：</code></strong>表示存储每个chunk在原始维度上起始和截止的维度。默认值为nullptr。<br><strong><code>char *tablesTransposed：</code></strong>表示存储码本数据的转置形态指针。默认值为nullptr。<br><strong><code>char *centroids：</code></strong>表示存储每个维度的平均值，用于对数据进行中心化处理。默认值为nullptr。</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● 1 &lt;= pqChunks &lt;= dim。使用较小pqChunks将使用更少内存，但会带来相应的精度损失。一般情况下，推荐使用pqChunks为dim / 8或者dim / 16（均向上取整）。默认值为512。<br>● funcType取值范围为1~3。1表示使用L2距离；2表示使用IP距离；3表示使用cosine距离。默认值为1。<br>● 1 &lt;= dim &lt;= 2000。默认值为1。<br>● pqTable目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。<br>● tablesTransposed目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。</td></tr>
</tbody></table>

## VectorArrayData接口<a name="ZH-CN_TOPIC_0000002416326913"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>VectorArrayData { int length; int maxlen; int dim; size_t itemsize; char *items; }</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">数据封装结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数值</td><td valign="middle"><strong><code>int length：</code></strong>表示结构体中存储的向量条数。<br><strong><code>int maxlen：</code></strong>表示结构体中存储的最大向量条数。<br><strong><code>int dim：</code></strong>表示结构体中存储的向量维度。<br><strong><code>size_t itemsize：</code></strong>保留字段，用户可以选择不设置。<br><strong><code>char *items：</code></strong>表示存储VectorArrayData中数据的指针。默认值为nullptr。</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● 1 &lt;= length &lt;= 100000000。<br>● maxlen是OpenGauss侧保留字段，非OpenGauss用户设置该值等于length值即可。<br>● 1 &lt;= dim &lt;= 2000。<br>● 对于不同接口，用户需要确保items指向不同大小的数据。</td></tr>
</tbody></table>

## ComputePQTable接口<a name="ZH-CN_TOPIC_0000002416446741"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">使用sample中存储的采样底库数据计算PQ码本，并将码本相关的数据存储在参数params中的对应参数里。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>VectorArrayData *sample：</code></strong>指向填充好采样底库数据的VectorArrayData实例的指针。不能为空指针。<br><strong><code>DiskPQParams *params：</code></strong>指向仅包含PQ参数，未填充训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● sample数据填充要求如下：items指向的数据大小为(8 + dim) * length * sizeof(float)字节，即每条向量前有8字节的metadata。非OpenGauss用户使用时，需在每条向量数据添加8字节的任意数据。<br>● params成员变量填充要求如下：dim除满足上述的范围限制要求之外，还需确保与sample中对应的dim字段保持一致。<br>● pqTable必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * 256 （256为每个chunk内的聚类数）* sizeof(float)字节。<br>● offsets必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于(pqChunks + 1) * sizeof(uint32_t)字节。<br>● tablesTransposed必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * 256 * sizeof(float)字节。<br>● centroids必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * sizeof(float)字节。</td></tr>
</tbody></table>

## ComputeVectorPQCode接口<a name="ZH-CN_TOPIC_0000002382647584"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">使用填充好PQ数据的params，对baseData中的底库数据进行量化，并将量化数据写入pqCode指向的缓存区中。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>VectorArrayData *baseData：</code></strong>指向填充好底库数据的VectorArrayData实例的指针。不能为空指针。用户可以根据自身内存的限制，在外层决定baseData中底库数据的大小。<br><strong><code>const DiskPQParams *params：</code></strong>指向填充好PQ参数和训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>uint8_t *pqCode：</code></strong>接收返回的压缩好的底库向量的指针。不能为空指针。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● baseData数据填充要求如下：items指向的数据大小为length * dim * sizeof(float)字节。注意此处与ComputePQTable接口不同， 无需在每条数据前填充代替metadata的数据。<br>● params成员变量填充要求如下：dim除满足上述的范围限制要求之外，还需确保与sample中对应的dim字段保持一致。<br>● pqTable必须指向内存大小为dim * 256 * sizeof(float)字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。<br>● offsets必须指向内存大小为(pqChunks + 1) * sizeof(uint32_t)字节数的offsets数据。用户需要保证指向的内存大小符合，否则有段错误风险。<br>● 对tablesTransposed填充值无要求。<br>● centroids必须指向内存大小为dim * sizeof(float)字节数的centroids数据。用户需要保证指向的内存大小符合，否则有段错误风险。<br>用户需保证pqCode指向的空间大小至少有length * pqChunks字节数。其中，length为VectorArrayData参数；pqChunks为DiskPQParams参数。</td></tr>
</tbody></table>

## GetPQDistanceTable接口<a name="ZH-CN_TOPIC_0000002382807448"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">使用填充好PQ数据的params，对vec指向的query数据进行ADC PQ距离计算，并将PQ距离表写入pqDistanceTable指向的缓存区中。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>char *vec：</code></strong>指向待计算的query数据的指针。<br><strong><code>const DiskPQParams *params：</code></strong>指向填充好PQ参数和训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *pqDistanceTable：</code></strong>接收返回的query与每个chunk内每个centroid距离的指针。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 用户需保证vec指向的空间大小至少有dim * sizeof(float)字节数。目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。<br>● params成员变量填充要求如下：pqTable指向值无要求。<br>● offsets必须指向内存大小为(pqChunks + 1) * sizeof(uint32_t)字节数的offsets数据。用户需要保证指向的内存大小符合，否则有段错误风险。<br>● tablesTransposed必须指向内存大小为dim * 256 * sizeof(float)字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。<br>● centroids必须指向内存大小为dim * sizeof(float)字节数的centroids数据。用户需要保证指向的内存大小符合，否则有段错误风险。<br>用户需保证pqDistanceTable指向的空间大小至少有pqChunks * 256 * sizeof(float)字节数。</td></tr>
</tbody></table>

## GetPQDistance接口<a name="ZH-CN_TOPIC_0000002416326917"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &amp;pqDistance);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">使用basecode指向的底库向量对应的压缩码字数据和GetPQDistanceTable接口中获取的pqDistanceTable，计算query与该底库向量的PQ距离。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const uint8_t *basecode：</code></strong>指向一个底库向量对应的压缩码字数据的指针。<br><strong><code>const DiskPQParams *params：</code></strong>指向填充好pqChunks数值的DiskPQParams实例的指针。不能为空指针。<br><strong><code>const float *pqDistanceTable：</code></strong>指向query对应的ADC PQ距离表的指针。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float &amp;pqDistance：</code></strong>接收最终输出的PQ距离的引用值。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 用户需保证basecode指向的数据大小至少有pqChunks个字节。<br>● 在params中，仅需填充pqChunks值，且与basecode中提到的pqChunks值对应。<br>● 用户需保证pqDistanceTable指向的数据大小至少有pqChunks * 256 * sizeof(float)字节数。<br>● 接口中不会在使用前对pqDistance置零，pqDistance最终结果为原pqDistance值 + 输出的query与basecode的PQ距离，因此推荐输入值为0。</td></tr>
</tbody></table>
