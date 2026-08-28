# IndexIL<a name="ZH-CN_TOPIC_0000001506414825"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456535188"></a>

IndexIL是一个基于连续内存申请机制的特征管理抽象类，服务于将下标索引作为label的检索算法，需要继承实现所有接口来使用。

要求存入底库的向量以及各个接口的query向量均为归一化后的FP16浮点数类型。（**IL**表示“Indices as Labels”。）

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

## AddFeatures接口<a name="ZH-CN_TOPIC_0000001506414693"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向特征库插入“n”个指定下标索引的特征向量，如果在下标处已存在特征向量，该插入操作相当于修改。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：插入特征向量数目。<br><strong><code>const float16_t *features</code></strong>：特征向量，长度为n * 向量维度dim。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 入参由该类的实现类约束。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## IndexIL接口<a name="ZH-CN_TOPIC_0000001456695020"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>IndexIL();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IndexIL的构造函数，生成特征管理对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~IndexIL接口<a name="ZH-CN_TOPIC_0000001506334781"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~IndexIL();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IndexIL的析构函数，销毁特征管理对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Finalize接口<a name="ZH-CN_TOPIC_0000001456375356"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual APP_ERROR Finalize() = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">释放特征库管理资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## GetFeatures接口<a name="ZH-CN_TOPIC_0000001506495833"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询“n”条指定下标索引的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：获取特征向量数目。<br><strong><code>const idx_t *indices</code></strong>：待查询的下标索引，长度为n。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float16_t *features</code></strong>：查询下标索引对应的特征向量，长度为n * 向量维度dim。在调用前需由用户自行申请内存，确保内存大小正确。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 入参由该类的实现类约束。<br>● “features”和“indices”需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000001456535092"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual int GetNTotal() const = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">查询当前特征库向量的最大占用空间。<br>特征向量从索引<strong><code>0</code></strong>开始插入，如果插入特征向量“indices”连续，则“ntotal”等于特征向量数目，否则“ntotal”等于插入向量的最大索引值加1（为性能考虑，算子会批操作内存，默认将最大索引位置及之前的空间都视为有效底库向量并纳入计算），用户需要通过该接口获取index内部记录的底库总量，进而申请对应的内存空间给对应的功能接口传递参数，详细描述请参见具体接口。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>int ntotal</code></strong>：请参见功能描述。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Init接口<a name="ZH-CN_TOPIC_0000001506334657"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize) = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">初始化特征库参数，申请底库内存资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dim</code></strong>：特征向量的维度。<br><strong><code>AscendMetricType metricType</code></strong>： 特征距离类别：向量内积、欧氏距离、余弦相似度。<br><strong><code>int capacity</code></strong>：底库最大容量，等于capacity * dim * sizeof(float) 字节内存数据。<br><strong><code>int resourceSize</code></strong>：提前申请Device的缓存资源，检索接口被调用时可以直接使用这里的资源，而不必调用<strong><code>aclrtmalloc</code></strong>去申请内存，达到优化加速。默认取值-1，代表按默认size申请缓存资源（128MB），可以根据检索业务的数据量和Device上的资源使用情况来更精确地配置实际需要使用的size大小。<br>例如：query的“batch”为“64”，底库总量为100万，而一个FP32数值占用4个字节，那么这里的“resourceSize”可以设置为： 64 * 1000000 * 4 = 256,000,000Byte。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">入参由该类的实现类进行约束。</td></tr>
</tbody></table>

## RemoveFeatures接口<a name="ZH-CN_TOPIC_0000001456534932"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual APP_ERROR RemoveFeatures(int n, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">删除向量库中“n”个指定下标索引的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：删除特征向量数目。<br><strong><code>const idx_t *indices</code></strong>：特征向量对应的下标索引。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 入参由该类的实现类约束。<br>● “indices”需要为非空指针，且长度应该为n，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SetNTotal接口<a name="ZH-CN_TOPIC_0000001456375256"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual APP_ERROR SetNTotal(int n) = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">为外部提供调整“ntotal”计数的接口。<br>每次增加底库向量后，Index内部虽然会根据最大插入下标更新“ntotal”值，但并没有记录[0, ntotal]范围内哪些区域是无效的空间，因此<strong><code>RemoveFeatures</code></strong>操作没有改变“ntotal”的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置“ntotal”，这样可以在可控范围内减少算子的计算量，以提高接口性能。<br>例如：当前插入100条向量，底库索引为0~99时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部“ntotal”保持不变，只能设为[ntotal, capacity]之间的值，再次执行删除索引为90~99的底库，此时可以手动把“ntotal”设置为[80, capacity]之间的值，设置为“80”时，可以使参与比对的底库数据量有效减少20条。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int n</code></strong>：由用户在业务面管理的最大底库的索引加1。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">入参由该类的实现类约束。</td></tr>
</tbody></table>
