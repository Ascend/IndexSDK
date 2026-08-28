# AscendIndexTS<a name="ZH-CN_TOPIC_0000001507640105"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001507879785"></a>

时空库功能类接口。添加底库特征时，每条特征可以配置一个属性FeatureAttr，执行检索功能时每一批query向量可以配置一个过滤器AttrFilter，该过滤器首先对全量的底库进行筛选并与符合条件的向量进行比对。

当前支持以下算法：

- 二值化特征检索（汉明距离）：使用前需要手动生成[BinaryFlat](../../05_user_guide.md#binaryflat)、[Mask](../../05_user_guide.md#mask)算子并移动到对应的“modelpath”目录中。
- Int8Flat（cos距离）、FP16Flat（IP距离）、Int8Flat（L2距离）：使用前需要手动生成[Mask](../../05_user_guide.md#mask)算子并移动到对应的“modelpath”目录中。
- 支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AddFeature接口<a name="ZH-CN_TOPIC_0000001458360182"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddFeature(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const uint8_t *customAttr = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">添加特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count</code></strong>：待添加的特征数量。<br><strong><code>const void *features</code></strong>：待添加的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型；FP16Flat距离为float类型。<br><strong><code>const FeatureAttr *attributes</code></strong>：待添加的特征属性，具体请参见<a href="./05_FeatureAttr.md#ZH-CN_TOPIC_0000001507967381">FeatureAttr</a>。<br><strong><code>const int64_t *labels</code></strong>：待添加的特征Label，使用上需要保证Label在Index实例中的唯一性。<br><strong><code>const uint8_t *customAttr</code></strong>：待添加的用户自定义特征属性。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”单次取值在[1, 1e6]区间，底库最大值1e9。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attributes”长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “labels”长度为count，各元素不重复且都不在底库中，否则可能出现越界读写错误并引起程序崩溃。<br>● “customAttr”取值为空指针或者长度为count * customAttrLen，否则可能出现越界读写错误并引起程序崩溃；customAttrLen在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>或<a href="#ZH-CN_TOPIC_0000002013206217">InitWithExtraVal</a>设置。</td></tr>
</tbody></table>

> [!NOTE]
>AddFeature不能与AddWithExtraVal接口混用。

## AddFeatureByIndice接口<a name="ZH-CN_TOPIC_0000002411433020"></a>

> [!NOTE]
>
>- AddFeatureByIndice接口不能和AddFeature、AddWithExtraVal接口混用。
>- 使用AddFeatureByIndice接口按位置添加底库之后，不能使用GetExtraValAttrByLabel等依赖Label的接口，AddFeatureByIndice和GetFeatureByIndice需配套使用。

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddFeatureByIndice(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *indices, const ExtraValAttr *extraVal = nullptr, const uint8_t *customAttr = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">按照位置来添加底库特征。此接口当前只支持FlatIP和Int8Flat（cos距离）。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count：</code></strong>待添加的特征数量。<br><strong><code>const void *features：</code></strong>待添加的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型；FP16Flat距离为float类型。<br><strong><code>const FeatureAttr *attributes：</code></strong>待添加的特征属性。<br><strong><code>const int64_t *indices：</code></strong>待添加的特征在底库中的位置。<br><strong><code>const ExtraValAttr *extraVal：</code></strong>待添加的附加特征属性。<br><strong><code>const uint8_t *customAttr：</code></strong>待添加的用户自定义特征属性。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”单次取值在[1, 1e6]区间，底库最大值1e9。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attributes”长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “indices”长度为count，否则可能出现越界读写错误并引起程序崩溃。需为严格递增且非负的值，值小于底库数量时表示替换，值大于等于底库数量时表示新增，此时值需要连续的。<br>● “extraVal”取值为空指针或者长度为count，否则可能出现越界读写错误并引起程序崩溃。取值为空指针时表示不需要添加附加属性。<br>● “customAttr”取值为空指针或者长度为count * customAttrLen，否则可能出现越界读写错误并引起程序崩溃。取值为空指针时表示不需要添加自定义属性。</td></tr>
</tbody></table>

## AddWithExtraVal接口<a name="ZH-CN_TOPIC_0000001976650872"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddWithExtraVal(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const ExtraValAttr *extraVal, const uint8_t *customAttr = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">添加附加属性特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count</code></strong>：待添加的特征数量。<br><strong><code>const void *features</code></strong>：待添加的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型。<br><strong><code>const FeatureAttr *attributes</code></strong>：待添加的特征属性，具体请参见<a href="./05_FeatureAttr.md#ZH-CN_TOPIC_0000001507967381">FeatureAttr</a>。<br><strong><code>const int64_t *labels</code></strong>：待添加的特征Label，使用上需要保证Label在Index实例中的唯一性。<br><strong><code>const ExtraValAttr *extraVal</code></strong>：待添加的附加特征属性，具体请参见<a href="./03_ExtraValAttr.md#ZH-CN_TOPIC_0000002013198657">ExtraValAttr</a>。<br><strong><code>const uint8_t *customAttr</code></strong>：待添加的用户自定义特征属性。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”单次取值在[1, 1e6]区间，底库最大值1e9。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attributes”长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “labels”长度为count，各元素不重复且都不在底库中，否则可能出现越界读写错误并引起程序崩溃。<br>● “extraVal”长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “customAttr”取值为空指针或者长度为count * customAttrLen，否则可能出现越界读写错误并引起程序崩溃；customAttrLen在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>或<a href="#ZH-CN_TOPIC_0000002013206217">InitWithExtraVal</a>设置。</td></tr>
</tbody></table>

## AscendIndexTS接口<a name="ZH-CN_TOPIC_0000001458200394"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexTS() = default;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexTS的构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table91172211633"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexTS(const AscendIndexTS &amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexTS &amp;</code></strong>：AscendIndexTS对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexTS接口<a name="ZH-CN_TOPIC_0000001507760865"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexTS() = default;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexTS的析构函数，销毁特征管理对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## DeleteFeatureByLabel接口<a name="ZH-CN_TOPIC_0000001458200398"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR DeleteFeatureByLabel(int64_t count, const int64_t *labels);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">批量移除指定Label的特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count</code></strong>：待移除的特征数量。<br><strong><code>const int64_t *labels</code></strong>：特征Label。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 1e6]区间。<br>● “labels”长度为count，各元素不重复且在底库实际存在，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## DeleteFeatureByToken接口<a id="ZH-CN_TOPIC_0000001458680018"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR DeleteFeatureByToken(int64_t count, const uint32_t *tokens);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">批量移除指定Token ID的特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count</code></strong>：待删除的Token数量。<br><strong><code>const uint32_t *tokens</code></strong>：Token对应的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 1e6]区间。<br>● “tokens”的长度为count，待移除的“tokens”需要在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## FastDeleteFeatureByIndice接口<a name="ZH-CN_TOPIC_0000002445152089"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR FastDeleteFeatureByIndice(int64_t count, const int64_t *indices);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">按照位置来删除底库特征。此接口只支持TSFlatIP和TSInt8FlatCos的附加相似度场景。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count：</code></strong>待删除的特征数量。<br><strong><code>const int64_t *indices：</code></strong>待删除的特征在底库中的位置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”需要大于0，小于等于底库数量。<br>● “indices”长度为count，否则可能出现越界读写错误并引起程序崩溃。值需大于等于0，小于底库数量。</td></tr>
</tbody></table>

## FastDeleteFeatureByRange接口<a name="ZH-CN_TOPIC_0000002445960745"></a>

<a name="table18950829154115"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR FastDeleteFeatureByRange(int64_t start, int64_t count);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">从start位置来批量删除count个底库特征。此接口只支持TSFlatIP和TSInt8FlatCos的附加相似度场景。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t start：</code></strong>批量删除的特征起始位置。<br><strong><code>int64_t count：</code></strong>批量删除的特征数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “start”需要大于等于0，小于底库数量。<br>● “count”需要大于0，小于等于底库数量。<br>● “start”与“count”的和小于等于底库数量。</td></tr>
</tbody></table>

## GetBaseByRange接口<a name="ZH-CN_TOPIC_0000001818301380"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基于范围查询底库。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t offset</code></strong>：获取底库特征初始偏移值。<br><strong><code>uint32_t num</code></strong>：特征数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int64_t *labels</code></strong>：特征Label。<br><strong><code>void *features</code></strong>：特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型；FP16Flat距离为float类型。<br><strong><code>FeatureAttr *attributes</code></strong>：特征属性。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 0 &lt;= offset &lt; 8.0e8<br>● 0 &lt; num &lt;= 8.0e8<br>● offset + num &lt;= ntotal<br>● “labels”长度为num，否则可能出现越界读写错误并引起程序崩溃。<br>● “features”长度为num * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attributes”长度为num，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetBaseByRangeWithExtraVal接口<a name="ZH-CN_TOPIC_0000001976495686"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetBaseByRangeWithExtraVal(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes, ExtraValAttr *extraVal) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">基于范围查询带附加属性的底库。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t offset</code></strong>：获取底库特征初始偏移值。<br><strong><code>uint32_t num</code></strong>：特征数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int64_t *labels</code></strong>：特征Label。<br><strong><code>void *features</code></strong>：特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型。<br><strong><code>FeatureAttr *attributes</code></strong>：特征属性。<br><strong><code>ExtraValAttr *extraVal</code></strong>：附加属性。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 0&lt;=offset&lt;8.0e8<br>● 0 &lt; num &lt;= 8.0e8<br>● offset + num &lt;= ntotal<br>● “labels”长度为num，否则可能出现越界读写错误并引起程序崩溃。<br>● “features”长度为num * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attributes”长度为num，否则可能出现越界读写错误并引起程序崩溃。<br>● <strong><code>extraVal</code></strong>长度为num，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetBaseMask接口<a name="ZH-CN_TOPIC_0000002445112157"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetBaseMask(int64_t count, uint8_t *mask);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取底库是否被快速删除的标志。如果某个bit位上为0，表示该位置的底库被删除了，是无效的。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count：</code></strong>mask数组有效长度。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>uint8_t *mask：</code></strong>标记底库是否被删除的数组。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”值需为[1, ceil(ntotal/8)]，否则可能出现越界读写错误并引起程序崩溃。其中，ntotal为底库特征数量。<br>● “mask”长度需大于等于count，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetCustomAttrByBlockId接口<a name="ZH-CN_TOPIC_0000001736682593"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetCustomAttrByBlockId(uint32_t blockId, uint8_t *&amp;customAttr) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取指定blockId的自定义属性。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t blockId</code></strong>：待获取的blockId。<br><strong><code>uint8_t *&amp;customAttr</code></strong>：Device侧的用户自定义特征属性。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“customAttr”长度为customAttrBlockSize * customAttrLen，否则可能出现越界读写错误并引起程序崩溃。customAttrBlockSize和customAttrLen在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>或<a href="#ZH-CN_TOPIC_0000002013206217">InitWithExtraVal</a>设置。</td></tr>
</tbody></table>

## GetExtraValAttrByLabel接口<a name="ZH-CN_TOPIC_0000001976655414"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetExtraValAttrByLabel(int64_t count, const int64_t *labels, ExtraValAttr *extraVal) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取指定Label特征的附加属性。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count</code></strong>：获取特征的数量。<br><strong><code>const int64_t *labels</code></strong>：特征Label。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>ExtraValAttr *extraVal</code></strong>：附加属性。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 1e6]区间。<br>● “labels”长度为count，各元素不重复且在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。如输入的“labels”不存在底库中，接口返回的附加属性中，“val”为“INT16_MIN”。<br>● “extraVal”长度为count，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetFeatureAttrByLabel接口<a name="ZH-CN_TOPIC_0000001594544301"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeatureAttrByLabel(int64_t count, const int64_t *labels, FeatureAttr *attributes) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取指定Label特征的属性。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count</code></strong>：获取特征的数量。<br><strong><code>const int64_t *labels</code></strong>：特征Label。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>FeatureAttr *attributes</code></strong>：特征属性。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 1e6]区间。<br>● “labels”长度为count，各元素不重复且在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。如输入的“labels”不存在底库中，接口返回的特征属性中，“time”为“INT32_MIN”，“tokenId”为“UINT32_MAX”。<br>● “attributes”的长度为count，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetFeatureByIndice接口<a name="ZH-CN_TOPIC_0000002411592888"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels = nullptr, void *features = nullptr, FeatureAttr *attributes = nullptr, ExtraValAttr *extraVal = nullptr) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">按照位置来获取底库特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count：</code></strong>待获取的特征数量。<br><strong><code>const int64_t *indices：</code></strong>待获取的特征在底库中的位置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int64_t *labels：</code></strong>待获取的特征对应的label。<br><strong><code>void *features：</code></strong>待获取的特征向量。<br><strong><code>FeatureAttr *attributes：</code></strong>待获取的特征时空属性。<br><strong><code>ExtraValAttr *extraVal：</code></strong>待获取的特征额外属性。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”单次取值在[1, 1e6]区间。<br>● “indices”长度为count，否则可能出现越界读写错误并引起程序崩溃。值需大于等于0，小于底库数量。<br>● “labels”为“nullptr”时表示不用获取，或者长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “features”为“nullptr”时表示不用获取，或者长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attributes”为“nullptr”时表示不用获取，或者长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “extraVal”为“nullptr”时表示不用获取，或者长度为count，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetFeatureByLabel接口<a name="ZH-CN_TOPIC_0000001507879789"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeatureByLabel(int64_t count, const int64_t *labels, void *features);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取指定Label的特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t count</code></strong>：获取特征的数量。<br><strong><code>const int64_t *labels</code></strong>：特征Label。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>void *features</code></strong>：根据指定Label获取的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat距离为float类型。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 1e6]区间。<br>● “labels”长度为count，各元素不重复且在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## GetFeatureNum接口<a name="ZH-CN_TOPIC_0000001544946953"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetFeatureNum(int64_t *totalNum);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该Index实例中的特征条数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int64_t *totalNum</code></strong>：底库中特征的数量。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Init接口<a id="ZH-CN_TOPIC_0000001458680014"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Init(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, AlgorithmType algType = AlgorithmType::FLAT_COS_INT8, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实例初始化函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t deviceId</code></strong>：Index使用的设备ID。<br><strong><code>uint32_t dim</code></strong>：底库向量的维度。<br><strong><code>uint32_t tokenNum</code></strong>：当前时空库Token的最大数量，需要和生成对应的Mask算子Token数量一致。<br><strong><code>AlgorithmType algType</code></strong>：底层使用的距离比对算法，默认为“AlgorithmType::FLAT_COS_INT8”，可选算法参见如下。<br>● “AlgorithmType::FLAT_HAMMING”：二值化特征检索（汉明距离）。<br>● “AlgorithmType::FLAT_COS_INT8”：Int8Flat（cos距离）。<br>● “AlgorithmType::FLAT_L2_INT8”：Int8Flat（L2距离）。<br>● “AlgorithmType::FLAT_IP_FP16”：FP16Flat（IP距离）。<br>● “AlgorithmType::FLAT_HPP_COS_INT8”：Int8Flat（cos距离）。<br><strong><code>MemoryStrategy memoryStrategy</code></strong>：底层使用的内存策略，默认为“MemoryStrategy::PURE_DEVICE_MEMORY”，可选策略参见如下。<br>● MemoryStrategy::PURE_DEVICE_MEMORY：纯Device内存策略。<br>● MemoryStrategy::HETERO_MEMORY：异构内存策略。<br>● MemoryStrategy::HPP：HPP的异构内存策略。<br><strong><code>customAttrLen</code></strong>：自定义属性长度。<br><strong><code>customAttrBlockSize</code></strong>：自定义属性blocksize的大小。<br><strong><code>maxFeatureRowCount：</code></strong>底库最大向量条数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 需要紧跟在构造函数后调用。<br>● “deviceId”为有效的设备ID，设置范围为[0, 1024]。<br>● “tokenNum”设置范围为(0, 3e5]。<br>● 对于二值化特征检索（汉明距离）算法，dim ∈ {256, 512, 1024}。<br>● 对于Int8Flat（cos距离、L2距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}；对于FP16Flat（IP距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}。<br>● “memoryStrategy::HETERO_MEMORY”当前只支持“AlgorithmType::FLAT_COS_INT8”算法。<br>● “customAttrLen”设置范围为[0, 32]，默认值为“0”，设置为“0”时表示无自定义属性。<br>● “customAttrBlockSize”设置范围为[0, 262144*64]，需要为1024*256的整数倍。默认值为“0”，设置为“0”时表示无自定义属性。<br>● “maxFeatureRowCount”设置范围为[262144 * 64, 262144 * 550 *3]，需要为256的整数倍。默认值为uint64的最大值。该参数只在“MemoryStrategy memoryStrategy”设置为“MemoryStrategy::HPP”时有效。<br>● 当<strong><code>MemoryStrategy memoryStrategy</code></strong>设置为“MemoryStrategy::HPP”时，Host侧的可用内存需要大于等于250GB、空闲CPU物理核数需要大于等于15核，且目前仅支持256维向量的检索。</td></tr>
</tbody></table>

## InitWithExtraVal接口<a id="ZH-CN_TOPIC_0000002013206217"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR InitWithExtraVal(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, AlgorithmType algType = AlgorithmType::FLAT_HAMMING, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实例带附加属性的初始化函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t deviceId</code></strong>：Index使用的设备ID。<br><strong><code>uint32_t dim</code></strong>：底库向量的维度。<br><strong><code>uint32_t tokenNum</code></strong>：当前时空库Token的最大数量，需要和生成对应的Mask算子Token数量一致。<br><strong><code>uint64_t resources</code></strong>：共享内存大小。<br><strong><code>AlgorithmType algType</code></strong>：底层使用的距离比对算法，默认为“AlgorithmType::FLAT_HAMMING”。可选算法参见如下。<br>● “AlgorithmType::FLAT_HAMMING”：二值化特征检索（汉明距离）。<br>● “AlgorithmType::FLAT_COS_INT8”：Int8Flat（cos距离）。<br><strong><code>MemoryStrategy memoryStrategy</code></strong>：底层使用的内存策略，默认为“MemoryStrategy::PURE_DEVICE_MEMORY”，可选策略参见如下。<br>● MemoryStrategy::PURE_DEVICE_MEMORY：纯Device内存策略。<br>● MemoryStrategy::HETERO_MEMORY：异构内存策略。<br><strong><code>customAttrLen</code></strong>：自定义属性长度。<br><strong><code>customAttrBlockSize</code></strong>：自定义属性blocksize的大小。<br><strong><code>maxFeatureRowCount：</code></strong>底库最大向量条数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 需要紧跟在构造函数后调用。<br>● “deviceId”为有效的设备ID，设置范围为[0, 1024]。<br>● “tokenNum”设置范围为(0, 3e5]。<br>● “uint64_t resources”合法范围为[1*1024*1024*1024, 32*1024*1024*1024]，使用附加属性时推荐申请4GB。<br>● 对于二值化特征检索（汉明距离）算法，dim ∈ {256, 512, 1024}。<br>● 对于Int8Flat（cos距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}<br>● “customAttrLen”设置范围为[0, 32]，默认值为“0”，设置为“0”时表示无自定义属性。<br>● “customAttrBlockSize”设置范围为[0, 262144*64]，需要为1024*256的整数倍。默认值为“0”，设置为“0”时表示无自定义属性。<br>● “maxFeatureRowCount”附加属性不支持HPP，默认为uint64的最大值。</td></tr>
</tbody></table>

## InitWithQuantify接口<a name="ZH-CN_TOPIC_0000002458673509"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR InitWithQuantify(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, const float *scale, AlgorithmType algType = AlgorithmType::FLAT_IP_FP16, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">底库向量化初始化接口。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t deviceId：</code></strong>Index使用的设备ID。<br><strong><code>uint32_t dim：</code></strong>底库向量的维度。<br><strong><code>uint32_t tokenNum：</code></strong>当前时空库Token的最大数量，需要和生成对应的Mask算子Token数量一致。<br><strong><code>uint64_t resources：</code></strong>共享内存大小。<br><strong><code>const float *scale：</code></strong>底库向量化缩放因子。缩放因子和底库相乘后转化为int8_t类型。<br><strong><code>AlgorithmType algType：</code></strong>底层使用的距离比对算法。默认为“AlgorithmType::FLAT_IP_FP16”，表示FP16Flat（IP距离），目前仅支持AlgorithmType::FLAT_IP_FP16算法。<br><strong><code>uint32_t customAttrLen：</code></strong>自定义属性长度。<br><strong><code>uint32_t customAttrBlockSize：</code></strong>自定义属性blocksize的大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 需要紧跟在构造函数后调用。<br>● “deviceId”为有效的设备ID，设置范围为[0, 1024]。<br>● “tokenNum”设置范围为(0, 3e5]。<br>● “uint64_t resources”合法范围为大于0小于等于4*1024*1024*1024。<br>● Scale反量化时需要进行除运算，不能接近0；Scale中因子绝对值大于等于1e-6f。<br>● 对于FP16Flat（IP距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}。<br>● 当前只支持FP16Flat（IP距离）算法的非共享模式。<br>● 本接口和AddFeatureByIndice配套使用。<br>● “customAttrLen”设置范围为[0, 32]，默认值为“0”，设置为“0”时表示无自定义属性。<br>● “customAttrBlockSize”设置范围为[0, 262144*64]，需要为1024*256的整数倍。默认值为“0”，设置为“0”时表示无自定义属性。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001507959881"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexTS &amp;operator=(const AscendIndexTS &amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexTS &amp;</code></strong>：常量AscendIndexTS。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Search接口<a name="ZH-CN_TOPIC_0000001507640109"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">计算输入特征和经过AttrFilter过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t count</code></strong>：待比较的特征数量。<br><strong><code>const void *features</code></strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat为float类型。<br><strong><code>const AttrFilter *attrFilter</code></strong>：属性过滤信息，具体请参见<a href="./02_AttrFilter.md#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。<br><strong><code>bool shareAttrFilter</code></strong>：不同query是否共享一个mask。<br><strong><code>uint32_t topk</code></strong>：计算余弦距离后需要保存的TopK大小。<br><strong><code>bool enableTimeFilter</code></strong>：时间戳属性过滤开关，默认为“true”，当<strong><code>enableTimeFilter = false</code></strong>时，不进行时间戳属性的过滤。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int64_t *labels</code></strong>：TopK特征的Label。<br><strong><code>float *distances</code></strong>：TopK特征的距离。<br><strong><code>uint32_t *validNums：</code></strong>每个query向量经过比对后得到的有效结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 10240]区间。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attrFilter”<strong><code>：</code></strong>当<strong><code>shareAttrFilter</code></strong>为true时，数组元素个数为1；当<strong><code>shareAttrFilter</code></strong>为false时，数组元素个数为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “topk”取值在[1, 100000]区间。<br>● “labels”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “validNums”长度为count，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchWithExtraMask接口<a name="ZH-CN_TOPIC_0000001494506850"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">计算输入特征和经过AttrFilter和外部Mask过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t count</code></strong>：待比较的特征数量。<br><strong><code>const void *features</code></strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat为float类型。<br><strong><code>const AttrFilter *attrFilter</code></strong>：属性过滤信息，具体请参见<a href="./02_AttrFilter.md#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。<br><strong><code>bool shareAttrFilter</code></strong>：同一个query是否共享一个Mask。<br><strong><code>uint32_t topk</code></strong>：计算余弦距离后需要保存的TopK大小。<br><strong><code>const uint8_t *extraMask</code></strong>：外部输入的额外的过滤Mask，以bit为单位，0和1分别代表过滤或者选中该条特征。<br><strong><code>uint64_t extraMaskLenEachQuery</code></strong>：外部输入Mask的长度，单位为字节。<br><strong><code>bool extraMaskIsAtDevice</code></strong>：用户外部输入的Mask是否已存在Device侧。<br><strong><code>bool enableTimeFilter</code></strong>：时间戳属性过滤开关，默认为“true”，当<strong><code>enableTimeFilter = false</code></strong>时，不进行时间戳属性的过滤。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int64_t *labels</code></strong>：TopK特征的Label。<br><strong><code>float *distances</code></strong>：TopK特征的距离。<br><strong><code>uint32_t *validNums：</code></strong>每个query向量经过比对后得到的有效结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 10240]区间。<br>● “topk”取值在[1, 100000]区间。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attrFilter”<strong><code>：</code></strong>当<strong><code>shareAttrFilter</code></strong>为true时，数组元素个数为1；当<strong><code>shareAttrFilter</code></strong>为false时，数组元素个数为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “validNums”长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “labels”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “extraMask”：当<strong><code>shareAttrFilter</code></strong>为true时，长度为“extraMaskLenEachQuery”；当<strong><code>shareAttrFilter</code></strong>为false时，长度为count * extraMaskLenEachQuery，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## SearchWithExtraMask带额外相似度接口<a name="ZH-CN_TOPIC_0000002373091106"></a>

<a name="table197013362381"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, const uint16_t *extraScore, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">计算输入特征和经过AttrFilter和外部Mask过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t count</code></strong>：待比较的特征数量。<br><strong><code>const void *features</code></strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat为float类型。<br><strong><code>const AttrFilter *attrFilter</code></strong>：属性过滤信息，具体请参见<a href="./02_AttrFilter.md#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。<br><strong><code>bool shareAttrFilter</code></strong>：同一个query是否共享一个Mask。<br><strong><code>uint32_t topk</code></strong>：计算余弦距离后需要保存的TopK大小。<br><strong><code>const uint8_t *extraMask</code></strong>：外部输入的额外的过滤Mask，以bit为单位，0和1分别代表过滤或者选中该条特征。<br><strong><code>uint64_t extraMaskLenEachQuery</code></strong>：外部输入Mask的长度，单位为字节。<br><strong><code>bool extraMaskIsAtDevice</code></strong>：用户外部输入的Mask是否已存在Device侧。<br><strong><code>const uint16_t *extraScore：</code></strong>用户输入的额外相似度，长度为count*totalPad（totalPad为底库长度按照16对齐的大小）。<br><strong><code>bool enableTimeFilter</code></strong>：时间戳属性过滤开关，默认为“true”，当<strong><code>enableTimeFilter = false</code></strong>时，不进行时间戳属性的过滤。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int64_t *labels</code></strong>：TopK特征的Label。若底库使用AddFeatureByIndice添加，则此处输出底库位置（indices）。<br><strong><code>float *distances</code></strong>：TopK特征的距离。<br><strong><code>uint32_t *validNums：</code></strong>每个query向量经过比对后得到的有效结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 10240]区间。<br>● “topk”取值在[1, 100000]区间。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attrFilter”<strong><code>：</code></strong>当<strong><code>shareAttrFilter</code></strong>为true时，数组元素个数为1；当<strong><code>shareAttrFilter</code></strong>为false时，数组元素个数为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “validNums”长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “labels”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “extraMask”：当<strong><code>shareAttrFilter</code></strong>为true时，长度为“extraMaskLenEachQuery”；当<strong><code>shareAttrFilter</code></strong>为false时，长度为count * extraMaskLenEachQuery，否则可能出现越界读写错误并引起程序崩溃。<br>● “extraScore”：长度为count*totalPad（totalPad为底库长度按照16对齐的大小），否则可能出现越界读写错误并引起程序崩溃。实际对应float16_t类型，值的范围在-1.0到1.0之间。当前仅对Int8FlatCos和FlatIP非共享Mask有效，否则“extraScore”不参与计算。</td></tr>
</tbody></table>

## SearchWithExtraVal接口<a name="ZH-CN_TOPIC_0000002013215285"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchWithExtraVal(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, const ExtraValFilter *extraValFilter, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">计算输入特征和经过AttrFilter和ExtraValFilter过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t count</code></strong>：待比较的特征数量。<br><strong><code>const void *features</code></strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型。当前仅支持Int8Flat（包括异构内存场景）和汉明距离。<br><strong><code>const AttrFilter *attrFilter</code></strong>：属性过滤信息，具体请参见<a href="./02_AttrFilter.md#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。<br><strong><code>bool shareAttrFilter</code></strong>：附加属性暂仅支持“false”，不同query非共享一个mask。<br><strong><code>uint32_t topk</code></strong>：计算余弦距离后需要保存的TopK大小。<br><strong><code>const ExtraValFilter *extraValFilter</code></strong>：附加属性过滤信息，具体请见<a href="./04_ExtraValFilter.md#ZH-CN_TOPIC_0000002013200765">ExtraValFilter</a>。<br><strong><code>bool enableTimeFilter</code></strong>：时间戳属性过滤开关，默认为“true”，当<strong><code>enableTimeFilter = false</code></strong>时，不进行时间戳属性的过滤。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>uint32_t *validNums：</code></strong>每个query向量经过比对后得到的有效结果个数。<br><strong><code>int64_t *labels</code></strong>：TopK特征的Label。<br><strong><code>float *distances</code></strong>：TopK特征的距离。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “count”取值在[1, 10240]区间。<br>● “features”长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。<br>● “attrFilter”<strong><code>：</code></strong>当<strong><code>shareAttrFilter</code></strong>为true时，数组元素个数为1；当<strong><code>shareAttrFilter</code></strong>为false时，数组元素个数为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “topk”取值在[1, 100000]区间。<br>● “labels”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “distances”长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。<br>● “validNums”长度为count，否则可能出现越界读写错误并引起程序崩溃。<br>● “extraValFilter”取值为空指针或者长度为count，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

> [!NOTE]
>
> SearchWithExtraVal不能与Search接口混用。

## SetHeteroParam接口<a name="ZH-CN_TOPIC_0000001630850578"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SetHeteroParam(size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置异构存储策略参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t deviceCapacity</code></strong>：异构内存策略下，Device侧存储底库容量（字节）。<br><strong><code>size_t deviceBuffer</code></strong>：异构内存策略下，Device侧缓存容量（字节）。<br><strong><code>size_t hostCapacity</code></strong>：异构内存策略下，Host侧存储底库容量（字节）。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 该接口需在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>接口设置内存策略为“MemoryStrategy::HETERO_MEMORY”（异构内存策略）后使用。<br>● “deviceCapacity”最小值为1G，最大值为Device实际剩余内存大小。<br>● “deviceBuffer”最小值为2 * 262144 * dim，最大值为“8G”。请根据Device侧实际剩余内存大小进行设置。<br>● <strong><code>deviceCapacity + deviceBuffer</code></strong>应小于Device实际剩余内存大小。<br>● “hostCapacity”取值范围：[1G, 512G]，请根据Host侧实际内存可申请的大小进行配置。</td></tr>
</tbody></table>

## SetSaveHostMemory接口<a name="ZH-CN_TOPIC_0000002106649489"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SetSaveHostMemory();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置使用节约host内存模式，默认不使用。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 该接口需要在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>接口之后，底库为0时使用。<br>● 该接口可以节约host内存，但是会降低删除类型和获取类型接口的性能。<br>● 使用该模式时，无法使用<a href="#ZH-CN_TOPIC_0000001458680018">DeleteFeatureByToken</a>接口。<br>● 该接口只支持汉明距离。</td></tr>
</tbody></table>
