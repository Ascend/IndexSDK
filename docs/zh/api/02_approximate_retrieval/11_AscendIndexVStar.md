# AscendIndexVStar<a name="ZH-CN_TOPIC_0000002044351677"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002044510693"></a>

昇腾自研向量检索算法，为用户提供昇腾侧高维大底库近似检索能力。使用自研矩阵近似策略，压缩特征向量后存底库，最后使用自研检索策略在底库中检索得到topK个最近似向量结果。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

此算法主要针对大底库场景的近似模糊搜索，相较暴力检索精度已有一定损失。在小底库场景，建议适当加大超参值，可改善精度损失问题。

## AscendIndexVStar接口<a name="ZH-CN_TOPIC_0000002044513265"></a>

> [!NOTE]
>
>- 创建Index实例时传入的参数params，需根据实际情况设置其中的params.dim。
>- params.subSpaceDim和params.nlist应与码本训练时对应参数保持一致。

<a name="table13851535141118"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>explicit AscendIndexVStar(const AscendIndexVstarInitParams&amp; params);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexVStar的构造函数，根据params中配置的值构造对应维度的Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexVstarInitParams&amp; params</code></strong>：构造配置参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>。</td></tr>
</tbody></table>

<a name="table11631734281"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexVStar(const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexVStar的构造函数，根据deviceList构造未知输入数据维度和超参的Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::vector&lt;int&gt;&amp; deviceList</code></strong>：device侧设备ID。<br><strong><code>bool verbose</code></strong>：是否开启“verbose”选项，开启后部分操作提供额外的打印提示。默认值为“false”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “deviceList”需要为合法有效的设备ID，当前仅支持一个device设备。<br>● 使用此构造函数创建Index实例后，需要先调用“LoadIndex”加载事先落盘后的Index实例，然后再进行其他操作。</td></tr>
</tbody></table>

<a name="table8937623141615"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexVStar(const AscendIndexVStar&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexVStar&amp;</code></strong>：AscendIndexVStar对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## LoadIndex接口<a name="ZH-CN_TOPIC_0000002008232688"></a>

<a name="table950712481817"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; indexPath, AscendIndexVStar* indexVStar = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将已有索引Index从磁盘读入Device。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; indexPath</code></strong>：数据文件路径；<br><strong><code>AscendIndexVStar* indexVStar</code></strong>：仅在调用“MultiSearch”接口场景使用，使所有Index共用第一个Index的码本。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 用户应保证“indexPath”文件路径所在的目录存在，且执行用户对目录具有读权限；出于安全加固考虑，目录层级中不能含有软链接。<br>● indexVStar在“MultiSearch”场景下不能为空指针；在单Index场景下必须为空指针，若单Index场景下使用合法Index指针，则原Index码本将被参数Index实例码本替代。</td></tr>
</tbody></table>

## WriteIndex接口<a name="ZH-CN_TOPIC_0000002044351681"></a>

<a name="table29774016915"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将索引index写入磁盘。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; indexPath</code></strong>：保存的数据文件路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 用户应保证“indexPath”文件路径所在的目录存在，且执行用户对目录具有写权限；出于安全加固考虑，目录层级中不能含有软链接。<br>● 当文件已经存在时，将执行覆盖写，此时程序执行用户应该是该文件的属主。</td></tr>
</tbody></table>

## AddCodeBooksByIndex接口<a name="ZH-CN_TOPIC_0000002044510697"></a>

<a name="table81089131197"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooksByIndex(AscendIndexVStar&amp; indexVStar);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">多Index检索场景下，当前Index通过该接口，将传入的参数Index实例的码本载入当前Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>AscendIndexVStar&amp; indexVStar</code></strong>：已填充好码本的Index实例。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">该接口仅在“MultiSearch”场景下使用。</td></tr>
</tbody></table>

## AddCodeBooksByPath接口<a name="ZH-CN_TOPIC_0000002008390980"></a>

<a name="table1523424814919"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooksByPath(const std::string&amp; codeBooksPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">通过码本路径将码本加载到当前Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; codeBooksPath</code></strong>：码本数据文件路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">用户应保证“codeBooksPath”文件路径所在的目录存在，且执行用户对目录具有读权限；出于安全加固考虑，目录层级中不能含有软链接。</td></tr>
</tbody></table>

## Add接口<a name="ZH-CN_TOPIC_0000002008232692"></a>

<a name="table18288921121213"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseData);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexVStar建库和向底库中添加新的特征向量的功能。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::vector&lt;float&gt;&amp; baseData</code></strong>：待添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“baseData”长度应该为n * dim，n为待添加进底库的向量数量，dim为向量维度。n ∈ [10000, 1e8]。<br>该接口不设置ID，底库默认ID范围为[ntotal, ntotal + n)，其中ntotal为Index已有底库数量，n为待添加进底库的向量数量。</td></tr>
</tbody></table>

> [!NOTE]
>
>- Add接口不能与AddWithIds接口混用。
>- 使用Add接口后，Search结果的labels可能会重复，如果业务上对label有要求，建议使用[AddWithIds接口](#addwithids接口)。

## AddWithIds接口<a name="ZH-CN_TOPIC_0000002044351685"></a>

<a name="table32483414124"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddWithIds(const std::vector&lt;float&gt;&amp; baseData, const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexVStar建库和向底库中添加新的特征向量的功能。允许用户指定添加底库的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::vector&lt;float&gt;&amp; baseData</code></strong>：待添加进底库的特征向量。<br><strong><code>const std::vector&lt;int64_t&gt;&amp; ids</code></strong>：待添加底库映射ID的数组。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “baseData”长度应该为n * dim，n为待添加进底库的向量数量，dim为向量维度。<br>● “ids”长度必须为n，用户需要根据自己的业务场景，保证“ids”的合法性，如底库中存在重复的ID，检索结果中的&quot;label&quot;将无法对应具体的底库向量。<br>● n∈[10000，1e8]。</td></tr>
</tbody></table>

## DeleteByIds接口<a name="ZH-CN_TOPIC_0000002044510701"></a>

<a name="table1284884631210"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR DeleteByIds(const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据参数中id数组删除底库中对应id的向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::vector&lt;int64_t&gt;&amp; ids</code></strong>：待删除底库数据的向量ID数组。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">ids中的ID，应为添加底库接口中的ID。</td></tr>
</tbody></table>

## DeleteById接口<a name="ZH-CN_TOPIC_0000002008390984"></a>

<a name="table9845165841212"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR DeleteById(int64_t id);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据参数ID删除底库中对应ID的向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t id</code></strong>：待删除的底库向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">ID应为添加底库接口中的ID。</td></tr>
</tbody></table>

## DeleteByRange接口<a name="ZH-CN_TOPIC_0000002008232696"></a>

<a name="table103969158136"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR DeleteByRange(int64_t startId, int64_t endId);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据参数ID范围删除底库中对应ID的向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int64_t startId</code></strong>：待删除底库的起始ID。<br><strong><code>int64_t endId</code></strong>：待删除底库的结束ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">待删除ID应为添加底库接口中的ID，ID ∈ [startId, endId]</td></tr>
</tbody></table>

## Search接口<a name="ZH-CN_TOPIC_0000002044351689"></a>

<a name="table197566920146"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(const AscendIndexSearchParams&amp; params) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现特征向量检索接口，根据输入的特征向量返回最相似的“topK”条特征的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexSearchParams&amp; params</code></strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a>。<br><strong><code>size_t n</code></strong>：查询的特征向量的条数。<br><strong><code>std::vector&lt;float&gt;&amp; queryData</code></strong>：特征向量数据。<br><strong><code>int topK</code></strong>：需要返回的最相似的结果个数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;float&gt;&amp; dists</code></strong>：查询向量与距离最近的前“topK”个向量间的距离值。<br><strong><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>：查询的距离最近的前“topK”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。<br>● queryData：长度应该大于等于n * dim。<br>● topK∈(0, 4096]。<br>● dists、labels：长度应该大于等于n * topK。</td></tr>
</tbody></table>

## SearchWithMask接口<a name="ZH-CN_TOPIC_0000002044510705"></a>

<a name="table777072291418"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">特征向量查询接口，根据输入的特征向量返回最相似的topK条特征的ID。mask为0、1比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，0表示不参与，1表示参与。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexSearchParams&amp; params</code></strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a><br><strong><code>size_t n</code></strong>：查询的特征向量的条数。<br><strong><code>std::vector&lt;float&gt;&amp; queryData</code></strong>：特征向量数据。<br><strong><code>int topK</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const std::vector&lt;uint8_t&gt;&amp; mask</code></strong>：特征底库掩码。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;float&gt;&amp; dists</code></strong>：查询向量与距离最近的前“topK”个向量间的距离值。<br><strong><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>：查询的距离最近的前“topK”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。<br>● queryData：长度应该大于等于n * dim。<br>● topK∈(0, 4096]。<br>● dists、labels：长度应该大于等于n * topK。<br>● mask：长度应该大于等于n * ceil(ntotal/8)，其中ntotal为底库特征数量。</td></tr>
</tbody></table>

## MultiSearch接口<a name="ZH-CN_TOPIC_0000002008390988"></a>

<a name="table158666394146"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR MultiSearch(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, bool merge) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现从多个AscendIndexVStar库执行特征向量查询的接口，根据输入的特征向量返回最相似的topK条特征距离及ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</code></strong>：待执行检索的多个index。<br><strong><code>const AscendIndexSearchParams&amp; params</code></strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a><br><strong><code>size_t n</code></strong>：查询的特征向量的条数。<br><strong><code>std::vector&lt;float&gt;&amp; queryData</code></strong>：特征向量数据。<br><strong><code>int topK</code></strong>：需要返回的最相似的结果个数。<br><strong><code>bool merge</code></strong>：是否需要合并多个Index上执行检索的结果</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;float&gt;&amp; dists</code></strong>：查询向量与距离最近的前“topK”个向量间的距离值。<br><strong><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>：查询的距离最近的前“topK”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。<br>● queryData：长度应该大于等于n * dim。<br>● topK∈(0, 4096]。<br>● dists、labels满足：当merge = true，长度应该大于等于n * topK。<br>● 当merge = false，长度应该大于等于indexes.size() * n * topK。<br>“indexes”需满足：0 &lt; indexes.size() ≤ 150</td></tr>
</tbody></table>

## MultiSearchWithMask接口<a name="ZH-CN_TOPIC_0000002008232700"></a>

<a name="table141672058131413"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR MultiSearchWithMask(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask, bool merge);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现从多个AscendIndexVStar库执行特征向量查询的接口，根据输入的特征向量返回最相似的topK条特征距离及ID。提供基于mask掩码决定底库是否参与距离计算的功能。mask为0、1比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，“0”表示不参与，“1”表示参与。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</code></strong>：待执行检索的多个index。<br><strong><code>const AscendIndexSearchParams&amp; params</code></strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a>。<br><strong><code>size_t n</code></strong>：查询的特征向量的条数。<br><strong><code>std::vector&lt;float&gt;&amp; queryData</code></strong>：特征向量数据。<br><strong><code>int topK</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const std::vector&lt;uint8_t&gt;&amp; mask</code></strong>：特征底库掩码。<br><strong><code>bool merge</code></strong>：是否需要合并多个Index上执行检索的结果。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;float&gt;&amp; dists</code></strong>：查询向量与距离最近的前“topK”个向量间的距离值。<br><strong><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>：查询的距离最近的前“topK”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。<br>● queryData：长度应该大于等于n*dim。<br>● topK∈(0, 4096]。<br>● dists、labels满足：当merge = true，长度应该大于等于n * topK。<br>● 当merge = false，长度应该大于等于indexes.size() * n * topK。<br>mask：长度应该大于等于n * ceil(ntotal_max/8)，其中ntotal_max为底库特征数量，为所有Index中最大的底库数量值。“indexes”需满足：0 &lt; indexes.size() ≤ 150</td></tr>
</tbody></table>

## SetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044351693"></a>

<a name="table4215111781514"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams&amp; params);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置AscendIndexVstar实例检索时的超参。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexVstarHyperParams&amp; params</code></strong>：检索时超参，具体请见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarhyperparams接口">AscendIndexVstarHyperParams</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● nProbeL1∈(16，nListL1], nProbeL1 % 8 == 0<br>● nProbeL2∈(16, nProbeL1 * nList2], nProbeL2 % 8 == 0<br>● l3SegmentNum∈(100,5000], l3SegmentNum % 8 == 0</td></tr>
</tbody></table>

## GetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044510709"></a>

<a name="table5860202961515"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams&amp; params) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取向量检索时的超参。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>AscendIndexVstarHyperParams&amp; params</code></strong>：检索时超参，具体请见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarhyperparams接口">AscendIndexVstarHyperParams</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## GetDim接口<a name="ZH-CN_TOPIC_0000002008390992"></a>

<a name="table6661184351519"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetDim(int&amp; dim) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取初始化索引时的维度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int&amp; dim</code></strong>：Index的维度。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000002008232704"></a>

<a name="table1919613597154"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetNTotal(uint64_t&amp; ntotal) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取当前索引的底库数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>uint64_t&amp; ntotal</code></strong>：当前Index的底库总向量条数。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Reset接口<a name="ZH-CN_TOPIC_0000002044351697"></a>

<a name="table19794117167"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Reset();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">重置索引接口，清除保存的索引数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">重置索引后，会保留用户初始化索引时输入的参数。</td></tr>
</tbody></table>

## operator= 接口<a name="ZH-CN_TOPIC_0000002008390996"></a>

<a name="table3792193711620"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexVStar&amp; operator=(const AscendIndexVStar&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexVStar&amp;</code></strong>：AscendIndexVStar对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
