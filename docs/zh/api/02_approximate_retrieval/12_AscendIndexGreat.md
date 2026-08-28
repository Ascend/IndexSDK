# AscendIndexGreat<a name="ZH-CN_TOPIC_0000002044829945"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002008751966"></a>

自研向量检索算法，为用户提供昇腾侧和鲲鹏侧高维大底库近似检索能力。使用自研检索策略在底库中检索得到topK个最近似向量结果。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

此算法主要针对大底库场景的近似模糊搜索，相较暴力检索精度已有一定损失。在小底库场景，建议适当加大超参值，可改善精度损失问题。

> [!NOTE]
>
>- 创建Index实例时传入的参数params，需根据实际情况设置其中的dim。
>- Index分为两种算法模式：KMode仅使用鲲鹏侧算法，AKMode昇腾加鲲鹏算法，在AKMode模式下需要提前生成对应算子。
>- subSpaceDimnlist应与码本训练时对应参数保持一致。

## AscendIndexGreat接口<a name="ZH-CN_TOPIC_0000002044829953"></a>

<a name="table5404639201712"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexGreat(const std::string&amp; mode, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexGreat的构造函数，创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; mode</code></strong>：指定算法模式。<br><strong><code>const std::vector&lt;int&gt;&amp; deviceList</code></strong>：指定的NPU侧设备ID。<br><strong><code>bool verbose</code></strong>：指定是否开启verbose选项，开启后部分操作提供额外的打印提示。默认值为“false”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● mode：只支持“KMode”和“AKMode”两种模式。<br>● deviceList：请使用<strong><code>npu-smi</code></strong>命令查询对应的NPUID，仅支持一个device设备ID。<br>● 使用此构造函数创建Index实例后，需要先调用“LoadIndex”加载事先落盘后的Index实例，然后再进行其他操作。</td></tr>
</tbody></table>

<a name="table72261454131719"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>explicit AscendIndexGreat(const AscendIndexGreatInitParams&amp; kModeInitParams);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexGreat的构造函数，创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">Index所需的初始化参数kModeInitParams，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>中的参数说明和参数约束。</td></tr>
</tbody></table>

<a name="table198261931819"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexGreat(const AscendIndexVstarInitParams&amp; aModeInitParams, const AscendIndexGreatInitParams&amp; kModeInitParams);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexGreat的构造函数，创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">Index所需的初始化参数aModeInitParams和kModeInitParams，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>和<a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">参考<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>和<a href="./05_AscendIndexIVFSP.md#ascendindexcodebookinitparams接口">AscendIndexGreatInitParams</a>中的参数说明和参数约束。<br>aModeInitParams和kModeInitParams的dim必须保持一致。</td></tr>
</tbody></table>

<a name="table32891532172215"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexGreat(const AscendIndexGreat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexGreat&amp;</code></strong>：常量AscendIndexGreat对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexGreat接口<a name="ZH-CN_TOPIC_0000002013257524"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexGreat() = default;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexGreat的析构函数，销毁AscendIndexGreat对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## operator =接口<a name="ZH-CN_TOPIC_0000002008751990"></a>

<a name="table39961720122213"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexGreat &amp;operator=(const AscendIndexGreat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexGreat&amp;</code></strong>：常量AscendIndexGreat对象。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Add接口<a name="ZH-CN_TOPIC_0000002044950953"></a>

<a name="table11133547191811"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseRawData);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向AscendIndexGreat底库中添加新的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::vector&lt;float&gt;&amp; baseRawData：</code></strong>添加进底库的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处数组“baseRawData”的长度应该为dim * nTotal。nTotal为准备添加进入底库内部的向量数量，dim为每个向量的维度。<br>● 底库向量总数的取值范围：10000 ≤ nTotal ≤ 1e8。<br>● 该算法不支持添加完底库之后再次添加。Add接口不能与AddWithIds接口混用。</td></tr>
</tbody></table>

## AddWithIds接口<a name="ZH-CN_TOPIC_0000002044829957"></a>

<a name="table2436200181918"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddWithIds (const std::vector&lt;float&gt;&amp; baseRawData, const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">向AscendIndexGreat底库中添加新的特征向量。使用AddWithIds接口添加特征，对应特征的默认ids为[0, ntotal)。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">● <strong><code>const std::vector&lt;float&gt;&amp; baseRawData</code></strong>：添加进底库的特征向量。<br>● <strong><code>const std::vector&lt;int64_t&gt;&amp; ids</code></strong>：添加进底库的特征向量ID。ID在Index实例中需唯一。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处数组“baseRawData”的长度应该为dim * nTotal。nTotal为准备添加进入底库内部的向量数量，dim为每个向量的维度。<br>● 底库向量总数的取值范围：10000 ≤ nTotal ≤ 1e8。<br>● “ids”长度必须为nTotal，用户需要根据自己的业务场景，保证“ids”的合法性，如底库中存在重复的ID，检索结果中的&quot;label&quot;将无法对应具体的底库向量。<br>● 该算法不支持添加完底库之后再次添加。AddWithIds接口不能与Add接口混用。</td></tr>
</tbody></table>

## LoadIndex接口<a name="ZH-CN_TOPIC_0000002008751978"></a>

<a name="table17789162191912"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将Index结构从磁盘读入，包括压缩降维后的特征向量和码本数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; indexPath</code></strong>：加载KMode索引的路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“indexPath”对应的文件为调用WriteIndex方法得到的落盘文件，程序执行用户对其有读权限。出于安全加固考虑，目录层级中不能含有软链接。</td></tr>
</tbody></table>

<a name="table98570373191"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和原始数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; aModeIndexPath</code></strong>：加载AMode索引的路径。<br><strong><code>const std::string&amp; kModeIndexPath</code></strong>：加载KMode索引的路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“aModeIndexPath”和“kModeIndexPath”对应的文件为调用WriteIndex方法得到的落盘文件，程序执行用户对其有读权限。出于安全加固考虑，目录层级中不能含有软链接。</td></tr>
</tbody></table>

## WriteIndex接口<a name="ZH-CN_TOPIC_0000002044950957"></a>

<a name="table84194504191"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>const std::string&amp; indexPath</code></strong>：写入KMode索引的路径。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">用户需要保证“indexPath”文件路径所在的目录存在，且执行用户对目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。</td></tr>
</tbody></table>

<a name="table14392122132014"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">● const std::string&amp; aModeIndexPath：写入AMode索引的路径。<br>● const std::string&amp; kModeIndexPath：写入KMode索引的路径。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">用户需要保证“aModeIndexPath”和“kModeIndexPath”文件路径所在的目录存在，且执行用户对目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。</td></tr>
</tbody></table>

## AddCodeBooks接口<a name="ZH-CN_TOPIC_0000002008751982"></a>

<a name="table339181620207"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooks(const std::string&amp; codeBooksPath);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">加载已经生成完毕的码本到Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; codeBooksPath</code></strong>：加载已经生成完毕的码本路径。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">该接口仅能在索引初始化“AKMode”时使用。<br>用户应该保证“codeBooksPath”文件路径所在的目录存在，且该文件内容必须为有效的码本。出于安全加固考虑，目录层级中不能含有软链接。</td></tr>
</tbody></table>

## Search接口<a name="ZH-CN_TOPIC_0000002008910274"></a>

<a name="table537563852013"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Search(const AscendIndexSearchParams&amp; searchParams);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexGreat特征向量查询接口，根据输入的特征向量返回最相似的“topK”条特征的距离及ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">searchParams结构体见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams接口</a>。<br><strong><code>size_t n：</code></strong>查询的特征向量的条数<strong><code>。</code></strong><br><strong><code>std::vector&lt;float&gt;&amp; queryData：</code></strong>特征向量数据<strong><code>。</code></strong><br><strong><code>int topK：</code></strong>需要返回的最相似的结果个数<strong><code>。</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;float&gt;&amp; dists</code></strong>：查询向量与距离最近的前“topK”个向量间的距离值。<br><strong><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● topK ∈ (0, 4096]<br>● <strong><code>n</code></strong>∈ (0, 10000]<br>● queryData不能为空，且数据长度必须大于等于n * dim。<br>● dists不能为空，且数据长度必须大于等于n * topK。<br>● labels不能为空，且数据长度必须大于等于n * topK。</td></tr>
</tbody></table>

## SearchWithMask接口<a name="ZH-CN_TOPIC_0000002044950961"></a>

<a name="table186956182018"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; searchParams, const std::vector&lt;uint8_t&gt;&amp; mask);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">实现AscendIndexGreat特征向量查询接口，根据输入的特征向量返回最相似的“topK”条特征的距离及ID，且用户可以输入一个uint8数组来掩盖特定底库ID，使该ID对应的特征向量不参与检索。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">searchParams结构体见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams接口</a><br><strong><code>size_t n：</code></strong>查询的特征向量的条数。<br><strong><code>std::vector&lt;float&gt;&amp; queryData：</code></strong>特征向量数据。<br><strong><code>int topK：</code></strong>需要返回的最相似的结果个数。<br><strong><code>const std::vector&lt;uint8_t&gt;&amp; mask</code></strong>：外部输入的额外的过滤mask，以bit为单位，0代表过滤该条特征；1代表选中该条特征。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;float&gt;&amp; dists</code></strong>：查询向量与距离最近的前“topK”个向量间的距离值。<br><strong><code>std::vector&lt;int64_t&gt;&amp;</code></strong> <strong><code>labels</code></strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● topK ∈ (0, 4096]<br>● n ∈ (0, 10000]<br>● queryData不能为空，且数据长度必须大于等于n * dim。<br>● dists不能为空，且指向的数据长度必须大于等于n * topK。<br>● labels不能为空，且指向的数据长度必须大于等于n * topK。<br>● mask指向的数据总量必须大于等于n * ceil(nTotal / 8)。</td></tr>
</tbody></table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000002044829965"></a>

<a name="table971712872115"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetNTotal (uint64_t&amp; nTotal) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取AscendIndexGreat已添加进底库的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>uint64_t&amp; nTotal</code></strong>：已添加进底库的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## GetDim接口<a name="ZH-CN_TOPIC_0000002008751986"></a>

<a name="table113422226216"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetDim(int&amp; dim) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取AscendIndexGreat已添加进底库的特征向量的维度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>int&amp; dim</code></strong>：已添加进底库的特征向量的维度。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## Reset接口<a name="ZH-CN_TOPIC_0000002008910278"></a>

<a name="table1974793512118"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR Reset();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">清空该Index数据保存的数据包括压缩降维后的特征向量和码本数据，同时保留用户初始化索引时输入的参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## SetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044950965"></a>

<a name="table1011347192118"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams&amp; params);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置该Index检索时的超参。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexHyperParams&amp; params</code></strong>：检索时的超参，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexhyperparams接口">AscendIndexHyperParams</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## GetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002400547905"></a>

<a name="table749915518225"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>APP_ERROR GetHyperSearchParams(AscendIndexHyperParams&amp; params) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该Index检索时的检索超参。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>AscendIndexHyperParams&amp; params</code></strong>：检索时的超参，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexhyperparams接口">AscendIndexHyperParams</a>。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>APP_ERROR</code></strong>：调用返回状态，具体请参见接口调用返回值参考。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
