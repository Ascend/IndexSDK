# AscendIndexMixSearchParams<a name="ZH-CN_TOPIC_0000002008910258"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002045034929"></a>

AscendIndexMixSearchParams.h文件，提供AscendIndexGreat和AscendIndexVStar需要的结构体。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

## AscendIndexGreatInitParams接口<a name="ZH-CN_TOPIC_0000002049404289"></a>

<a name="table17465519101616"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexGreatInitParams();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">KMode模式初始化参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">参数默认值见<a href="#table10419189143817">AscendIndexGreatInitParams</a>。</td></tr>
</tbody></table>

<a id="table10419189143817"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexGreatInitParams(int dim, int degree, int convPQM, int evaluationType, int expandingFactor);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">KMode模式初始化参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">● <strong><code>int dim</code></strong>：特征向量的维度。<br>● <strong><code>int degree</code></strong>：在索引构建阶段控制图索引的精细程度，值越大图索引越精细，占用空间越大，检索时更准确。<br>● <strong><code>int convPQM</code></strong>：PQ量化向量分段数。<br>● <strong><code>int evaluationType</code></strong>：距离评估算法类型，0代表IP，1代表L2。<br>● <strong><code>int expandingFactor</code></strong>：初始构图阶段，连接每一层搜索时邻居的数量。注意与检索阶段的<strong><code>expandingFactor</code></strong>区分。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● dim ∈ {128, 256, 512, 1024}，默认值为“256”。<br>● degree ∈ [50, 100]，默认值为“50”。<br>● convPQM：大于等于16，并且convPQM是8的倍数且能被dim整除，默认值为“128”。<br>● evaluationType ∈ {0，1}，默认值为“0”。<br>● expandingFactor∈ [200, 400]，expandingFactor必须是10的倍数，默认值为“300”。</td></tr>
</tbody></table>

## AscendIndexVstarInitParams接口<a name="ZH-CN_TOPIC_0000002013246410"></a>

<a name="table20955195613391"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexVstarInitParams();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">Vstar模式初始化参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">参数默认值见<a href="#table42921559204019">AscendIndexVstarHyperParams</a>。</td></tr>
</tbody></table>

<a id="table899624214019"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexVstarInitParams(int dim, int subSpaceDim, int nlist, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false, int64_t resourceSize = VSTAR_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">Vstar模式初始化参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dim</code></strong>：特征向量的维度。<br><strong><code>int subSpaceDim</code></strong>：第一次降维后的维度大小。<br><strong><code>int nlist</code></strong>：一级聚类的数量。<br><strong><code>const std::vector&lt;int&gt;&amp; deviceList</code></strong>：指定的NPU physical ID。<br><strong><code>bool verbose</code></strong>：指定是否开启verbose选项，开启后部分操作提供额外的打印提示。默认值为“false”。<br>int64_t resourceSize：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的“VSTAR_DEFAULT_MEM”，大小为128M。该参数通过底库大小和search的batch数共同确定。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">dim ∈ {128, 256, 512, 1024}，默认值为“1024”。<br>subSpaceDim ∈ {32，64，128}。subSpaceDim必须小于dim。默认值为“128”。<br>nlist∈ {256, 512, 1024}。默认值为“1024”。<br>deviceList：请使用<strong><code>npu-smi</code></strong>命令查询对应的NPU卡physical ID，仅支持一个device设备ID。<br>resourceSize ∈ [128M, 2048M]。</td></tr>
</tbody></table>

## AscendIndexVstarHyperParams接口<a name="ZH-CN_TOPIC_0000002013404694"></a>

<a name="table201855541164"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexVstarHyperParams();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">VSTAR模式超参结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">参数默认值见<a href="#table42921559204019">AscendIndexVstarHyperParams</a>。</td></tr>
</tbody></table>

<a id="table42921559204019"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexVstarHyperParams(int nProbeL1, int nProbeL2, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">VSTAR模式超参结构体</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int nProbeL1</code></strong>：一阶段检索搜索的聚类数。<br><strong><code>int nProbeL2</code></strong>：二阶段检索搜索的聚类数。<br><strong><code>int l3SegmentNum</code></strong>：三阶段检索的段数量，从nProbeL2中用于搜索数据段数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● nProbeL1∈ [32, nListL1]，且nProbeL1必须是8的整数倍，默认值为“72”。<br>● nProbeL2∈ (16, nProbeL1 * n]，当dim为1024时n为16，其余维度n为32，且nProbeL2必须是8的整数倍，默认值为“64”。<br>● l3SegmentNum∈ (100, 5000]，且l3SegmentNum必须是8的整数倍。默认值为“512”。</td></tr>
</tbody></table>

## AscendIndexHyperParams接口<a name="ZH-CN_TOPIC_0000002049325253"></a>

<a name="table93967711712"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexHyperParams();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">GREAT检索时的超参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">参数默认值见<a href="#table1334182412417">AscendIndexHyperParams</a>。</td></tr>
</tbody></table>

<a id="table1334182412417"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexHyperParams(const std::string&amp; mode, const AscendIndexVstarHyperParams&amp; vstarHyperParam, int expandingFactor);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">GREAT检索时的超参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; mode</code></strong>：指定算法模式。<br><strong><code>const AscendIndexVstarHyperParams&amp; vstarHyperParam：</code></strong>详细说明请参见<a href="#table42921559204019">AscendIndexVstarHyperParams</a>。<br><strong><code>int expandingFactor</code></strong>：检索阶段每一层搜索时邻居的数量，注意与构图阶段的<strong><code>expandingFactor</code></strong>区分。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● mode∈ {“KMode”,“AKMode”}。默认值“AKMode”。<br>● expandingFactor ∈ [10, 200]。默认值为“150”。</td></tr>
</tbody></table>

<a name="table88027219236"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexHyperParams(const std::string&amp; mode, int expandingFactor);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">GREAT检索时的超参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const std::string&amp; mode</code></strong>：指定算法模式。<br><strong><code>int expandingFactor</code></strong>：检索阶段每一层搜索时邻居的数量，注意与构图阶段的<strong><code>expandingFactor</code></strong>区分。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● mode∈ {“KMode”,“AKMode”}。默认值“AKMode”。<br>● expandingFactor ∈ [10, 200]。默认值为“150”。</td></tr>
</tbody></table>

## AscendIndexSearchParams接口<a name="ZH-CN_TOPIC_0000002044950949"></a>

<a name="table414612258177"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexSearchParams(size_t n, std::vector&lt;float&gt;&amp; queryData, int topK, std::vector&lt;float&gt;&amp; dists, std::vector&lt;int64_t&gt;&amp; labels);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">检索时的搜索参数结构体。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">参数值</td><td valign="middle"><strong><code>size_t n</code></strong>：查询的特征向量的条数。<br><strong><code>std::vector&lt;float&gt;&amp; queryData</code></strong>：特征向量数据。<br><strong><code>int topK：</code></strong>需要返回的最相似的结果个数。<br><strong><code>std::vector&lt;float&gt;&amp; dists</code></strong>：查询向量与距离最近的前“topK”个向量间的距离值。<br><strong><code>std::vector&lt;int64_t&gt;&amp;</code></strong> <strong><code>labels</code></strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</td></tr>
<tr><td width="140" align="center" valign="middle">参数约束</td><td valign="middle">● topK ∈ (0, 4096]。<br>● n ∈ (0, 10000]。<br>● queryData不能为空，且数据长度必须大于等于n * dim。<br>● dists不能为空，且指向的数据长度必须大于等于n * topK。<br>● labels不能为空，且指向的数据长度必须大于等于n * topK。</td></tr>
</tbody></table>
