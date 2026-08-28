# AscendIndexIVFConfig<a name="ZH-CN_TOPIC_0000001456535024"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456695128"></a>

AscendIndexIVF需要使用对应的AscendIndexIVFConfig执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="140" align="center" valign="middle">成员</td><td valign="middle">类型</td><td valign="middle">说明</td></tr>
<tr><td width="140" align="center" valign="middle">flatConfig</td><td valign="middle">AscendIndexConfig</td><td valign="middle">参数配置对象。</td></tr>
<tr><td width="140" align="center" valign="middle">useKmeansPP</td><td valign="middle">bool</td><td valign="middle">是否使用NPU加速IVF聚类过程。</td></tr>
<tr><td width="140" align="center" valign="middle">cp</td><td valign="middle">ClusteringParameters</td><td valign="middle">聚类相关参数，具体可以参见Faiss相关接口说明。不建议修改此参数，其中训练迭代次数参数默认为16。迭代次数设置过大，会显著增加训练时长。</td></tr>
</tbody></table>

> [!NOTE]
>
> AscendIndexIVFSQConfig继承于[AscendIndexConfig](../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig)。

## AscendIndexIVFConfig接口<a name="ZH-CN_TOPIC_0000001506334629"></a>

<a name="table1319620316150"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFConfig();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">默认构造函数，默认devices为{0}，使用第0个昇腾AI处理器进行计算，默认resources为128MB，默认useKmeansPP为“false”。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table3725347611"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFConfig的构造函数，生成AscendIndexIVFConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小并设置默认迭代数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVF_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>

<a name="table745471811619"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFConfig的构造函数，生成AscendIndexIVFConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小并设置默认迭代数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVF_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>

## SetDefaultClusteringConfig接口<a name="ZH-CN_TOPIC_0000001506495669"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline void SetDefaultClusteringConfig();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">设置此时的AscendIndexIVF的迭代次数为默认值“10”。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
