# AscendIndexBinaryFlatConfig<a name="ZH-CN_TOPIC_0000001506495777"></a>

AscendIndexBinaryFlat需要使用对应的AscendIndexBinaryFlatConfig执行对应资源的初始化，配置执行检索过程中的硬件资源“devices”和预置的内存池大小“resources”。

- AscendIndexBinaryFlat仅支持单个昇腾AI处理器的<term>Atlas 推理系列产品</term>，依赖AICPU算子和BinaryFlat算子，请参考[自定义算子介绍](../../05_user_guide.md#自定义算子介绍)生成对应算子。
- AscendIndexBinaryFlat仅支持标准态部署方式。

**成员介绍<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="140" align="center" valign="middle">成员</td><td valign="middle">类型</td><td valign="middle">说明</td></tr>
<tr><td width="140" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device侧设备ID。AscendIndexBinaryFlat类仅支持单个&lt;term&gt;Atlas 推理系列产品&lt;/term&gt;的加速卡。</td></tr>
<tr><td width="140" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">Device侧内存池大小，单位为字节，默认参数值为1024MB，合法范围为[1024*1024*1024, 32*1024*1024*1024]，10million底库推荐申请5GB。</td></tr>
</tbody></table>

**接口说明<a name="section108610580175"></a>**

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlatConfig() = default;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">默认构造函数，默认devices为{ 0 }，使用第0个昇腾AI处理器进行计算，默认resources为1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">AscendIndexBinaryFlat仅支持单个昇腾AI处理器的Atlas 推理系列产品，如果第0个昇腾AI处理器不可用则无法使用默认构造。</td></tr>
</tbody></table>

<a name="table092314378186"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">“devices”使用initializer_list的构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID，对于该类，仅支持单Device，即“devices”长度为“1”。<br><strong><code>int64_t resources：</code></strong>预置的内存池大小，默认值为1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，长度为1。<br>● “resources”合法范围为[1024*1024*1024, 32*1024*1024*1024]，10million底库推荐申请5GB。</td></tr>
</tbody></table>

<a name="table1743710521181"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexBinaryFlatConfig(std::vector&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">“devices”使用vector的构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID，对于该类，仅支持单Device，即“devices”长度为“1”。<br><strong><code>int64_t resources</code></strong>：预置的内存池大小，默认值为1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，长度为1。<br>● “resources”合法范围为[1024*1024*1024, 32*1024*1024*1024]，10million底库推荐申请5GB。</td></tr>
</tbody></table>
