# AscendIndexInt8Config<a id="ZH-CN_TOPIC_0000001456854968"></a>

AscendIndexInt8需要使用对应的AscendIndexInt8Config执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="140" align="center" valign="middle">成员</td><td valign="middle">类型</td><td valign="middle">说明</td></tr>
<tr><td width="140" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">设备侧预置的内存池大小，单位为字节。</td></tr>
</tbody></table>

**接口说明<a name="section135441937164218"></a>**

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Config()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Config的默认构造函数，默认指定的deviceList为0（即指定NPU的第0个昇腾AI处理器作为AscendFaiss执行检索的异构计算平台），采用默认的资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table012165162914"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Config(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Config的构造函数，生成AscendIndexInt8Config，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resources</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“INDEX_INT8_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resources”配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>

<a name="table9202719152913"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8Config(std::vector&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8Config的构造函数，生成AscendIndexInt8Config，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resources</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“INDEX_INT8_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resources”配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>
