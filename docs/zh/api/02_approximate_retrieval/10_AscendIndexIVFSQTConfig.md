# AscendIndexIVFSQTConfig<a name="ZH-CN_TOPIC_0000001506495881"></a>

AscendIndexIVFSQT需要使用对应的AscendIndexIVFSQTConfig执行对应资源的初始化。

**AscendIndexIVFSQTConfig<a name="section6579185362314"></a>**

> [!NOTE]
>AscendIndexIVFSQTConfig继承于[AscendIndexIVFSQConfig](./08_AscendIndexIVFSQConfig.md#ascendindexivfsqconfig)。

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">默认构造函数，默认devices为{0}，使用第0个昇腾AI处理器进行计算，默认resource为384MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table42413462115"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQTConfig的构造函数，生成AscendIndexIVFSQTConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小并执行默认的初始化。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVFSQT_DEFAULT_TEMP_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>

<a name="table0812225238"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSQTConfig的构造函数，生成AscendIndexIVFSQTConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小并执行默认的初始化。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVFSQT_DEFAULT_TEMP_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>

**SetDefaultIVFSQConfig<a name="section18396165022414"></a>**

<a name="table14953182017255"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline void SetDefaultIVFSQConfig();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">执行默认的初始化，设置迭代数为16，每个centroids最多设置512个点。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
