
# AscendIndexConfig<a id="ZH-CN_TOPIC_0000001506414705"></a>

AscendIndex需要使用对应的AscendIndexConfig执行对应资源的初始化，AscendIndexConfig中需要配置执行检索过程中的硬件资源和内存池大小等。

> [!NOTE]
>内存池大小单位为**Byte**，此参数用于指定Device侧预留的内存池大小。内存池用于存储昇腾硬件上进行距离计算的结果，底库规模较大时，建议预留更大的内存池大小。

**成员介绍<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="140" align="center" valign="middle">成员</td><td valign="middle">类型</td><td valign="middle">说明</td></tr>
<tr><td width="140" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">Device侧内存池大小，单位为字节，默认参数为头文件中的<strong>INDEX_DEFAULT_MEM</strong>。</td></tr>
<tr><td width="140" align="center" valign="middle">slim</td><td valign="middle">bool</td><td valign="middle">AscendIndexConfig成员变量，是否动态增加内存。</td></tr>
<tr><td width="140" align="center" valign="middle">filterable</td><td valign="middle">bool</td><td valign="middle">AscendIndexConfig成员变量，是否按照id进行过滤。</td></tr>
<tr><td width="140" align="center" valign="middle">dBlockSize</td><td valign="middle">uint32_t</td><td valign="middle">配置Device侧的blockSize。</td></tr>
</tbody></table>

**接口说明<a name="section1197816229504"></a>**

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexConfig()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexConfig默认构造函数，默认指定的deviceList为0（即指定NPU的第0个昇腾AI处理器作为AscendFaiss执行检索的异构计算平台），默认的资源池大小为32MB（32*1024*1024字节）。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table0786126165110"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexConfig的构造函数，生成AscendIndexConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resources</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“INDEX_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值“DEFAULT_BLOCK_SIZE”为16384 * 16 = 262144。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resources”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>

<a name="table23967285518"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexConfig(std::vector&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexConfig的构造函数，生成AscendIndexConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resources</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“INDEX_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值“DEFAULT_BLOCK_SIZE”为16384 * 16 = 262144。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resources”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>
