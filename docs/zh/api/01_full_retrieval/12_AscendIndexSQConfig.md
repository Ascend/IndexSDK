# AscendIndexSQConfig<a name="ZH-CN_TOPIC_0000001456375392"></a>

AscendIndexSQ需要使用对应的AscendIndexSQConfig执行对应资源的初始化。

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexSQConfig()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexSQConfig的默认构造函数，默认指定的deviceList为0（即指定NPU的第0个昇腾AI处理器作为AscendFaiss执行检索的异构计算平台），采用默认的资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table108621239568"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexSQConfig的构造函数，生成AscendIndexSQConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的“SQ_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size，默认值为16384 * 16 = 262144，该值会影响最大可创建Index的数量与检索的性能。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。<br>● “blockSize”可配置的值的集合为{16384 * 8，16384 * 16，16384 * 32，16384 * 64}。</td></tr>
</tbody></table>

<a name="table1735412445711"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexSQConfig的构造函数，生成AscendIndexSQConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的“SQ_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size，默认值为16384 * 16 = 262144，该值会影响最大可创建Index的数量与检索的性能。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。<br>● “blockSize”可配置的值的集合为{16384 * 8，16384 * 16，16384 * 32，16384 * 64}。</td></tr>
</tbody></table>
