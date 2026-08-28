# AscendIndexInt8FlatConfig<a name="ZH-CN_TOPIC_0000001456535040"></a>

AscendIndexInt8Flat需要使用对应的AscendIndexInt8FlatConfig执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="140" align="center" valign="middle">成员</td><td valign="middle">类型</td><td valign="middle">说明</td></tr>
<tr><td width="140" align="center" valign="middle">dIndexMode</td><td valign="middle">Int8IndexMode</td><td valign="middle">配置Index int8检索模式。</td></tr>
<tr><td width="140" align="center" valign="middle">dBlockSize</td><td valign="middle">uint32_t</td><td valign="middle">配置Device侧的blockSize。</td></tr>
</tbody></table>

**接口说明<a name="section136272015172914"></a>**

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8FlatConfig(uint32_t blockSize =BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，配置Device侧blockSize，配置int8的检索模式。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>uint32_t blockSize</code></strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值“BLOCK_SIZE”为16384 * 16 = 262144。<br><strong><code>Int8IndexMode indexMode</code></strong>：配置Index int8检索模式。默认值为<strong><code>DEFAULT_MODE</code></strong>。<br>● <strong><code>DEFAULT_MODE</code></strong>模式，默认模式。<br>● <strong><code>PIPE_SEARCH_MODE</code></strong>模式，该模式针对batch大于或等于<strong><code>128</code></strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB<strong><code>。</code></strong><br>● <strong><code>WITHOUT_NORM_MODE</code></strong>模式，暂时不支持本模式。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “blockSize”可配置的值的集合为{16384， 32768， 65536， 131072， 262144}<br>● <strong><code>indexMode</code></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</td></tr>
</tbody></table>

<a name="table1258103643012"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8FlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，此时根据Devices中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。配置Device侧blockSize，配置int8的检索模式。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“INT8_FLAT_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值“BLOCK_SIZE”为16384 * 16 = 262144。<br><strong><code>Int8IndexMode indexMode</code></strong>：配置Index int8检索模式。默认值为<strong><code>DEFAULT_MODE</code></strong>。<br>● <strong><code>DEFAULT_MODE</code></strong>模式，默认模式。<br>● <strong><code>PIPE_SEARCH_MODE</code></strong>模式，该模式针对batch大于或等于<strong><code>128</code></strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB。<br>● <strong><code>WITHOUT_NORM_MODE</code></strong>模式，暂时不支持本模式。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。当batch大于等于96时，为提升算法性能，建议“resourceSize”设置为大于等于2 * 1024MB。<br>● “blockSize”可配置的值的集合为{16384， 32768， 65536， 131072， 262144}<br>● <strong><code>indexMode</code></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</td></tr>
</tbody></table>

<a name="table8629135217302"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8FlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。配置Device侧blockSize，配置int8的检索模式。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“INT8_FLAT_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值“BLOCK_SIZE”为16384 * 16 = 262144。<br><strong><code>Int8IndexMode indexMode</code></strong>：配置Index int8检索模式。默认值为“DEFAULT_MODE”。<br>● <strong><code>DEFAULT_MODE</code></strong>模式，默认模式。<br>● <strong><code>PIPE_SEARCH_MODE</code></strong>模式，该模式针对batch大于或等于<strong><code>128</code></strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB<strong><code>。</code></strong><br>● <strong><code>WITHOUT_NORM_MODE</code></strong>模式，暂时不支持本模式。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。当batch大于等于96时，为提升算法性能，建议“resourceSize”设置为大于等于2 * 1024MB。<br>● “blockSize”可配置的值的集合为{16384， 32768， 65536， 131072， 262144}。<br>● <strong><code>indexMode</code></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</td></tr>
</tbody></table>
