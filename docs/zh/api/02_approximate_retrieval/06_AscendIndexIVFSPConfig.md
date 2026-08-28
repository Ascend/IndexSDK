# AscendIndexIVFSPConfig<a id="ZH-CN_TOPIC_0000001635696057"></a>

AscendIndexIVFSP需要使用对应的AscendIndexIVFSPConfig执行对应资源的初始化。

**公共参数<a name="section17656114673616"></a>**

<table><tbody>
<tr><td width="140" align="center" valign="middle">参数名</td><td valign="middle">数据类型</td><td valign="middle">参数说明</td></tr>
<tr><td width="140" align="center" valign="middle">handleBatch</td><td valign="middle">int</td><td valign="middle">检索时每次下发计算的候选桶数量，默认值为64。</td></tr>
<tr><td width="140" align="center" valign="middle">nprobe</td><td valign="middle">int</td><td valign="middle">检索时总的候选桶数量，默认值为64。</td></tr>
<tr><td width="140" align="center" valign="middle">searchListSize</td><td valign="middle">int</td><td valign="middle">检索时每次下发计算的每个桶的最大样本数量，默认值为32768。若桶太大，程序会自动根据searchListSize将桶拆成多次算子下发计算距离。</td></tr>
</tbody></table>

**接口说明<a name="section74781713710"></a>**

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFSPConfig();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">默认构造函数，默认devices为{0}，使用第0个昇腾AI处理器进行计算，默认resources为128MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table121971648373"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline explicit AscendIndexIVFSPConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSPConfig构造函数，生成AscendIndexIVFSPConfig，指定Device侧设备ID和资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resources</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVF_SP_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：预置的内存块大小，单位为Byte。默认参数为头文件中的“DEFAULT_BLOCK_SIZE”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，当前仅支持1个NPU设备。<br>● “resources”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>

<a name="table56061252785"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline explicit AscendIndexIVFSPConfig(std::vector&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFSPConfig构造函数，生成AscendIndexIVFSPConfig，指定Device侧设备ID和资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resources</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVF_SP_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。<br><strong><code>uint32_t blockSize</code></strong>：预置的内存块大小，单位为Byte。默认参数为头文件中的“DEFAULT_BLOCK_SIZE”。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，当前仅支持1个NPU设备。<br>● “resources”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</td></tr>
</tbody></table>
