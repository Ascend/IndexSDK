# AscendIndexFlatConfig<a name="ZH-CN_TOPIC_0000001456375216"></a>

AscendIndexFlat需要使用对应的AscendIndexFlatConfig执行对应资源的初始化。

**接口说明<a name="section140920164419"></a>**

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatConfig的默认构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table46951722104415"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“FLAT_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_L2”时建议设置1024MB。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_INNER_PRODUCT”时建议设置1280MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>

<a name="table842319354444"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“FLAT_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_L2”时建议设置1024MB。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_INNER_PRODUCT”时建议设置1280MB。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>
