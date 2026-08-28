# AscendIndexIVFRaBitQConfig<a name="ZH-CN_TOPIC_0000002544944511"></a>

AscendIndexIVFRaBitQ需要使用对应的AscendIndexIVFRaBitQConfig执行对应资源的初始化。

## 成员介绍<a name="section4211138173219"></a>

<a name="table388535175015"></a>

<table><tbody>
<tr><td align="center" valign="middle"><strong>成员</strong></td><td width="80" align="center" valign="middle"><strong>类型</strong></td><td align="center" valign="middle"><strong>说明</strong></td></tr>
<tr><td align="center" valign="middle">useRandomOrthogonalMatrix</td><td width="80" align="center" valign="middle">bool</td><td valign="middle">是否使用随机正交矩阵，默认为true。</td></tr>
<tr><td align="center" valign="middle">needRefine</td><td width="80" align="center" valign="middle">bool</td><td valign="middle">是否需要精排，默认为false。</td></tr>
<tr><td align="center" valign="middle">matrixSeed</td><td width="80" align="center" valign="middle">int</td><td valign="middle">生成随机正交矩阵的随机种子，默认为12345。</td></tr>
<tr><td align="center" valign="middle">refineAlpha</td><td width="80" align="center" valign="middle">float</td><td valign="middle">精排相关参数，检索时原本需要检索前k个，需要精排则检索前k * refineAlpha个，再从中取topk。<br>该值默认为2，设置得越大，召回率越高，检索效率越低。</td></tr>
</tbody></table>

## AscendIndexIVFRaBitQConfig<a name="section6579185362314"></a>

>**说明：**
>AscendIndexIVFRaBitQConfig继承于[AscendIndexIVFConfig](../02_approximate_retrieval/04_AscendIndexIVFConfig.md#ascendindexivfconfig)。

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="250" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig();</code></strong></td></tr>
<tr><td width="250" align="center" valign="middle">功能描述</td><td valign="middle">默认构造函数，默认devices为{0}，使用第0个昇腾AI处理器进行计算，默认resource为128MB。</td></tr>
<tr><td width="250" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table3725347611"></a>

<table><tbody>
<tr><td width="250" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="250" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFRaBitQConfig的构造函数，生成AscendIndexIVFRaBitQConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小并执行默认初始化。</td></tr>
<tr><td width="250" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVF_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="250" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>

<a name="table745471811619"></a>

<table><tbody>
<tr><td width="250" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="250" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFRaBitQConfig的构造函数，生成AscendIndexIVFRaBitQConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小并执行默认初始化。</td></tr>
<tr><td width="250" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVF_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="250" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>

<a name="table1037111614358"></a>

<table><tbody>
<tr><td width="250" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, bool useRandomOrthogonalMatrix_, bool needRefine_, int matrixSeed_, float alpha_, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="250" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFRaBitQConfig的构造函数，生成AscendIndexIVFRaBitQConfig，此时根据输入参数执行初始化。</td></tr>
<tr><td width="250" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>bool useRandomOrthogonalMatrix_</code></strong>：是否使用随机正交矩阵。<br><strong><code>bool needRefine_</code></strong>：是否需要精排。<br><strong><code>int matrixSeed_</code></strong>：生成随机正交矩阵的随机种子。<br><strong><code>float alpha_</code></strong>：精排相关参数。<br><strong><code>int resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“IVF_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</td></tr>
<tr><td width="250" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="250" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>
