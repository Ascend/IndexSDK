# `AscendIndexSQConfig`<a name="en-us_TOPIC_0000001456375392"></a>

`AscendIndexSQ` requires the corresponding `AscendIndexSQConfig` to initialize its resources.

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexSQConfig()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The default constructor of <code>AscendIndexSQConfig</code>. The default <code>deviceList</code> is <code>0</code>, which means the first Ascend AI Processor of the NPU is selected as the heterogeneous computing platform for <code>AscendFaiss</code> retrieval. The default resource pool size is used.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table108621239568"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexSQConfig</code>. It creates an <code>AscendIndexSQConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>SQ_DEFAULT_MEM</code> defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>10,000,000</code> and the batch size is greater than or equal to <code>16</code>, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Configures the <code>blockSize</code> on the Device side. It constrains the amount of data processed in a single <code>tik</code> operator execution and the size of vectors stored in each shard of the base library. The default value is <code>16384 * 16 = 262144</code>. This value affects the maximum number of <code>Index</code> objects that can be created and retrieval performance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>. The valid values of <code>blockSize</code> are <code>{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}</code>.</td></tr>
</tbody></table>

<a name="table1735412445711"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexSQConfig</code>. It creates an <code>AscendIndexSQConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>SQ_DEFAULT_MEM</code> defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>10,000,000</code> and the batch size is greater than or equal to <code>16</code>, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Configures the <code>blockSize</code> on the Device side. It constrains the amount of data processed in a single <code>tik</code> operator execution and the size of vectors stored in each shard of the base library. The default value is <code>16384 * 16 = 262144</code>. This value affects the maximum number of <code>Index</code> objects that can be created and retrieval performance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>. The valid values of <code>blockSize</code> are <code>{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}</code>.</td></tr>
</tbody></table>
