# `AscendIndexInt8Config`<a id="en-us_TOPIC_0000001456854968"></a>

`AscendIndexInt8` requires the corresponding `AscendIndexInt8Config` to initialize the associated resources.

`Member Description`<a name="section1372191465013"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="150" align="center" valign="middle"><code>deviceList</code></td><td valign="middle"><code>std::vector&lt;int&gt;</code></td><td valign="middle">Device-side device ID list.</td></tr>
<tr><td width="150" align="center" valign="middle"><code>resourceSize</code></td><td valign="middle"><code>int64_t</code></td><td valign="middle">Preallocated memory pool size on the device side, in bytes.</td></tr>
</tbody></table>

`API Description`<a name="section135441937164218"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Config()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Default constructor of <code>AscendIndexInt8Config</code>. The default <code>deviceList</code> is <code>0</code>, which means Ascend AI Processor 0 on the NPU is used as the heterogeneous computing platform for AscendFaiss retrieval. The default resource pool size is used.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table012165162914"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Config(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8Config</code>. It creates an <code>AscendIndexInt8Config</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INDEX_INT8_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs, and the maximum number is 64. The configured <code>resources</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

<a name="table9202719152913"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Config(std::vector&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8Config</code>. It creates an <code>AscendIndexInt8Config</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INDEX_INT8_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs, and the maximum number is 64. The configured <code>resources</code> value must not exceed 16 \* 1024 MB (16 \* 1024 \* 1024 \* 1024 bytes).</td></tr>
</tbody></table>
