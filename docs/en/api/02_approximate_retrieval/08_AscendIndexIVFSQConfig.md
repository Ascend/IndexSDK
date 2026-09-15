# `AscendIndexIVFSQConfig`<a id="en-us_TOPIC_0000001456375204"></a>

`AscendIndexIVFSQ` requires the corresponding `AscendIndexIVFSQConfig` to initialize the corresponding resources.

**`AscendIndexIVFSQConfig`<a name="section015013311183"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default <code>devices</code> value is <code>{0}</code>, so the 0th Ascend AI Processor is used for computation. The default <code>resource</code> value is 384 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table19736185071817"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQConfig</code>. It creates an <code>AscendIndexIVFSQConfig</code>, sets the Ascend AI Processor resources on the device side according to the values configured in <code>devices</code>, configures the resource pool size, and performs default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resourceSize:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVFSQ_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The configured <code>resourceSize</code> value must not exceed 10 * 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>

<a name="table1056711401917"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQConfig</code>. It creates an <code>AscendIndexIVFSQConfig</code>, sets the Ascend AI Processor resources on the device side according to the values configured in <code>devices</code>, configures the resource pool size, and performs default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resourceSize:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVFSQ_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The configured <code>resourceSize</code> value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>

**`SetDefaultIVFSQConfig`<a name="section039015215286"></a>**

<a name="table1185313082915"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline void SetDefaultIVFSQConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Perform default initialization. Set the number of iterations to 16 and set a maximum of 512 points for each centroid.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
