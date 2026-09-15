# `AscendIndexConfig`<a name="en-us_TOPIC_0000001506414705"></a>

`AscendIndex` must use the corresponding `AscendIndexConfig` to initialize the relevant resources. `AscendIndexConfig` must configure the hardware resources and memory pool size used during retrieval.

> [!NOTE]
> The memory pool size unit is `Byte`. This parameter specifies the size of the preallocated memory pool on the device side. The memory pool stores the results of distance calculations on Ascend hardware. When the base library is large, you are advised to reserve a larger memory pool.

**Members<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="150" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="150" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device-side device IDs.</td></tr>
<tr><td width="150" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">Device-side memory pool size, in bytes. The default parameter is <code>INDEX_DEFAULT_MEM</code> in the header file.</td></tr>
<tr><td width="150" align="center" valign="middle">slim</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexConfig</code>. Indicates whether to increase memory dynamically.</td></tr>
<tr><td width="150" align="center" valign="middle">filterable</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexConfig</code>. Indicates whether to filter by ID.</td></tr>
<tr><td width="150" align="center" valign="middle">dBlockSize</td><td valign="middle">uint32_t</td><td valign="middle">Device-side block size configuration.</td></tr>
</tbody></table>

**API Description<a name="section1197816229504"></a>**

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexConfig()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Default constructor of <code>AscendIndexConfig</code>. The default <code>deviceList</code> is <code>0</code>, which means that the Ascend AI Processor with ID <code>0</code> on the NPU is used as the heterogeneous computing platform for <code>AscendFaiss</code> retrieval. The default resource-pool size is <code>32 MB</code> (<code>32*1024*1024</code> bytes).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table0786126165110"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexConfig</code>. It creates an <code>AscendIndexConfig</code> and sets device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, while also configuring the resource-pool size.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>INDEX_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Device-side block size configuration. It constrains the amount of data processed in one <code>tik</code> operator call and the size of vectors stored in each partition of the base-library shard. The default value of <code>DEFAULT_BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The maximum number is 64. The configured value of <code>resources</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

<a name="table23967285518"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexConfig(std::vector&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexConfig</code>. It creates an <code>AscendIndexConfig</code> and sets device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, while also configuring the resource-pool size.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>INDEX_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Device-side block size configuration. It constrains the amount of data processed in one <code>tik</code> operator call and the size of vectors stored in each partition of the base-library shard. The default value of <code>DEFAULT_BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The maximum number is 64. The configured value of <code>resources</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>
