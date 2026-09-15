# `AscendIndexIVFSQTConfig`<a name="en-us_TOPIC_0000001506495881"></a>

`AscendIndexIVFSQT` uses the corresponding `AscendIndexIVFSQTConfig` to initialize the required resources.

**AscendIndexIVFSQTConfig<a name="section6579185362314"></a>**

> [!NOTE]
> `AscendIndexIVFSQTConfig` inherits from [`AscendIndexIVFSQConfig`](./08_AscendIndexIVFSQConfig.md#ascendindexivfsqconfig).

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default <code>devices</code> value is <code>{0}</code>, which uses the 0th Ascend AI Processor for computation. The default <code>resource</code> value is <code>384 MB</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table42413462115"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexIVFSQTConfig</code>. It creates an <code>AscendIndexIVFSQTConfig</code> instance and, based on the values configured in <code>devices</code>, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVFSQT_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to <code>1024 MB</code> when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicate device IDs. The configured value of <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

<a name="table0812225238"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexIVFSQTConfig</code>. It creates an <code>AscendIndexIVFSQTConfig</code> instance and, based on the values configured in <code>devices</code>, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVFSQT_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to <code>1024 MB</code> when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicate device IDs. The configured value of <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

**SetDefaultIVFSQConfig<a name="section18396165022414"></a>**

<a name="table14953182017255"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline void SetDefaultIVFSQConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs the default initialization, sets the number of iterations to 16, and sets a maximum of 512 points for each centroid.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
