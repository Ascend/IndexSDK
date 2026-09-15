# `AscendIndexFlatConfig`<a name="en-us_TOPIC_0000001456375216"></a>

`AscendIndexFlat` requires the corresponding `AscendIndexFlatConfig` to initialize the corresponding resources.

**API Description**<a name="section140920164419"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The default constructor of <code>AscendIndexFlatConfig</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table46951722104415"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatConfig</code>. It creates an <code>AscendIndexFlatConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>4194304</code> and the batch size is greater than or equal to <code>16</code>, use the following recommendations.<br>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_L2</code>, the recommended value is <code>1024 MB</code>. When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_INNER_PRODUCT</code>, the recommended value is <code>1280 MB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</td></tr>
</tbody></table>

<a name="table842319354444"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatConfig</code>. It creates an <code>AscendIndexFlatConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>4194304</code> and the batch size is greater than or equal to <code>16</code>, use the following recommendations.<br>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_L2</code>, the recommended value is <code>1024 MB</code>. When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_INNER_PRODUCT</code>, the recommended value is <code>1280 MB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</td></tr>
</tbody></table>
