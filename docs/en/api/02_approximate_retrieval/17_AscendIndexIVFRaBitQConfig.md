# `AscendIndexIVFRaBitQConfig`<a name="en-us_TOPIC_0000002544944511"></a>

`AscendIndexIVFRaBitQ` must use the corresponding `AscendIndexIVFRaBitQConfig` to initialize the relevant resources.

## `Member Overview`<a name="section4211138173219"></a>

<a name="table388535175015"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="210" align="center" valign="middle">useRandomOrthogonalMatrix</td><td valign="middle">bool</td><td valign="middle">Whether to use a random orthogonal matrix. Default: <code>true</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">needRefine</td><td valign="middle">bool</td><td valign="middle">Whether refinement is required. Default: <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">matrixSeed</td><td valign="middle">int</td><td valign="middle">Random seed used to generate the random orthogonal matrix. Default: 12345.</td></tr>
<tr><td width="210" align="center" valign="middle">refineAlpha</td><td valign="middle">float</td><td valign="middle">Refinement-related parameter. During retrieval, if the original plan is to retrieve the top <code>k</code>, refinement retrieves the top <code>k * refineAlpha</code> results first, and then takes the top <code>k</code> from them.<br>The default value is 2. A larger value gives higher recall but lower retrieval efficiency.</td></tr>
</tbody></table>

## `AscendIndexIVFRaBitQConfig`<a name="section6579185362314"></a>

>`Note:`
>`AscendIndexIVFRaBitQConfig` inherits from [AscendIndexIVFConfig](../02_approximate_retrieval/04_AscendIndexIVFConfig.md#ascendindexivfconfig).

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default <code>devices</code> is <code>{0}</code>. Computation uses the Ascend AI Processor with ID 0, and the default <code>resource</code> is <code>128 MB</code>.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table3725347611"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFRaBitQConfig</code>, which creates an <code>AscendIndexIVFRaBitQConfig</code>. It configures device-side Ascend AI Processor resources according to the values in <code>devices</code>, sets the resource pool size, and performs default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base index size and the <code>search</code> batch size. When the base index is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The maximum number is 64. The configured value of <code>resourceSize</code> must not exceed <code>4 * 1024 MB</code> (<code>4 * 1024 * 1024 * 1024</code> bytes). When set to <code>-1</code>, the device-side Ascend AI Processor resource is configured to the default value of <code>128 MB</code>.</td></tr>
</tbody></table>

<a name="table745471811619"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFRaBitQConfig</code>, which creates an <code>AscendIndexIVFRaBitQConfig</code>. It configures device-side Ascend AI Processor resources according to the values in <code>devices</code>, sets the resource pool size, and performs default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int resourceSize</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base index size and the <code>search</code> batch size. When the base index is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The maximum number is 64. The configured value of <code>resourceSize</code> must not exceed <code>4 * 1024 MB</code> (<code>4 * 1024 * 1024 * 1024</code> bytes). When set to <code>-1</code>, the device-side Ascend AI Processor resource is configured to the default value of <code>128 MB</code>.</td></tr>
</tbody></table>

<a name="table1037111614358"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, bool useRandomOrthogonalMatrix_, bool needRefine_, int matrixSeed_, float alpha_, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFRaBitQConfig</code>, which creates an <code>AscendIndexIVFRaBitQConfig</code>. It performs initialization according to the input parameters.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>bool useRandomOrthogonalMatrix_</code>: Whether to use a random orthogonal matrix.<br><code>bool needRefine_</code>: Whether refinement is required.<br><code>int matrixSeed_</code>: Random seed used to generate the random orthogonal matrix.<br><code>float alpha_</code>: Refinement-related parameter.<br><code>int resourceSize</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base index size and the <code>search</code> batch size. When the base index is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The maximum number is 64. The configured value of <code>resourceSize</code> must not exceed <code>4 * 1024 MB</code> (<code>4 * 1024 * 1024 * 1024</code> bytes). When set to <code>-1</code>, the device-side Ascend AI Processor resource is configured to the default value of <code>128 MB</code>.</td></tr>
</tbody></table>
