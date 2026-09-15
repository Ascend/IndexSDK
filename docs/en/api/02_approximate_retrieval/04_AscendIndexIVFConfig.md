# `AscendIndexIVFConfig`<a name="en-us_TOPIC_0000001456535024"></a>

## Overview<a name="en-us_TOPIC_0000001456695128"></a>

`AscendIndexIVF` uses the corresponding `AscendIndexIVFConfig` to initialize the corresponding resources.

**Members<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="210" align="center" valign="middle">flatConfig</td><td valign="middle">AscendIndexConfig</td><td valign="middle">Parameter configuration object.</td></tr>
<tr><td width="210" align="center" valign="middle">useKmeansPP</td><td valign="middle">bool</td><td valign="middle">Whether to use NPU acceleration for the IVF clustering process.</td></tr>
<tr><td width="210" align="center" valign="middle">cp</td><td valign="middle">ClusteringParameters</td><td valign="middle">Clustering-related parameters. For details, see the relevant Faiss API documentation. You are not advised to modify this parameter. The default number of training iterations is 16. Setting the number of iterations too large significantly increases the training time.</td></tr>
</tbody></table>

> [!NOTE]
>
> `AscendIndexIVFConfig` inherits from [AscendIndexConfig](../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig)

## `AscendIndexIVFConfig`<a name="en-us_TOPIC_0000001506334629"></a>

<a name="table1319620316150"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default value of <code>devices</code> is <code>{0}</code>, which uses the 0th Ascend AI Processor for computation. The default value of <code>resources</code> is 128 MB. The default value of <code>useKmeansPP</code> is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table3725347611"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexIVFConfig</code>. It creates <code>AscendIndexIVFConfig</code>, sets the device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, configures the memory pool size, and sets the default number of iterations.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: Preset memory pool size on the device side, in bytes. This memory space stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the search batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicated device IDs, and the maximum number is 64. The configured <code>resourceSize</code> cannot exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When it is set to <code>-1</code>, the device-side Ascend AI Processor resource configuration uses the default value of 128 MB.</td></tr>
</tbody></table>

<a name="table745471811619"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexIVFConfig</code>. It creates <code>AscendIndexIVFConfig</code>, sets the device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, configures the memory pool size, and sets the default number of iterations.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: Preset memory pool size on the device side, in bytes. This memory space stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the search batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicated device IDs, and the maximum number is 64. The configured <code>resourceSize</code> cannot exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When it is set to <code>-1</code>, the device-side Ascend AI Processor resource configuration uses the default value of 128 MB.</td></tr>
</tbody></table>

## `SetDefaultClusteringConfig`<a name="en-us_TOPIC_0000001506495669"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline void SetDefaultClusteringConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the number of iterations for <code>AscendIndexIVF</code> to the default value 10.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
