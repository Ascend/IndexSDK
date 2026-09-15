# `AscendIndexBinaryFlatConfig`<a name="en-us_TOPIC_0000001506495777"></a>

`AscendIndexBinaryFlat` uses the corresponding `AscendIndexBinaryFlatConfig` to initialize the corresponding resources and configure the device-side hardware resources `devices` and the preset memory pool size `resources` during retrieval.

- `AscendIndexBinaryFlat` supports only Atlas Inference Series products with a single Ascend AI Processor. It depends on the AICPU operator and the BinaryFlat operator. See [Introduction to Custom Operators](../../05_user_guide.md#custom-operator-introduction) to generate the corresponding operators.
- `AscendIndexBinaryFlat` supports only standard deployment mode.

**Members<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="210" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device-side device IDs. The <code>AscendIndexBinaryFlat</code> class supports only a single accelerator card of the Atlas Inference Series products.</td></tr>
<tr><td width="210" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">Size of the device-side memory pool, in bytes. The default value is 1024 MB. The valid range is [1024*1024*1024, 32*1024*1024*1024]. For a base library with 10 million vectors, 5 GB is recommended.</td></tr>
</tbody></table>

**API Description<a name="section108610580175"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlatConfig() = default;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default value of <code>devices</code> is <code>{ 0 }</code>, which uses the 0th Ascend AI Processor for computation. The default value of <code>resources</code> is 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>AscendIndexBinaryFlat</code> supports only Atlas Inference Series products with a single Ascend AI Processor. If the 0th Ascend AI Processor is unavailable, you cannot use the default constructor.</td></tr>
</tbody></table>

<a name="table092314378186"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor that uses <code>initializer_list</code> for <code>devices</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs. For this class, only a single device is supported, that is, the length of <code>devices</code> must be 1.<br><code>int64_t resources</code>: Preset memory pool size. The default value is 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicated device IDs, and the length must be 1. The valid range of <code>resources</code> is [1024*1024*1024, 32*1024*1024*1024]. For a 10 million base library, 5 GB is recommended.</td></tr>
</tbody></table>

<a name="table1743710521181"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlatConfig(std::vector&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor that uses <code>vector</code> for <code>devices</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs. For this class, only a single device is supported, that is, the length of <code>devices</code> must be 1.<br><code>int64_t resources</code>: Preset memory pool size. The default value is 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicated device IDs, and the length must be 1. The valid range of <code>resources</code> is [1024*1024*1024, 32*1024*1024*1024]. For a 10 million base library, 5 GB is recommended.</td></tr>
</tbody></table>
