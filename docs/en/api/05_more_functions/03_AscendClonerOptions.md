# `AscendClonerOptions`<a name="en-us_TOPIC_0000001456854804"></a>

## Function Description<a name="en-us_TOPIC_0000001456535196"></a>

Configuration parameters for the `AscendCloner` interface.

**Members<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="210" align="center" valign="middle">reserveVecs</td><td valign="middle">long</td><td valign="middle">Currently unused. Number of features reserved in memory.</td></tr>
<tr><td width="210" align="center" valign="middle">verbose</td><td valign="middle">bool</td><td valign="middle">Whether to print copy logs.</td></tr>
<tr><td width="210" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">Resource pool size.</td></tr>
<tr><td width="210" align="center" valign="middle">slim</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexSQConfig</code>. Whether to dynamically increase memory. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">filterable</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexSQConfig</code>. Whether to filter by ID. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">indexMode</td><td valign="middle">uint32_t</td><td valign="middle">Index INT8 retrieval mode. The default value is <code>0</code> (<code>DEFAULT_MODE</code>).</td></tr>
<tr><td width="210" align="center" valign="middle">blockSize</td><td valign="middle">uint32_t</td><td valign="middle"><code>blockSize</code> configured on the device side. The default value of <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</td></tr>
</tbody></table>

## `AscendClonerOptions`<a name="en-us_TOPIC_0000001506414885"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendClonerOptions()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendClonerOptions</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
