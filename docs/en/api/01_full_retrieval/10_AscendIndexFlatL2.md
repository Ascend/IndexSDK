# `AscendIndexFlatL2`<a name="en-us_TOPIC_0000001456375424"></a>

## Overview<a name="en-us_TOPIC_0000001877955534"></a>

`AscendIndexFlatL2` is a brute-force feature retrieval algorithm that stores FP16 floating-point values and uses the L2 distance.

It supports multithreaded concurrent calls. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> The `AscendIndexFlatL2` algorithm supports online operator conversion. If the environment variable `MX_INDEX_USE_ONLINEOP` is set to `1` (`export MX_INDEX_USE_ONLINEOP=1`), it converts the operators online and calls them. To use online operators, the user must explicitly call `(void)aclFinalize()` at the end of the application. The header file `#include "acl/acl.h"` is required.

## `AscendIndexFlatL2`<a name="en-us_TOPIC_0000001506495761"></a>

<a name="en-us_TOPIC_0000001294312541_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatL2</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexFlatL2 *index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is <code>{32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 3584, 4096}</code>. The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be <code>faiss::MetricType::METRIC_L2</code>.</td></tr>
</tbody></table>

<a name="en-us_TOPIC_0000001294591937_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatL2</code>. It creates an <code>AscendIndexFlatL2</code> with dimension <code>dims</code>. The dimension of a vector set managed by one <code>Index</code> is unique. It then sets Device-side resources according to the values configured in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: The dimension of a set of feature vectors managed by <code>AscendIndexFlatL2</code>.<br><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 4096, 3584}</td></tr>
</tbody></table>

<a name="en-us_TOPIC_0000001247793230_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2(const AscendIndexFlatL2&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexFlatL2&amp;</code>: A constant <code>AscendIndexFlatL2</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="en-us_TOPIC_0000001294312453_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexFlatL2()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The destructor of <code>AscendIndexFlatL2</code>. It destroys the <code>AscendIndexFlatL2</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `copyFrom`<a name="en-us_TOPIC_0000001456375400"></a>

<a name="en-us_TOPIC_0000001248112146_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies an existing <code>index</code> to Ascend based on <code>AscendIndexFlat</code>, clears the current base library of <code>AscendIndexFlatL2</code>, and keeps the existing Device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is <code>{64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3584}</code>. The value range of the total number of base library vectors is <code>0 &lt;= n &lt; 1e9</code>. The <code>metric_type</code> parameter must be <code>faiss::MetricType::METRIC_L2</code>.</td></tr>
</tbody></table>

## `copyTo`<a name="en-us_TOPIC_0000001456535052"></a>

<a name="en-us_TOPIC_0000001247793178_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexFlatL2</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user must free the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001456695116"></a>

<a name="en-us_TOPIC_0000001294432513_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2&amp; operator=(const AscendIndexFlatL2&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexFlatL2&amp;</code>: A constant <code>AscendIndexFlatL2</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
