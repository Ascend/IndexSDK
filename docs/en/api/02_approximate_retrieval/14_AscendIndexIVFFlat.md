# `AscendIndexIVFFlat`<a name="en-us_TOPIC_0000002478095516"></a>

## Overview<a name="en-us_TOPIC_0000002510095475"></a>

`AscendIndexIVFFlat` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports only IP distance.

## `AscendIndexIVFFlat`<a name="en-us_TOPIC_0000002509975505"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFFlat(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFFlatConfig config)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFFlat</code>, which creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimensionality of the base-index retrieval vectors.<br><code>faiss::MetricType metric</code>: Distance type. Currently only <code>faiss::METRIC_INNER_PRODUCT</code> is supported.<br><code>int nlist</code>: Number of IVF buckets.<br><code>AscendIndexIVFFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> currently supports only 128. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}.</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this index as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFFlat&amp;</code>: Constant <code>AscendIndexIVFFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `~AscendIndexIVFFlat`<a name="en-us_TOPIC_0000002477935546"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendIndexIVFFlat()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVFFlat</code>, which destroys the <code>AscendIndexIVFFlat</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000002484264062"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFFlat&amp;</code>: Constant <code>AscendIndexIVFFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFFlat</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Currently only CPU clustering is supported, and <code>useKmeansPP</code> cannot be set to <code>true</code>.</td></tr>
</tbody></table>
