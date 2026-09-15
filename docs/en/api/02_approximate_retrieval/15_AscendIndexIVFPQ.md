# `AscendIndexIVFPQ`<a name="en-us_TOPIC_0000002478095516"></a>

## Overview<a name="en-us_TOPIC_0000002510095475"></a>

`AscendIndexIVFPQ` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports only L2 distance and, for performance reasons, only retrieval top-k values within 320.

## `AscendIndexIVFPQ`<a name="en-us_TOPIC_0000002509975505"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFPQ(int dims, faiss::MetricType metric, int nlist, int msubs, int nbits, AscendIndexIVFPQConfig config)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFPQ</code>, which creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimensionality of the base-index retrieval vectors.<br><code>faiss::MetricType metric</code>: Distance type. Currently only <code>faiss::METRIC_L2</code> is supported.<br><code>int nlist</code>: Number of IVF buckets.<br><code>int msubs</code>: Number of subspaces to split into.<br><code>int nbits</code>: Number of bits in the PQ code length. For example, when <code>nbits = 8</code>, the PQ code indices range from 0 to 255.<br><code>AscendIndexIVFPQConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> currently supports only 128. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 262144, 524288}. <code>msubs</code> ∈ {2, 4, 8, 16, 32}. <code>nbits</code> currently supports only 8. <code>config.useKmeansPP</code>: when <code>true</code>, NPU K-Means is used for coarse clustering; when <code>false</code>, CPU clustering is used. For large <code>nlist</code>, use <code>resourceSize</code> ≥ 512 MB (<code>nlist</code>=262144) or ≥ 1 GB (<code>nlist</code>=524288), and training sample count ≥ <code>nlist</code> × 40.</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFPQ&amp; operator=(const AscendIndexIVFPQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this index as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFPQ&amp;</code>: Constant <code>AscendIndexIVFPQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `~AscendIndexIVFPQ`<a name="en-us_TOPIC_0000002477935546"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendIndexIVFPQ()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVFPQ</code>, which destroys the <code>AscendIndexIVFPQ</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000002484264062"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFPQ&amp; operator=(const AscendIndexIVFPQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFPQ&amp;</code>: Constant <code>AscendIndexIVFPQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFPQ</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Setting <code>useKmeansPP</code> to <code>true</code> enables NPU clustering; otherwise CPU clustering is used.</td></tr>
</tbody></table>

## `remove_ids`<a name="en-us_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void remove_ids(size_t n, const idx_t *ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Removes the trained vectors in <code>AscendIndexIVFPQ</code> corresponding to the provided index IDs, by calling the relevant APIs in <code>AscendIndexIVFPQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n</code>: Number of feature vectors to delete.<br><code>const idx_t *ids</code>: IDs of the feature vectors to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## `copyFrom`<a name="en-us_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFPQ *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Reads trained data from the <code>IndexIVFPQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFPQ *index</code>: <code>IVFPQ</code> index, a type of index in the Faiss library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling this API, ensure that the data in <code>index</code> already has trained centroids and an inverted list, and that all parameters are complete.</td></tr>
</tbody></table>

## `copyTo`<a name="en-us_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(const faiss::IndexIVFPQ *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Saves the trained data into the <code>IndexIVFPQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFPQ *index</code>: <code>IVFPQ</code> index, a type of index in the Faiss library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling this API, ensure that the original vectors have been trained and added to the index, so that no empty centroids, codebooks, or inverted lists are read into <code>index</code>.</td></tr>
</tbody></table>

## `update`<a name="en-us_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; update(idx_t n, const float *x, idx_t *ids)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Batch-updates the vectors in the <code>AscendIndexIVFPQ</code> base index corresponding to <code>ids</code> to <code>x</code>. IDs that do not exist in the base index are not updated, and the list of missing IDs is returned.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to update.<br><code>float *x</code>: List of feature vectors to update.<br><code>idx_t *ids</code>: List of feature-vector IDs to update.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>std::vector&lt;idx_t&gt; noExistIds</code>: Returns the list of vector IDs that do not exist.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>
