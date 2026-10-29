# `AscendIndexIVFRaBitQ`<a name="en-us_TOPIC_0000002513157720"></a>

## Overview<a name="en-us_TOPIC_0000002544797635"></a>

`AscendIndexIVFRaBitQ` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports L2 distance computation.

## `AscendIndexIVFRaBitQ`<a name="en-us_TOPIC_0000002513317654"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFRaBitQConfig config)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFRaBitQ</code>, which creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimensionality of the base-index retrieval vectors.<br><code>faiss::MetricType metric</code>: Distance type. Supports <code>faiss::METRIC_L2</code> and <code>faiss::METRIC_IP</code>.<br><code>int nlist</code>: Number of IVF buckets.<br><code>AscendIndexIVFRaBitQConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> currently supports only 128. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}.</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFRaBitQ&amp; operator=(const AscendIndexIVFRaBitQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this index as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFRaBitQ&amp;</code>: Constant <code>AscendIndexIVFRaBitQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `~AscendIndexIVFRaBitQ`<a name="en-us_TOPIC_0000002544837623"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendIndexIVFRaBitQ()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVFRaBitQ</code>, which destroys the <code>AscendIndexIVFRaBitQ</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000002513157724"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFRaBitQ&amp; operator=(const AscendIndexIVFRaBitQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFRaBitQ&amp;</code>: Constant <code>AscendIndexIVFRaBitQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000002544797639"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFRaBitQ</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Setting <code>useKmeansPP</code> to <code>true</code> enables NPU clustering; otherwise CPU clustering is used. For precision issues, see <a href="../../07_faq.md#floating-point-computation-precision-issues">floating-point computation precision issues</a>.</td></tr>
</tbody></table>

## `remove_ids`<a name="en-us_TOPIC_0000002513157728"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void remove_ids(size_t n, const idx_t* ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Removes the trained vectors in <code>AscendIndexIVFRaBitQ</code> corresponding to the provided index IDs, by calling the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n</code>: Number of feature vectors to delete.<br><code>const idx_t *ids</code>: IDs of the feature vectors to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## `copyFrom`<a name="en-us_TOPIC_0000002557609263"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFRaBitQ *index)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Provides a CPU-side <code>IndexIVFRaBitQ</code> index, loads data from the trained index to the device side for subsequent retrieval, and calls the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFRaBitQ *index</code>: Trained CPU-side <code>IndexIVFRaBitQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The pointer <code>index</code> must be non-null, and it must point to a trained <code>IndexIVFRaBitQ</code> index. Before calling this API to read data, configure <code>AscendIndexIVFRaBitQConfig</code> and create an <code>AscendIndexIVFRaBitQ</code> object according to the normal procedure.</td></tr>
</tbody></table>

## `copyTo`<a name="en-us_TOPIC_0000002557689209"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFRaBitQ *index) const</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Provides a CPU-side <code>IndexIVFRaBitQ</code> index, downloads the trained data from the device side into the CPU index for persistence, and calls the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFRaBitQ *index</code>: Trained CPU-side <code>IndexIVFRaBitQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The pointer <code>index</code> must be non-null. Before calling this API to persist data, create an <code>AscendIndexIVFRaBitQ</code> object and train it into the index according to the normal procedure.</td></tr>
</tbody></table>

## `search`<a name="en-us_TOPIC_IVFRABITQ_SEARCH"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Searches feature vectors with <code>AscendIndexIVFRaBitQ</code>. <code>SearchParameters.sel</code> applies an <code>IDSelector</code> during the query. On multiple devices, each device filters its local candidates with the same global ID semantics, then the host merges the top-k.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query vectors.<br><code>const float *x</code>: Query vectors, length <code>n * dim</code>.<br><code>idx_t k</code>: Number of nearest results to return.<br><code>const SearchParameters *params</code>: Optional. <code>params-&gt;sel</code> supports <code>IDSelectorRange</code>, <code>IDSelectorBatch</code>, <code>IDSelectorArray</code>, <code>IDSelectorBitmap</code>, <code>faiss::ascend::IDSelectorRoaring</code>, and a single <code>IDSelectorNot</code> of those types. A null <code>params</code> or null <code>sel</code> disables ID filtering. <code>IDSelectorArray.ids</code>, <code>IDSelectorBitmap.bitmap</code>, and the frozen/live pointers of <code>IDSelectorRoaring</code> must stay valid until <code>search</code> returns (bitmap and frozen Roaring are zero-copy). Array/Bitmap/Roaring cache hits use the payload pointer and length, not the contents; rewriting the same buffer in place does not invalidate the cache, so use a new pointer to change the set. <code>IDSelectorBitmap</code> is a dense byte bitmap and is not converted to Roaring. <code>IDSelectorRoaring</code> accepts a CRoaring frozen serialization from the same SDK version (<code>roaring_bitmap_frozen_serialize</code>) or a live <code>roaring_bitmap_t*</code> (frozen on the first miss). When <code>SearchParametersIVF</code> is passed, <code>nprobe</code> applies only to this query (must be <code>&gt; 0</code> and <code>&lt;= nlist</code>) and does not change the index <code>nprobe</code>. A base <code>SearchParameters</code> that only filters still uses the index <code>nprobe</code>. Faiss defaults <code>SearchParametersIVF.nprobe</code> to 1; for filter-only queries keep the base class, or set <code>nprobe</code> to <code>index.getNumProbes()</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distances, length <code>n * k</code>.<br><code>idx_t *labels</code>: IDs, length <code>n * k</code>. Unused slots are <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>IVFRABITQ_SELECTOR_FILTER_STAGE</code> selects the filter stage: <code>auto</code>, <code>pre</code>, <code>post</code>, or <code>both</code>. The default <code>auto</code> uses L2 prefilter on the normal kernel path and post-filter on the SIMT path (Ascend 950). <code>pre</code> filters only in L2 distance; <code>post</code> filters only in AICPU TopK; <code>both</code> runs both stages and is for comparison. The SIMT path does not support L2 prefilter and returns a parameter error when set to <code>pre</code>; use <code>auto</code> or <code>post</code> there. <code>IDSelectorRoaring</code> does not use L2 prefilter and is post-filtered in AICPU TopK under <code>auto</code>. The same selector is applied per device with global ID semantics, then top-k is merged on the host. Array/Bitmap/Roaring query cache hits use the buffer pointer and length and do not scan contents; rewriting ids/bitmap/frozen in place and passing the same pointer reuses the previous filter.</td></tr>
</tbody></table>

## `update`<a name="en-us_TOPIC_0000002566242121"></a>

<a name="table962730101715"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; update(idx_t n, float* x, idx_t* ids)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Batch-updates the vectors in the <code>AscendIndexIVFRaBitQ</code> base index corresponding to <code>ids</code> to <code>x</code>. IDs that do not exist in the base index are not updated, and the list of missing IDs is returned.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to update.<br><code>float* x</code>: List of feature vectors to update.<br><code>idx_t *ids</code>: List of feature-vector IDs to update.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>std::vector&lt;idx_t&gt; noExistIds</code>: Returns the list of vector IDs that do not exist.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>n * dim</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>
