# `AscendIndexBinaryFlat`<a name="en-us_TOPIC_0000001506334701"></a>

## Overview<a name="en-us_TOPIC_0000001456694988"></a>

The `AscendIndexBinaryFlat` class inherits from Faiss `IndexBinary` and is used for binary feature retrieval.

It supports only Atlas Inference Series products.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `add`<a name="en-us_TOPIC_0000001456854896"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const uint8_t *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds feature vectors to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const uint8_t *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims/8 * n</code>. Otherwise, out-of-bounds reads or writes may occur or the program may crash. <code>n &gt; 0</code>. The <code>add</code> operation must ensure that the final base library size <code>ntotal</code> is the smaller of the actual chip memory capacity and <code>1e9</code>.</td></tr>
</tbody></table>

> [!NOTE]
>
>- The `add` API cannot be used together with the `add_with_ids` API.
>- After you use the `add` API, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` API.

## `add_with_ids`<a name="en-us_TOPIC_0000001506414809"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds feature vectors to the base library and specifies the corresponding IDs.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const uint8_t *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *xids</code>: IDs of the feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 &lt; n</code>. The <code>add</code> operation must ensure that the final base library size <code>n</code> is the smaller of the actual chip memory capacity and <code>1e9</code>. The length of pointer <code>x</code> must be <code>dims/8 * n</code>, and the length of pointer <code>xids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. You need to ensure that <code>xids</code> is valid according to your service scenario. If duplicate IDs exist in the base library, the labels in the search results cannot be mapped to specific base-library vectors.</td></tr>
</tbody></table>

## `AscendIndexBinaryFlat`<a name="en-us_TOPIC_0000001456535056"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(int dims, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexBinaryFlat</code>. It creates an <code>AscendIndexBinaryFlat</code> with dimension <code>dims</code> and sets device-side resources based on the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexBinaryFlat</code>.<br><code>AscendIndexBinaryFlatConfig config</code>: Device-side resource configuration.<br><code>bool usedFloat</code>: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the <code>search</code> API. The default value is <code>false</code>. Set it to <code>true</code> to enable the performance improvement.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ { 256, 512, 1024 }.</td></tr>
</tbody></table>

<a name="table191641015539"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(const faiss::IndexBinaryFlat *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexBinaryFlat</code>. It creates an Ascend retrieval index based on an existing <code>index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexBinaryFlat *index</code>: CPU-side index resource.<br><code>AscendIndexBinaryFlatConfig config</code>: Device-side resource configuration.<br><code>bool usedFloat</code>: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the <code>search</code> API. The default value is <code>false</code>. Set it to <code>true</code> to enable the performance improvement.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU index pointer. <code>index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;ntotal</code> is the smaller of the actual chip memory capacity and <code>1e9</code>.</td></tr>
</tbody></table>

<a name="table142022518319"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(const faiss::IndexBinaryIDMap *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexBinaryFlat</code>. It creates an Ascend retrieval index based on an existing <code>index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexBinaryIDMap *index</code>: CPU-side index resource.<br><code>AscendIndexBinaryFlatConfig config</code>: Device-side resource configuration.<br><code>bool usedFloat</code>: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the <code>search</code> API. The default value is <code>false</code>. Set it to <code>true</code> to enable the performance improvement.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>faiss::IndexBinaryIDMap</code> pointer. <code>index-&gt;index</code> must be a valid <code>IndexBinaryFlat</code> pointer. <code>index-&gt;index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;index-&gt;ntotal</code> is the smaller of the actual chip memory capacity and <code>1e9</code>.</td></tr>
</tbody></table>

<a name="table145324411437"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlat(const AscendIndexBinaryFlat &amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor of <code>AscendIndexBinaryFlat</code> as deleted. Therefore, <code>AscendIndexBinaryFlat</code> is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexBinaryFlat &amp;</code>: Constant <code>AscendIndexBinaryFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `~AscendIndexBinaryFlat`<a name="en-us_TOPIC_0000001506495917"></a>

<a name="table13115573310"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexBinaryFlat() = default;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexBinaryFlat</code>. It destroys the <code>AscendIndexBinaryFlat</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `copyFrom`<a name="en-us_TOPIC_0000001506414941"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexBinaryFlat *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies data from an existing <code>Index</code> to <code>AscendIndexBinaryFlat</code>, clears the current base library of <code>AscendIndexBinaryFlat</code>, and retains the original device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexBinaryFlat *index</code>: <code>faiss::IndexBinaryFlat</code> pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexBinaryFlat</code> pointer. <code>index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;ntotal</code> is the smaller of the actual chip memory capacity and <code>1e9</code>.</td></tr>
</tbody></table>

<a name="table1570816514419"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexBinaryIDMap *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies data from an existing <code>index</code> to <code>AscendIndexBinaryFlat</code>, clears the current base library of <code>AscendIndexBinaryFlat</code>, and retains the original device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexBinaryIDMap *index</code>: <code>faiss::IndexBinaryIDMap</code> pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>faiss::IndexBinaryIDMap</code> pointer. <code>index-&gt;index</code> must be a valid <code>IndexBinaryFlat</code> pointer. <code>index-&gt;index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;index-&gt;ntotal</code> is the smaller of the actual chip memory capacity and <code>1e9</code>.</td></tr>
</tbody></table>

## `copyTo`<a name="en-us_TOPIC_0000001456855048"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexBinaryFlat *index) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies data from an existing <code>AscendIndexBinaryFlat</code> to <code>faiss::IndexBinaryFlat index</code>, and clears the original resources of <code>index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexBinaryFlat *index</code>: <code>faiss::IndexBinaryFlat</code> pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexBinaryFlat</code> pointer. The user must release the resources of the copied <code>index</code>.</td></tr>
</tbody></table>

<a name="table19831553111512"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexBinaryIDMap *index) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies data from an existing <code>AscendIndexBinaryFlat</code> to <code>faiss::IndexBinaryIDMap index</code>, and clears the original resources of <code>index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexBinaryIDMap *index</code>: <code>faiss::IndexBinaryIDMap</code> pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexBinaryIDMap</code> pointer. The user must release the copied <code>Index</code> resources.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001456535072"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlat &amp;operator = (const AscendIndexBinaryFlat &amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment constructor of <code>AscendIndexBinaryFlat</code> as deleted. Therefore, <code>AscendIndexBinaryFlat</code> is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexBinaryFlat &amp;</code>: Constant <code>AscendIndexBinaryFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `remove_ids`<a name="en-us_TOPIC_0000001506495769"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the specified feature vectors from the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to delete. For details about usage and definition, see the relevant Faiss documentation.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Number of feature vectors deleted successfully, with invalid IDs ignored.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reset`<a name="en-us_TOPIC_0000001456855028"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clears the base-library vectors of this <code>AscendIndexBinaryFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `search`<a id="en-us_TOPIC_0000001456375288"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels, const SearchParameters *params) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Feature vector query API. It returns the IDs and corresponding distances of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query vectors.<br><code>const uint8_t *x</code>: Query vectors.<br><code>idx_t k</code>: Number of most similar results to return.<br><code>const SearchParameters *params</code>: Optional Faiss parameters. The default value is <code>nullptr</code>, and this parameter is not supported for now.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>int32_t *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of feature vector data <code>x</code> must be <code>dims/8 * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>0 &lt; n ≤ 1e9</code>, <code>0 &lt; k ≤ 1e5</code>. The <code>n ≤ 1e9</code> limit is far beyond the actual available resources, so you are advised to choose an appropriate number of query vectors according to your service scenario.</td></tr>
</tbody></table>

<a name="table1659211341612"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Feature vector query API. It returns the IDs and corresponding distances of the <code>k</code> most similar features based on the input feature vectors. This API is used for the retrieval mode in which binary features are stored in the base library and float features are used for retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query vectors.<br><code>const float *x</code>: Query vectors.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of feature vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>0 &lt; n ≤ 1e9</code>, <code>0 &lt; k ≤ 1e5</code>. The <code>n ≤ 1e9</code> limit is far beyond the actual available resources, so you are advised to choose an appropriate number of query vectors according to your service scenario.</td></tr>
</tbody></table>

## `setRemoveFast`<a name="en-us_TOPIC_0000002024780673"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>static void setRemoveFast(bool removeFast);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets whether to quickly delete vectors from the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>bool removeFast</code>: Set it to <code>true</code> to use fast deletion, or <code>false</code> not to use it.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Fast deletion improves the performance of deleting the base library, but it slightly reduces the performance of adding data to the base library. If you do not call this API, fast deletion is disabled by default. This API can be called only once, and you must call it before you construct the index object.</td></tr>
</tbody></table>
