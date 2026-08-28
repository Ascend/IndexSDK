# Approximate Retrieval<a name="ZH-CN_TOPIC_0000001482524834"></a>

## `AscendIndexBinaryFlat`<a name="ZH-CN_TOPIC_0000001506334701"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456694988"></a>

The `AscendIndexBinaryFlat` class inherits from Faiss `IndexBinary` and is used for binary feature retrieval.

It supports only Atlas Inference Series products.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `add`<a name="ZH-CN_TOPIC_0000001456854896"></a>

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

### `add_with_ids`<a name="ZH-CN_TOPIC_0000001506414809"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds feature vectors to the base library and specifies the corresponding IDs.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const uint8_t *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *xids</code>: IDs of the feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 &lt; n</code>. The <code>add</code> operation must ensure that the final base library size <code>n</code> is the smaller of the actual chip memory capacity and <code>1e9</code>. The length of pointer <code>x</code> must be <code>dims/8 * n</code>, and the length of pointer <code>xids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. You need to ensure that <code>xids</code> is valid according to your service scenario. If duplicate IDs exist in the base library, the labels in the search results cannot be mapped to specific base-library vectors.</td></tr>
</tbody></table>

### `AscendIndexBinaryFlat`<a name="ZH-CN_TOPIC_0000001456535056"></a>

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

### `~AscendIndexBinaryFlat`<a name="ZH-CN_TOPIC_0000001506495917"></a>

<a name="table13115573310"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexBinaryFlat() = default;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexBinaryFlat</code>. It destroys the <code>AscendIndexBinaryFlat</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001506414941"></a>

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

### `copyTo`<a name="ZH-CN_TOPIC_0000001456855048"></a>

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

### `operator=`<a name="ZH-CN_TOPIC_0000001456535072"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexBinaryFlat &amp;operator = (const AscendIndexBinaryFlat &amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment constructor of <code>AscendIndexBinaryFlat</code> as deleted. Therefore, <code>AscendIndexBinaryFlat</code> is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexBinaryFlat &amp;</code>: Constant <code>AscendIndexBinaryFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `remove_ids`<a name="ZH-CN_TOPIC_0000001506495769"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the specified feature vectors from the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to delete. For details about usage and definition, see the relevant Faiss documentation.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Number of feature vectors deleted successfully, with invalid IDs ignored.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reset`<a name="ZH-CN_TOPIC_0000001456855028"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clears the base-library vectors of this <code>AscendIndexBinaryFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `search`<a id="ZH-CN_TOPIC_0000001456375288"></a>

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

### `setRemoveFast`<a name="ZH-CN_TOPIC_0000002024780673"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>static void setRemoveFast(bool removeFast);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets whether to quickly delete vectors from the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>bool removeFast</code>: Set it to <code>true</code> to use fast deletion, or <code>false</code> not to use it.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Fast deletion improves the performance of deleting the base library, but it slightly reduces the performance of adding data to the base library. If you do not call this API, fast deletion is disabled by default. This API can be called only once, and you must call it before you construct the index object.</td></tr>
</tbody></table>

## `AscendIndexBinaryFlatConfig`<a name="ZH-CN_TOPIC_0000001506495777"></a>

`AscendIndexBinaryFlat` uses the corresponding `AscendIndexBinaryFlatConfig` to initialize the corresponding resources and configure the device-side hardware resources `devices` and the preset memory pool size `resources` during retrieval.

- `AscendIndexBinaryFlat` supports only Atlas Inference Series products with a single Ascend AI Processor. It depends on the AICPU operator and the BinaryFlat operator. See [Introduction to Custom Operators](../user_guide.md#generating-operators) to generate the corresponding operators.
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

## `AscendIndexIVF`<a name="ZH-CN_TOPIC_0000001456375220"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506334721"></a>

`AscendIndexIVF` serves as the base class of IVF-based indexes in the feature retrieval component and defines APIs for other IVF indexes in feature retrieval.

For IVF algorithms, the linear scaling on the Atlas 300I Duo inference card depends on the proportion of distance-computation workload in the entire search process. Compared with other computation types, only the distance-computation workload can be evenly distributed across multiple compute units. Therefore, scaling is better in large-batch and large-`nprobe` scenarios, and worse in small-batch and small-`nprobe` scenarios.

> [!NOTE]
> IVF algorithms should follow the rule `nlist * 2MB + resourceSize < NPU-side memory` to avoid memory allocation failures at runtime. For example, if the memory on the NPU card is 64 GB, `nlist` should be smaller than 32768. Since `32768 * 2MB = 64GB`, runtime may exceed the NPU memory size. This limit exists because the current retrieval service prioritizes large-page memory, and the allocation granularity of large-page memory is 2 MB. When every bucket in `nlist` contains data, the hardware allocates memory aligned to the 2 MB granularity. `resourceSize` is the shared memory size specified by the user in `AscendIndexIVFConfig`, and the default value is 128 MB.

### `AscendIndexIVF`<a name="ZH-CN_TOPIC_0000001506414821"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVF(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFConfig config = AscendIndexIVFConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexIVF</code>. It creates <code>AscendIndexIVF</code> and sets device-side resources based on the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexIVF</code>.<br><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. The current supported values are <code>faiss::MetricType::METRIC_L2</code> and <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.<br><code>int nlist</code>: Number of clustering centers. This corresponds to the <code>coarse_centroid_num</code> parameter in the operator generation script.<br><code>AscendIndexIVFConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}.</td></tr>
</tbody></table>

<a name="table9624174810199"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVF(const AscendIndexIVF&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor of this index as deleted. Therefore, it is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVF&amp;</code>: Constant <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~AscendIndexIVF`<a name="ZH-CN_TOPIC_0000001506334765"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexIVF();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVF</code>. It destroys the <code>AscendIndexIVF</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001506334601"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVF* index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle"><code>AscendIndexIVF</code> copies data from an existing <code>index</code> to Ascend and retains the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVF* index</code>: CPU-side index resource.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The <code>probe</code> value of this <code>index</code> must be greater than 0 and less than or equal to <code>nlist</code>.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000001506615113"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVF* index) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexIVF</code> to the CPU side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIVF* index</code>: CPU-side index resource.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The resources occupied by <code>Index</code> are released by the user.</td></tr>
</tbody></table>

### `getNumLists`<a name="ZH-CN_TOPIC_0000001506614893"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getNumLists() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the current <code>nlist</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>nlist</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getNumProbes`<a name="ZH-CN_TOPIC_0000001456534948"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getNumProbes() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the current <code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getListCodesAndIds`<a name="ZH-CN_TOPIC_0000001456854940"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the feature vectors and corresponding IDs at a specific <code>nlistId</code> in the current <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId</code>: Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;uint8_t&gt;&amp; codes</code>: Feature vectors at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.<br><code>std::vector&lt;ascend_idx_t&gt;&amp; ids</code>: IDs of the feature vectors at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 ≤ listId &lt; nlist</code>.</td></tr>
</tbody></table>

### `getListLength`<a name="ZH-CN_TOPIC_0000001506614973"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual uint32_t getListLength(int listId) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the length of a specific <code>nlistId</code> in the current <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId</code>: Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Length of the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 ≤ listId &lt; nlist</code>.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001506495837"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVF&amp; operator=(const AscendIndexIVF&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment constructor of this index as deleted. Therefore, it is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVF&amp;</code>: Constant <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reclaimMemory`<a name="ZH-CN_TOPIC_0000001506615049"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t reclaimMemory() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Reduces the memory occupied by the base library without changing the number of base-library entries. This API inherits from <code>AscendIndex</code> and provides a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Amount of memory reduced, in bytes.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reserveMemory`<a name="ZH-CN_TOPIC_0000001506334617"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reserveMemory(size_t numVecs) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Abstract API that reserves memory for the base library before the base library is built. This API inherits from <code>AscendIndex</code> and provides a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t numVecs</code>: Number of base-library vectors for which to reserve memory.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">In a single-card environment: <code>0 &lt; numVecs ≤ 2e8</code>. In a multi-card environment: <code>0 &lt; numVecs ≤ 1e9</code> (<code>numVecs</code> divided by the number of cards must be smaller than <code>2e8</code>). Exceeding the limit throws an exception and stops the program.</td></tr>
</tbody></table>

### `reset`<a name="ZH-CN_TOPIC_0000001506414685"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clears the base-library vectors of this <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `setNumProbes`<a name="ZH-CN_TOPIC_0000001506614937"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void setNumProbes(int nprobes);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the current <code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobes</code>: <code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 &lt; nprobes ≤ nlist</code>.</td></tr>
</tbody></table>

## `AscendIndexIVFConfig`<a name="ZH-CN_TOPIC_0000001456535024"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456695128"></a>

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
> `AscendIndexIVFConfig` inherits from [AscendIndexConfig](./full_retrieval.md#ascendindexconfig)

### `AscendIndexIVFConfig`<a name="ZH-CN_TOPIC_0000001506334629"></a>

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

### `SetDefaultClusteringConfig`<a name="ZH-CN_TOPIC_0000001506495669"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline void SetDefaultClusteringConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the number of iterations for <code>AscendIndexIVF</code> to the default value 10.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendIndexIVFSP`<a name="ZH-CN_TOPIC_0000001635576081"></a>

### Overview<a name="ZH-CN_TOPIC_0000001635815481"></a>

The Ascend-native IVFSP retrieval algorithm uses an in-house matrix approximation strategy to compress feature vectors before storing them in the base library. It then uses an in-house inverted-list strategy to select the base-library entries most likely to contain the ground truth. Finally, it uses an in-house retrieval strategy on the filtered base library to obtain the top K vector results.

`AscendIndexIVFSP` supports only standard mode scenarios and Atlas Inference Series products.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `add`<a name="ZH-CN_TOPIC_0000001585895568"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds feature vectors to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const float *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The total number of base-library vectors, <code>n</code>, is usually greater than 0 and less than <code>1e9</code>. The amount of data added at one time must be smaller than or equal to the base-library data size.</td></tr>
</tbody></table>

> [!NOTE]
>
>- The `add` API cannot be used together with the `add_with_ids` API.
>- After you use the `add` API, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` API.
>- The `add` API is optimized for small-batch addition scenarios. In this scenario, accuracy may decrease depending on the dataset. You are advised to use small-batch addition when a base library already exists.

### `add_with_ids`<a name="ZH-CN_TOPIC_0000001586055512"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds feature vectors to the base library and specifies the corresponding IDs.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const float *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. You need to ensure that <code>ids</code> is valid according to your service scenario. If duplicate IDs exist in the base library, the <code>label</code> in the retrieval results cannot be mapped to a specific base-library vector.<br>The value range of <code>n</code> is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

> [!NOTE]
> The `add_with_ids` API is optimized for small-batch addition scenarios. In this scenario, accuracy may decrease depending on the dataset. You are advised to use small-batch addition when a base library already exists.

### `AscendIndexIVFSP`<a name="ZH-CN_TOPIC_0000001585736168"></a>

> [!NOTE]
>
> - Before you pass parameter `config` to the function, set the values of `conf.handleBatch`, `conf.nprobe`, and `conf.searchListSize` according to the actual situation. For field descriptions, see [Common Parameters](#ZH-CN_TOPIC_0000001635696057).
> - The values of `conf.handleBatch` and `conf.searchListSize` must be consistent with the `nprobe handle batch` and `search list size` values used when generating the [IVFSP](../user_guide.md#ivfsp) service operator model file.
> - `conf.filterable`, inherited from [AscendIndexConfig](./full_retrieval.md#ascendindexconfig) false by default. If you want to use the `search_with_filter()` API, set `conf.filterable = true`. Setting `conf.filterable` to `true` stores extra information on the NPU card and consumes more NPU-side memory.

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexIVFSP</code>. It sets device-side resources based on the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexIVFSP</code>.<br><code>int nonzeroNum</code>: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.<br><code>int nlist</code>: Number of clustering centers. This corresponds to the value of the <code>&lt;centroid num&gt;</code> parameter in the generation of the IVFSP service operator model file.<br><code>const char *codeBookPath</code>: Path of the codebook file used by IVFSP.<br><code>faiss::ScalarQuantizer::QuantizerType qType</code>: Scalar quantization type. The current supported value is only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.<br><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. The current <code>faiss::MetricType metric</code> supports only <code>METRIC_L2</code>.<br><code>AscendIndexIVFSPConfig</code>: Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The values of <code>&lt;dim&gt;</code>, <code>&lt;nonzero num&gt;</code>, and <code>&lt;centroid num&gt;</code> used when training and generating the codebook must correspond to the <code>dims</code>, <code>nonzeroNum</code>, and <code>nlist</code> parameters of this function. The codebook loaded from <code>codeBookPath</code> must correspond to the <code>dims</code>, <code>nonzeroNum</code>, and <code>nlist</code> parameters of this function, and the user who runs the program must be the owner of the codebook file. The codebook file cannot be a symbolic link. When <code>dims</code> ∈ {64, 128, 256}, <code>nlist</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dims</code> ∈ {512, 768}, <code>nlist</code> ∈ {256, 512, 1024, 2048}. <code>nonzeroNum</code> must be a multiple of 16 and less than or equal to <code>min(128, dims)</code>. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>}.</td></tr>
</tbody></table>

<a name="table49022324218"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &amp;codeBookSharedIdx, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexIVFSP</code>. It sets device-side resources based on the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexIVFSP</code>.<br><code>int nonzeroNum</code>: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.<br><code>int nlist</code>: Number of clustering centers. This corresponds to the value of the <code>&lt;centroid num&gt;</code> parameter in the generation of the IVFSP service operator model file.<br><code>const AscendIndexIVFSP &amp;codeBookSharedIdx</code>: <code>AscendIndexIVFSP</code> object that shares the codebook.<br><code>faiss::ScalarQuantizer::QuantizerType qType</code>: Scalar quantization type. The current supported value is only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.<br><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. The current <code>faiss::MetricType metric</code> supports only <code>METRIC_L2</code>.<br><code>AscendIndexIVFSPConfig</code>: Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The values of <code>&lt;dim&gt;</code>, <code>&lt;nonzero num&gt;</code>, and <code>&lt;centroid num&gt;</code> used when training and generating the codebook must correspond to the <code>dims</code>, <code>nonzeroNum</code>, and <code>nlist</code> parameters of this function. The shared codebook configuration of <code>codeBookSharedIdx</code> must match the codebook configuration of the current index, and the device resources must also match. When <code>dims</code> ∈ {64, 128, 256}, <code>nlist</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dims</code> ∈ {512, 768}, <code>nlist</code> ∈ {256, 512, 1024, 2048}. <code>nonzeroNum</code> must be a multiple of 16 and less than or equal to <code>min(128, dims)</code>. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>}.</td></tr>
</tbody></table>

<a name="table8581162710235"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSP(const AscendIndexIVFSP&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor of this index as deleted. Therefore, it is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSP&amp;</code>: Constant <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table186918413239"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSP();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVFSP</code>. It destroys the <code>AscendIndexIVFSP</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table241282321712"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexIVFSP</code>. It sets device-side resources based on the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexIVFSP</code>.<br><code>int nonzeroNum</code>: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.<br><code>int nlist</code>: Number of clustering centers. This corresponds to the value of the <code>&lt;centroid num&gt;</code> parameter in the generation of the IVFSP service operator model file.<br><code>faiss::ScalarQuantizer::QuantizerType qType</code>: Scalar quantization type. The current supported value is only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.<br><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. The current <code>faiss::MetricType metric</code> supports only <code>METRIC_L2</code>.<br><code>AscendIndexIVFSPConfig</code>: Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">When <code>dims</code> ∈ {64, 128, 256}, <code>nlist</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dims</code> ∈ {512, 768}, <code>nlist</code> ∈ {256, 512, 1024, 2048}. <code>nonzeroNum</code> must be a multiple of 16 and less than or equal to <code>min(128, dims)</code>. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>}.</td></tr>
</tbody></table>

### `loadAllData API`<a id="ZH-CN_TOPIC_0000001585736172"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void loadAllData(const char *dataPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Load the <code>Index</code> structure from disk into the <code>Device</code>, including the compressed, reduced-dimensional feature vectors and the codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const char *dataPath:</code> Path to the data file.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The file corresponding to <code>dataPath</code> should be the file written by <code>saveAllData</code>, and the process user must have read permission for it. The file must not be a symbolic link.<br>This API does not support codebook sharing. If you need codebook sharing, you are advised to use the <code>loadAllData</code> overload that accepts <code>codeBookSharedIdx</code>.</td></tr>
</tbody></table>

<a name="table115591219131513"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>static std::shared_ptr&lt;AscendIndexIVFSP&gt; loadAllData(const AscendIndexIVFSPConfig &amp;config, const uint8_t *data, size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Restore an <code>AscendIndexIVFSP</code> object from memory.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSPConfig &amp;config:</code> Device-side resource configuration. Currently, you only need to set <code>config.deviceList</code> and <code>config.resourceSize</code>. The other configuration parameters are restored from memory. <code>const uint8_t *data:</code> Memory pointer obtained by <code>saveAllData</code>. <code>size_t dataLen:</code> Actual length of the <code>data</code> pointer. <code>const AscendIndexIVFSP *codeBookSharedIdx:</code> Pointer to the <code>AscendIndexIVFSP</code> that shares the codebook. The default value is <code>nullptr</code>, which means that the codebook is not shared.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A smart pointer to the <code>AscendIndexIVFSP</code> object restored from memory.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>data</code> must be a non-null valid pointer. <code>dataLen</code> must be the actual length of the <code>data</code> pointer. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The codebook configuration of the shared <code>codeBookSharedIdx</code> must match the codebook configuration of the current <code>Index</code>, and the device resources configuration must also match.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001635975413"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSP&amp; operator=(const AscendIndexIVFSP&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSP&amp;:</code> A constant <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `remove_ids`<a name="ZH-CN_TOPIC_0000001635576085"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implement the API for deleting the specified feature vectors from the base vector set in <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel:</code> Feature vectors to delete. For details about the usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">The number of deleted feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reset`<a name="ZH-CN_TOPIC_0000001635815485"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clear the base vectors in this <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `saveAllData`<a name="ZH-CN_TOPIC_0000001635696053"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void saveAllData(const char *dataPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Write the <code>Index</code> structure from the <code>Device</code> side to disk. The data written to disk includes the compressed, reduced-dimensional feature vectors and the codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const char *dataPath:</code> Path to the output data file.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the directory containing the <code>dataPath</code> file exists, and that the process user has write permission for the directory. For security hardening, the directory hierarchy must not contain symbolic links.<br>When the file corresponding to <code>dataPath</code> already exists, the file is overwritten. In this case, the process user should be the file owner.</td></tr>
</tbody></table>

<a name="table11876949141314"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void saveAllData(uint8_t *&amp;data, size_t &amp;dataLen) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Store the <code>AscendIndexIVFSP</code> object in memory.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>uint8_t *&amp;data:</code> Memory pointer used to store <code>AscendIndexIVFSP</code> data.<br><code>size_t &amp;dataLen:</code> Actual length of the <code>data</code> pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The input <code>data</code> must be a null pointer. After the API returns, the user must call <code>delete</code> to free the memory after using <code>data</code>. Otherwise, a memory leak occurs.</td></tr>
</tbody></table>

### `search`<a name="ZH-CN_TOPIC_0000001635815489"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implement the feature vector search API for <code>AscendIndexIVFSP</code>, and return the IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n:</code> Number of query feature vectors.<br><code>const float *x:</code> Feature vector data.<br><code>idx_t k:</code> Number of most similar results to return.<br><code>const SearchParameters *params:</code> Optional Faiss parameter. The default value is <code>nullptr</code>, and this parameter is currently unsupported.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances:</code> Distance values between the query vectors and the top <code>k</code> nearest vectors. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid distances with 65504 or <code>-65504</code>.<br><code>idx_t *labels:</code> IDs of the top <code>k</code> nearest vectors to the query. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid labels with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of the query feature vector data <code>x</code> should be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The value range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096.</td></tr>
</tbody></table>

### `search_with_filter`<a name="ZH-CN_TOPIC_0000001585736176"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Feature vector search API for <code>AscendIndexIVFSP</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It also provides CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array with a length of <code>n * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four numbers of each filter, which are 128 bits, represent the corresponding CID. The last two numbers represent the left-closed timestamp range, that is, [<code>x</code>, <code>y</code>).</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n:</code> Number of query feature vectors.<br><code>const float *x:</code> Feature vector data.<br><code>idx_t k:</code> Number of most similar results to return.<br><code>const void *filters:</code> Filter conditions.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances:</code> Distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels:</code> IDs of the top <code>k</code> nearest vectors to the query.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value range of <code>n</code> is <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096. <code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and their lengths should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>filters</code> must be a non-null pointer, and its length must be a <code>uint32_t</code> array of <code>n * 6</code>. Otherwise, out-of-bounds reads may occur and cause the program to crash.</td></tr>
</tbody></table>

### `setNumProbes`<a name="ZH-CN_TOPIC_0000001635576089"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setNumProbes(int nprobes);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Set the total number of candidate buckets used during search.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobes:</code> <code>nprobe</code> count of <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobes</code> must be a multiple of 16 and satisfy <code>0 &lt; nprobes &lt;= nlist</code>.</td></tr>
</tbody></table>

### `setVerbose`<a name="ZH-CN_TOPIC_0000001586055516"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setVerbose(bool verbose);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Set whether to print the progress of adding feature vectors to the base vector set.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>bool verbose:</code> Whether to print the progress of adding feature vectors to the base vector set.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `trainCodeBook`<a name="ZH-CN_TOPIC_0000002148530670"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void trainCodeBook(const AscendIndexCodeBookInitParams &amp;codeBookInitParams) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">IVFSP codebook training API. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable <code>OMP_NUM_THREADS=4</code> to speed it up.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">const AscendIndexCodeBookInitParams &amp;codeBookInitParams: Initialization parameters required for codebook training.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See the <code>AscendIndexCodeBookInitParams</code> API.</td></tr>
</tbody></table>

### `addCodeBook`<a name="ZH-CN_TOPIC_0000002148372594"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void addCodeBook(const char *codeBookPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Add a trained codebook.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">const char *codeBookPath: Codebook path.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The file corresponding to <code>codeBookPath</code> should be the codebook file produced by <code>trainCodeBook</code>, and the process user must have read permission for it. The file must not be a symbolic link.</td></tr>
</tbody></table>

### `AscendIndexCodeBookInitParams`<a name="ZH-CN_TOPIC_0000002183731529"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum, std::string codeBookOutputDir, std::string learnDataPath, bool verbose);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization structure for IVFSP codebook training.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Values</td><td valign="middle"><code>int numIter:</code> Number of training iterations. The default value is 1.<br><code>int device:</code> Logical device ID. The default value is 0.<br><code>float ratio:</code> Sampling rate of the original samples used for training. The default value is <code>1.0</code>.<br><code>int batchSize:</code> Train with batches of size <code>batchSize</code>. This value must match <code>&lt;batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The default value is 32768.<br><code>int codeNum:</code> Operate on at most <code>codeNum</code> samples at a time when updating the codebook. This value must be a power of two and must match <code>&lt;codebook_batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The default value is 32768.<br><code>std::string codeBookOutputDir:</code> Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.<br><code>std::string learnDataPath:</code> Path to the original feature file used for training. The file supports the bin and npy formats. For bin files, the storage order is row-major and the data type is <code>float32</code>.<br><code>bool verbose:</code> Whether to enable additional output. The default value is <code>true</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Constraints</td><td valign="middle"><code>numIter</code> ∈ (0, 20]. <code>ratio</code> ∈ (0, 1.0]. <code>batchSize</code> ∈ (0, 32768]. <code>codeNum</code> ∈ (0, 32768]. When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner. Before you run codebook training, refer to the <code>IVFSP</code> operator model file generation instructions.</td></tr>
</tbody></table>

### `trainCodeBookFromMem`<a name="ZH-CN_TOPIC_0000002257319034"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">IVFSP codebook training API. Training data is loaded from memory. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable <code>OMP_NUM_THREADS=4</code> to speed it up.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams: Initialization parameters required for codebook training.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Constraints</td><td valign="middle">For details about <code>AscendIndexCodeBookInitFromMemParams</code>, see <code>AscendIndexCodeBookInitFromMemParams</code>.</td></tr>
</tbody></table>

### `AscendIndexCodeBookInitFromMemParams`<a name="ZH-CN_TOPIC_0000002291969193"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCodeBookInitFromMemParams (int numIter, int device, float ratio, int batchSize, int codeNum,bool verbose,std::string codeBookOutputDir,const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization structure for IVFSP codebook training. Training data is loaded from memory.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Values</td><td valign="middle"><code>int numIter:</code> Number of training iterations. The default value is 1.<br><code>int device:</code> Logical device ID. The default value is 0.<br><code>float ratio:</code> Sampling rate of the original samples used for training. The default value is <code>1.0</code>.<br><code>int batchSize:</code> Train with batches of size <code>batchSize</code>. This value must match <code>&lt;batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The value must be greater than 0, and the default value is 32768.<br><code>int codeNum:</code> Operate on at most <code>codeNum</code> samples at a time when updating the codebook. This value must be a power of two and must match <code>&lt;codebook_batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The value must be greater than 0, and the default value is 32768.<br><code>std::string codeBookOutputDir:</code> Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.<br><code>bool verbose:</code> Whether to enable additional output. The default value is <code>true</code>.<br><code>const float *memLearnData:</code> Pointer to in-memory data. The default value is a null pointer.<br><code>size_t memLearnDataSize:</code> Length of the in-memory data. The default value is 0.<br><code>bool isTrainAndAdd:</code> Whether to add the codebook directly to the <code>Index</code> after training. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Constraints</td><td valign="middle"><code>numIter</code> ∈ (0, 20]. <code>ratio</code> ∈ (0, 1.0]. <code>memLearnDataSize % dim == 0</code>. <code>memLearnDataSize &lt;= 25G</code>. When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner. Before you run codebook training, refer to the <code>IVFSP</code> operator model file generation instructions. When <code>isTrainAndAdd</code> is <code>true</code>, the trained codebook is added directly to the <code>Index</code> and is not written to disk. When <code>isTrainAndAdd</code> is <code>false</code>, the codebook is saved to <code>codeBookOutputDir</code>, and you must call <code>addCodeBook</code> manually. <code>memLearnDataSize</code> must be the actual length of the <code>memLearnData</code> pointer. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `AscendIndexIVFSPConfig`<a id="ZH-CN_TOPIC_0000001635696057"></a>

`AscendIndexIVFSP` requires the corresponding `AscendIndexIVFSPConfig` to initialize the corresponding resources.

**Common Parameters<a name="section17656114673616"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">Parameter</td><td valign="middle">Data Type</td><td valign="middle">Parameter Description</td></tr>
<tr><td width="210" align="center" valign="middle">handleBatch</td><td valign="middle">int</td><td valign="middle">Number of candidate buckets submitted for computation each time during search. The default value is 64.</td></tr>
<tr><td width="210" align="center" valign="middle">nprobe</td><td valign="middle">int</td><td valign="middle">Total number of candidate buckets used during search. The default value is 64.</td></tr>
<tr><td width="210" align="center" valign="middle">searchListSize</td><td valign="middle">int</td><td valign="middle">Maximum number of samples in each bucket submitted for computation each time during search. The default value is 32768. If a bucket is too large, the program automatically splits the bucket into multiple operator submissions according to <code>searchListSize</code> to compute distances.</td></tr>
</tbody></table>

**API Description<a name="section74781713710"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSPConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default <code>devices</code> value is <code>{0}</code>, so the 0th Ascend AI Processor is used for computation. The default <code>resources</code> value is 128 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table121971648373"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline explicit AscendIndexIVFSPConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSPConfig</code>. It creates an <code>AscendIndexIVFSPConfig</code> and specifies the device IDs on the device side and the resource pool size.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resources:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVF_SP_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br><code>uint32_t blockSize:</code> Preallocated memory block size, in bytes. The default value is <code>DEFAULT_BLOCK_SIZE</code> in the header file.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. Currently, only one NPU device is supported. The configured <code>resources</code> value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>

<a name="table56061252785"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline explicit AscendIndexIVFSPConfig(std::vector&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSPConfig</code>. It creates an <code>AscendIndexIVFSPConfig</code> and specifies the device IDs on the device side and the resource pool size.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resources:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVF_SP_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br><code>uint32_t blockSize:</code> Preallocated memory block size, in bytes. The default value is <code>DEFAULT_BLOCK_SIZE</code> in the header file.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. Currently, only one NPU device is supported. The configured <code>resources</code> value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>

## `AscendIndexIVFSQ`<a name="ZH-CN_TOPIC_0000001506334625"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456694964"></a>

The `AscendIndexIVFSQ` class uses IVF for acceleration and is a two-stage approximate retrieval algorithm.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval internally uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexIVFSQ`<a name="ZH-CN_TOPIC_0000001506414893"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQ</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code>.<br><code>AscendIndexIVFSQConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

<a name="table1823217151014"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(int dims, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQ</code>. It creates an <code>AscendIndexIVFSQ</code>, and the device-side resources are set according to the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims:</code> Dimension of the feature vectors managed by <code>AscendIndexIVFSQ</code>.<br><code>int nlist:</code> Number of cluster centroids. This parameter corresponds to <code>coarse_centroid_num</code> in the operator generation script.<br><code>faiss::ScalarQuantizer::QuantizerType qtype:</code> Quantizer type of <code>AscendIndexIVFSQ</code>.<br><code>faiss::MetricType metric:</code> Distance metric used by <code>AscendIndex</code> for feature vector similarity search.<br><code>bool encodeResidual:</code> Whether to encode residuals.<br><code>AscendIndexIVFSQConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {64, 128, 256, 384, 512}. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. <code>qtype = ScalarQuantizer::QuantizerType::QT_8bit</code>, and only <code>ScalarQuantizer::QuantizerType::QT_8bit</code> is supported. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.<br>Note:<br>Currently, when <code>metric = faiss::MetricType::METRIC_INNER_PRODUCT</code>, <code>encodeResidual</code> only supports <code>false</code>. That is, the IVFSQ method with residual encoding is not currently supported. When <code>encodeResidual</code> is <code>true</code>, the code can run successfully, but there is an accuracy issue.</td></tr>
</tbody></table>

<a name="table134501935171012"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQ</code>. It creates an <code>AscendIndexIVFSQ</code>, and the device-side resources are set according to the values configured in <code>config</code>. This API does not perform initialization. The subclass performs the initialization-related work. This API will be deprecated later, so do not use it.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims:</code> Dimension of the feature vectors managed by <code>AscendIndexIVFSQ</code>.<br><code>int nlist:</code> Number of cluster centroids. This parameter corresponds to <code>coarse_centroid_num</code> in the operator generation script.<br><code>faiss::MetricType metric:</code> Distance metric used by <code>AscendIndex</code> for feature vector similarity search.<br><code>AscendIndexIVFSQConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {64, 128, 256, 384, 512}. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(const AscendIndexIVFSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> copy constructor as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSQ&amp;:</code> Constant <code>AscendIndexIVFSQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~AscendIndexIVFSQ`<a name="ZH-CN_TOPIC_0000001456534936"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSQ();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor for <code>AscendIndexIVFSQ</code>. It destroys the <code>AscendIndexIVFSQ</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456375244"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>index</code> to Ascend based on <code>AscendIndexIVFSQ</code>, while keeping the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer.<br><code>index-&gt;d</code> ∈ {256}. <code>index-&gt;sq.d</code> ∈ {32, 64, 128}. The dimension of <code>index</code> must be greater than the dimension of <code>index-&gt;sq</code>, and it must be divisible by the dimension of <code>index-&gt;sq</code>. Do not call this API on an updated object.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000001506334649"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFScalarQuantizer *index) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy the retrieval resources of <code>AscendIndexIVFSQ</code> to the CPU side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user is responsible for freeing the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001456854860"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ&amp; operator=(const AscendIndexIVFSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSQ&amp;:</code> Constant <code>AscendIndexIVFSQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000001456854976"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Train <code>AscendIndexIVFSQ</code>. This class inherits the relevant APIs in <code>AscendIndex</code> and provides a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n:</code> Number of feature vectors in the training set.<br><code>const float *x:</code> Feature vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering, and a small training set may affect query accuracy. The value range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `AscendIndexIVFSQConfig`<a id="ZH-CN_TOPIC_0000001456375204"></a>

`AscendIndexIVFSQ` requires the corresponding `AscendIndexIVFSQConfig` to initialize the corresponding resources.

**`AscendIndexIVFSQConfig`<a name="section015013311183"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default <code>devices</code> value is <code>{0}</code>, so the 0th Ascend AI Processor is used for computation. The default <code>resource</code> value is 384 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table19736185071817"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQConfig</code>. It creates an <code>AscendIndexIVFSQConfig</code>, sets the Ascend AI Processor resources on the device side according to the values configured in <code>devices</code>, configures the resource pool size, and performs default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resourceSize:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVFSQ_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The configured <code>resourceSize</code> value must not exceed 10 * 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>

<a name="table1056711401917"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQConfig</code>. It creates an <code>AscendIndexIVFSQConfig</code>, sets the Ascend AI Processor resources on the device side according to the values configured in <code>devices</code>, configures the resource pool size, and performs default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resourceSize:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVFSQ_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The configured <code>resourceSize</code> value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>

**`SetDefaultIVFSQConfig`<a name="section039015215286"></a>**

<a name="table1185313082915"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline void SetDefaultIVFSQConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Perform default initialization. Set the number of iterations to 16 and set a maximum of 512 points for each centroid.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendIndexIVFSQT`<a name="ZH-CN_TOPIC_0000001456375224"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506615005"></a>

The `AscendIndexIVFSQT` class contains the three-stage retrieval `IVFSQ` algorithm with dimensionality reduction. You need to pass two parameters to specify the dimensions before and after dimensionality reduction, and the original dimension must be divisible by the reduced dimension. It is suitable for scenarios with a base vector set on the order of 10 million.

You need to generate the operators required for three-stage retrieval according to the `IVFSQT` operator generation method.

This type provides fuzzy clustering. Before bucket assignment, use the `threshold` parameter to control the degree of fuzziness. Set the `threshold` value according to the base vector set capacity and the available memory size. A `threshold` that is too large can cause insufficient memory and lead to failure. For Atlas 200/300/500 inference product environments, you are advised to set it to [1.0, 1.1]. For Atlas inference series product environments, you are advised to set it to [1.0, 1.5]. For search, you are advised to use `batch size = 65536`.

The workflow is: 1. Construct the `Index` object. 2. Train the data. 3. Add the data. 4. Update the data. 5. Search the data. 6. Destroy the `Index` object. After `update`, adding data is no longer supported. If you need to search new data, destroy the original `Index` object and use the workflow again from the beginning.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval internally uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexIVFSQT`<a name="ZH-CN_TOPIC_0000001506495685"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQT</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.<br><code>AscendIndexIVFSQTConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. <code>index-&gt;d</code> ∈ {256}. <code>index-&gt;sq.d</code> ∈ {32, 64, 128}. The dimension of <code>index</code> must be greater than the dimension of <code>index-&gt;sq</code>, and it must be divisible by the dimension of <code>index-&gt;sq</code>.</td></tr>
</tbody></table>

<a name="table124585216195"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQT(int dimIn, int dimOut, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQT</code>. It creates an <code>AscendIndexIVFSQT</code>, and the device-side resources are set according to the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dimIn:</code> Dimension of the original feature vectors managed by <code>AscendIndexIVFSQT</code>.<br><code>int dimOut:</code> Dimension of the reduced feature vectors managed by <code>AscendIndexIVFSQT</code>.<br><code>int nlist:</code> Number of cluster centroids. This parameter corresponds to <code>coarse_centroid_num</code> in the operator generation script.<br><code>faiss::ScalarQuantizer::QuantizerType qtype:</code> Quantizer type of <code>AscendIndexIVFSQT</code>.<br><code>faiss::MetricType metric:</code> Distance metric used by <code>AscendIndex</code> for feature vector similarity search.<br><code>AscendIndexIVFSQTConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dimIn</code> ∈ {256}. <code>dimOut</code> ∈ {32, 64, 128}. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. <code>qtype = ScalarQuantizer::QuantizerType::QT_8bit</code>, and only the <code>ScalarQuantizer::QuantizerType::QT_8bit</code> quantizer type is supported. <code>metric = faiss::MetricType::METRIC_INNER_PRODUCT</code>, and only <code>faiss::MetricType::METRIC_INNER_PRODUCT</code> is supported.</td></tr>
</tbody></table>

<a name="table68594118203"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQT(const AscendIndexIVFSQT&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> copy constructor as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSQT&amp;:</code> <code>AscendIndexIVFSQT</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~AscendIndexIVFSQT`<a name="ZH-CN_TOPIC_0000001456854984"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSQT();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor for <code>AscendIndexIVFSQT</code>. It destroys the <code>AscendIndexIVFSQT</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456695060"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>index</code> to Ascend based on <code>AscendIndexIVFSQT</code>, while preserving the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer.<br><code>index-&gt;d</code> ∈ {256}. <code>index-&gt;sq.d</code> ∈ {32, 64, 128}. The dimension of <code>index</code> must be greater than the dimension of <code>index-&gt;sq</code>, and it must be divisible by the dimension of <code>index-&gt;sq</code>. Do not call this API on an updated object.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000001506495825"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy the retrieval resources of <code>AscendIndexIVFSQT</code> to the CPU side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user frees the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

### `fineTune`<a name="ZH-CN_TOPIC_0000001456694860"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void fineTune(size_t n, const float *x);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Fine-tune and optimize the centroids to avoid uneven bucket assignment.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n:</code> Number of feature vectors.<br><code>const float *x:</code> Feature vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `getFuzzyK`<a name="ZH-CN_TOPIC_0000001456855008"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getFuzzyK() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Get the maximum value used when a vector is assigned to buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>int:</code> Maximum value used when a vector is assigned to buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getListCodesAndIds`<a name="ZH-CN_TOPIC_0000001687739112"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the feature vectors and corresponding IDs for a specific <code>nlistId</code> in the current <code>AscendIndexIVFSQT</code> <code>nlist</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId:</code> Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;uint8_t&gt;&amp; codes:</code> Feature vectors at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.<br><code>std::vector&lt;ascend_idx_t&gt;&amp; ids:</code> Feature vector IDs at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `getListLength`<a name="ZH-CN_TOPIC_0000001735977797"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>uint32_t getListLength(int listId) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the length for a specific <code>nlistId</code> in the current <code>AscendIndexIVFSQT</code> <code>nlist</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId:</code> Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Length at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `getLowerBound`<a name="ZH-CN_TOPIC_0000001506614885"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getLowerBound() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the threshold for second-level clustering.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Threshold for second-level clustering.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getMergeThres`<a name="ZH-CN_TOPIC_0000001506615073"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getMergeThres() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Get the threshold for merging sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Threshold for merging sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getQMax`<a name="ZH-CN_TOPIC_0000001456535208"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>float getQMax() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the maximum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Maximum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getQMin`<a name="ZH-CN_TOPIC_0000001506615029"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>float getQMin() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the minimum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Minimum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getThreshold`<a name="ZH-CN_TOPIC_0000001506334633"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>float getThreshold() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Get the threshold used to determine whether a vector is assigned to multiple buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>float:</code> Threshold used to determine whether a vector is assigned to multiple buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001506615085"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQT&amp; operator=(const AscendIndexIVFSQT&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSQT&amp;</code>: An <code>AscendIndexIVFSQT</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `remove_ids`<a name="ZH-CN_TOPIC_0000001506615053"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes base library features by ID.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: The feature vectors to delete. For details about usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">The number of deleted feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version.</td></tr>
</tbody></table>

### `reset`<a name="ZH-CN_TOPIC_0000001506334789"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Resets the index and clears the feature data.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Do not continue using this object after you call this API.</td></tr>
</tbody></table>

### `setAddTotal`<a name="ZH-CN_TOPIC_0000001456375316"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setAddTotal(size_t addTotal);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the total number of base library vectors to add. The default value is 100000000. You must set <code>PreciseMemControl</code> to <code>true</code> first.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t addTotal</code>: The total number of base library vectors to add.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `setFuzzyK`<a name="ZH-CN_TOPIC_0000001456534940"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setFuzzyK(int value);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the maximum value for each vector when it is assigned to a bucket.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int value</code>: The maximum value for each vector when it is assigned to a bucket. You are advised to keep it at the default value 3.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>value</code> is (0, 10].</td></tr>
</tbody></table>

### `setLowerBound`<a name="ZH-CN_TOPIC_0000001506334777"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setLowerBound(int lowerBound);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the threshold for second-level clustering. The default value is 32.<br>If the number of elements in a first-level clustering bucket is greater than <code>lowerBound</code>, second-level clustering is performed. Otherwise, the original state is retained.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int lowerBound</code>: The threshold for second-level clustering.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `setMemoryLimit`<a name="ZH-CN_TOPIC_0000001506614917"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setMemoryLimit(float memoryLimit);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the Host memory limit. The default value is 32, in <code>GB</code>. You must set <code>PreciseMemControl</code> to <code>true</code> first.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>float memoryLimit</code>: The memory limit.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `setMergeThres`<a name="ZH-CN_TOPIC_0000001456694900"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setMergeThres(int mergeThres);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the threshold for merging sub-buckets. The default value is 5.<br>If the number of elements in a sub-bucket after second-level clustering is smaller than <code>mergeThres</code>, merge the elements of that sub-bucket into other sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int mergeThres</code>: The threshold for merging sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `setNumProbes`<a name="ZH-CN_TOPIC_0000001736410013"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setNumProbes(int nprobes) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the <code>nprobe</code> value of the current <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobes</code>: The <code>nprobe</code> value of <code>AscendIndexIVFSQT</code>. You are advised to keep it at the default value 64.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobes</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. <code>l2Probe</code> ≥ <code>nprobes</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, and <code>l2Probe</code> ≤ <code>nprobes * 64</code>. <code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}. For details about <code>l2Probe</code> and <code>l3SegmentNum</code>, see <code>setSearchParams</code>. <code>setNumProbes</code> is expected to be removed in September 2025. Use <code>setSearchParams</code> instead.</td></tr>
</tbody></table>

### `setPreciseMemControl`<a name="ZH-CN_TOPIC_0000001506334681"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setPreciseMemControl(bool preciseMemControl);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Specifies whether to precisely limit the memory size on the Host side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>bool preciseMemControl</code>: The default value is <code>false</code>, which disables precise memory limiting on the Host side. <code>true</code> enables it.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

### `setSearchParams`<a name="ZH-CN_TOPIC_0000002052679693"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the parameters that affect retrieval accuracy and performance.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobe</code>: The <code>nprobe</code> value of <code>AscendIndexIVFSQT</code>. You are advised to keep it at the default value 64.<br><code>int l2Probe</code>: The number of sub-buckets selected during second-stage retrieval. The default value is 48.<br><code>int l3SegmentNum</code>: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is 96.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobe</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. <code>l2Probe</code> ≥ <code>nprobe</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, and <code>l2Probe</code> ≤ <code>nprobe * 64</code>.<br><code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}.</td></tr>
</tbody></table>

### `setSortMode`<a name="ZH-CN_TOPIC_0000002165943965"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setSortMode(int mode);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the <code>topk</code> sorting mode. Mode 0 means approximate sorting. Mode 1 means exact sorting.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int mode</code>: The <code>topk</code> sorting mode.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">You must call this API before the <code>Search</code> API. <code>mode</code> supports only 0 or 1, and the default is 0. Mode 0: Approximate sorting truncates part of the <code>topk</code> results to improve performance. Mode 1: Exact sorting improves retrieval accuracy at the cost of some performance.</td></tr>
</tbody></table>

### `setThreshold`<a name="ZH-CN_TOPIC_0000001456854808"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setThreshold(float value);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the threshold for determining whether a vector is assigned to multiple buckets. The default value is <code>1.0</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>float value</code>: The threshold for determining whether a vector is assigned to multiple buckets. You are advised to set it in the range [1.0, 1.5]. Because the Device side has a memory limit, once memory usage reaches the limit, the OOM mechanism is triggered and kills the process. You can check the Device-side memory limit data first (<code>/sys/fs/cgroup/memory/usermemory/memory.limit_in_bytes</code>) to estimate the size of the base library to add. If memory is tight, you are advised to keep the parameter in the range [1.0, 1.1].</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>value</code> is [0, <code>fuzzyK</code> - 1]. For the valid range of <code>fuzzyK</code>, see the <code>getFuzzyK</code> API.</td></tr>
</tbody></table>

### `setUseCpuUpdate`<a name="ZH-CN_TOPIC_0000002167379329"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>setUseCpuUpdate(int numThreads);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Specifies whether to use the CPU for update.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int numThreads</code>: The number of CPU cores used for update. The default value is the current number of CPU cores.<br>If the current CPU has more than 96 cores: if the current core count is smaller than the input <code>numThreads</code>, set <code>numThreads</code> to 96; if <code>96 &lt; numThreads &lt;=</code> the current core count, set <code>numThreads</code> to 96; if <code>numThreads &lt;= 96</code>, keep the input value. If the current CPU has 96 cores or fewer: if the current core count is smaller than the input <code>numThreads</code> and <code>numThreads &lt;= 96</code>, set <code>numThreads</code> to the current core count; if <code>0 &lt; numThreads &lt;=</code> the current core count, keep the input value.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>numThreads</code> must be greater than 0. Configure it before you use <code>update</code>.</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000001456375352"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFSQT</code>. This class inherits the relevant APIs in <code>AscendIndexIVFSQ</code> and provides concrete implementations.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of feature vectors in the training set.<br><code>const float *x</code>: Feature vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A training set that is too small may affect query accuracy. The valid range of <code>n</code> here is <code>nlist ≤ n ≤ 7,000,000</code>. The pointer <code>x</code> must be a non-null pointer, and its length must be <code>dimIn * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `update`<a name="ZH-CN_TOPIC_0000001506414869"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void update(bool cleanData = true);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">This is the second stage of three-stage retrieval. After all base library data has been added and before <code>search</code> is called, this API trains sub-bucket centers and assigns vectors to buckets according to those centers.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>cleanData</code>: Specifies whether to clear intermediate data. The default value is <code>true</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">You only need to call this API once in a full retrieval workflow.</td></tr>
</tbody></table>

### `updateTParams`<a name="ZH-CN_TOPIC_0000001456854936"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void updateTParams(int l2Probe, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Passes in the parameters required for three-stage retrieval during testing.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int l2Probe</code>: The number of sub-buckets selected during second-stage retrieval. The default value is 48.<br><code>int l3SegmentNum</code>: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is 96.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobe</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. <code>l2Probe</code> ≥ <code>nprobe</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, and <code>l2Probe</code> ≤ <code>nprobe * 64</code>.<br><code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}. For details about the <code>nprobe</code> setting, see <code>setSearchParams</code>. <code>updateTParams</code> is expected to be removed in September 2026. Use <code>setSearchParams</code> instead.</td></tr>
</tbody></table>

## `AscendIndexIVFSQTConfig`<a name="ZH-CN_TOPIC_0000001506495881"></a>

`AscendIndexIVFSQT` uses the corresponding `AscendIndexIVFSQTConfig` to initialize the required resources.

**AscendIndexIVFSQTConfig<a name="section6579185362314"></a>**

> [!NOTE]
> `AscendIndexIVFSQTConfig` inherits from [`AscendIndexIVFSQConfig`](#ascendindexivfsqconfig).

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default <code>devices</code> value is <code>{0}</code>, which uses the 0th Ascend AI Processor for computation. The default <code>resource</code> value is <code>384 MB</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table42413462115"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexIVFSQTConfig</code>. It creates an <code>AscendIndexIVFSQTConfig</code> instance and, based on the values configured in <code>devices</code>, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVFSQT_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to <code>1024 MB</code> when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicate device IDs. The configured value of <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

<a name="table0812225238"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSQTConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexIVFSQTConfig</code>. It creates an <code>AscendIndexIVFSQTConfig</code> instance and, based on the values configured in <code>devices</code>, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVFSQT_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to <code>1024 MB</code> when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, non-duplicate device IDs. The configured value of <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

**SetDefaultIVFSQConfig<a name="section18396165022414"></a>**

<a name="table14953182017255"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline void SetDefaultIVFSQConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs the default initialization, sets the number of iterations to 16, and sets a maximum of 512 points for each centroid.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendIndexVStar`<a name="ZH-CN_TOPIC_0000002044351677"></a>

### Overview<a name="ZH-CN_TOPIC_0000002044510693"></a>

Ascend's self-developed vector retrieval algorithm provides approximate retrieval for high-dimensional large base libraries on the Ascend side. It uses a self-developed matrix approximation strategy to compress feature vectors before storing them in the base library, and then uses a self-developed retrieval strategy to retrieve the top `K` most similar vectors from the base library.

The vectors stored in the base library and the query vectors passed to each API must be normalized `float` values.

This algorithm does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you must acquire a lock before use. Otherwise, the retrieval API may fail. It also does not support sharing one Device across threads.

This algorithm is mainly designed for approximate fuzzy search in large base library scenarios, and its accuracy is lower than brute-force retrieval. In small base library scenarios, you are advised to increase the hyperparameter values appropriately to reduce the loss in accuracy.

### `AscendIndexVStar`<a name="ZH-CN_TOPIC_0000002044513265"></a>

> [!NOTE]
>
>- When you create an `Index` instance, set `params.dim` according to the actual situation.
>- `params.subSpaceDim` and `params.nlist` should match the corresponding parameters used for codebook training.

<a name="table13851535141118"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>explicit AscendIndexVStar(const AscendIndexVstarInitParams&amp; params);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexVStar</code>. It creates an <code>Index</code> with the corresponding dimension based on the values configured in <code>params</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVstarInitParams&amp; params</code>: The constructor parameters. For details, see <code>AscendIndexVstarInitParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">For details, see <code>AscendIndexVstarInitParams</code>.</td></tr>
</tbody></table>

<a name="table11631734281"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVStar(const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexVStar</code>. It creates an <code>Index</code> with an unknown input data dimension and unknown hyperparameters based on <code>deviceList</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;int&gt;&amp; deviceList</code>: Device-side device IDs.<br><code>bool verbose</code>: Specifies whether to enable the <code>verbose</code> option. When enabled, some operations provide additional print prompts. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceList</code> must contain valid device IDs. Currently, only one device is supported. After you create an <code>Index</code> instance with this constructor, you must first call <code>LoadIndex</code> to load the pre-saved <code>Index</code> instance from disk, and then you can perform other operations.</td></tr>
</tbody></table>

<a name="table8937623141615"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVStar(const AscendIndexVStar&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares this copy constructor as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVStar&amp;</code>: An <code>AscendIndexVStar</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `LoadIndex`<a name="ZH-CN_TOPIC_0000002008232688"></a>

<a name="table950712481817"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; indexPath, AscendIndexVStar* indexVStar = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads an existing index from disk into the Device.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; indexPath</code>: The data file path.<br><code>AscendIndexVStar* indexVStar</code>: Used only in the <code>MultiSearch</code> scenario so that all <code>Index</code> instances share the codebook of the first <code>Index</code> instance.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the directory that contains <code>indexPath</code> exists and that the user who runs the process has read permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links. <code>indexVStar</code> must not be a null pointer in the <code>MultiSearch</code> scenario. It must be a null pointer in the single-<code>Index</code> scenario. If a valid <code>Index</code> pointer is used in the single-<code>Index</code> scenario, the original <code>Index</code> codebook is replaced by the codebook of the parameter <code>Index</code> instance.</td></tr>
</tbody></table>

### `WriteIndex`<a name="ZH-CN_TOPIC_0000002044351681"></a>

<a name="table29774016915"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the index to disk.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; indexPath</code>: The file path where the data is saved.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the directory that contains <code>indexPath</code> exists and that the user who runs the process has write permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links. If the file already exists, it is overwritten. In this case, the user who runs the process must be the owner of the file.</td></tr>
</tbody></table>

### `AddCodeBooksByIndex`<a name="ZH-CN_TOPIC_0000002044510697"></a>

<a name="table81089131197"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooksByIndex(AscendIndexVStar&amp; indexVStar);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">In a multi-<code>Index</code> retrieval scenario, this API loads the codebook of the input <code>Index</code> instance into the current <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>AscendIndexVStar&amp; indexVStar</code>: An <code>Index</code> instance with the codebook already populated.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is used only in the <code>MultiSearch</code> scenario.</td></tr>
</tbody></table>

### `AddCodeBooksByPath`<a name="ZH-CN_TOPIC_0000002008390980"></a>

<a name="table1523424814919"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooksByPath(const std::string&amp; codeBooksPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads a codebook into the current <code>Index</code> from the codebook path.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; codeBooksPath</code>: The codebook data file path.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the directory that contains <code>codeBooksPath</code> exists and that the user who runs the process has read permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links.</td></tr>
</tbody></table>

### `Add`<a name="ZH-CN_TOPIC_0000002008232692"></a>

<a name="table18288921121213"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseData);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Builds the <code>AscendIndexVStar</code> base library and adds new feature vectors to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseData</code>: The feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of <code>baseData</code> must be <code>n * dim</code>, where <code>n</code> is the number of vectors to add to the base library and <code>dim</code> is the vector dimension. <code>n</code> must be in the range [10000, 1e8].<br>This API does not set IDs. The default ID range of the base library is [<code>ntotal</code>, <code>ntotal</code> + <code>n</code>), where <code>ntotal</code> is the number of vectors already in the <code>Index</code>, and <code>n</code> is the number of vectors to add to the base library.</td></tr>
</tbody></table>

> [!NOTE]
>
>- The `Add` API cannot be used together with the `AddWithIds` API.
>- After you use the `Add` API, the labels in the `Search` results may be duplicated. If your business logic requires labels, you are advised to use the [AddWithIds API](#addwithids).

### `AddWithIds`<a name="ZH-CN_TOPIC_0000002044351685"></a>

<a name="table32483414124"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddWithIds(const std::vector&lt;float&gt;&amp; baseData, const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Builds the <code>AscendIndexVStar</code> base library and adds new feature vectors to the base library. This API allows the user to specify the IDs of the base library vectors to add.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseData</code>: The feature vectors to add to the base library.<br><code>const std::vector&lt;int64_t&gt;&amp; ids</code>: The array of IDs to map to the base library vectors to add.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of <code>baseData</code> must be <code>n * dim</code>, where <code>n</code> is the number of vectors to add to the base library and <code>dim</code> is the vector dimension. The length of <code>ids</code> must be <code>n</code>. Based on your own business scenario, ensure that <code>ids</code> are valid. If duplicate IDs exist in the base library, the <code>label</code> in the retrieval results cannot correspond to a specific base library vector. <code>n</code> must be in the range [10000, 1e8].</td></tr>
</tbody></table>

### `DeleteByIds`<a name="ZH-CN_TOPIC_0000002044510701"></a>

<a name="table1284884631210"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteByIds(const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the vector data in the base library that corresponds to the IDs in the parameter array.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;int64_t&gt;&amp; ids</code>: The array of vector IDs to delete from the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The IDs in <code>ids</code> must be IDs used by the base library addition API.</td></tr>
</tbody></table>

### `DeleteById`<a name="ZH-CN_TOPIC_0000002008390984"></a>

<a name="table9845165841212"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteById(int64_t id);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the vector data in the base library that corresponds to the parameter ID.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int64_t id</code>: The ID of the base library vector to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The ID must be an ID used by the base library addition API.</td></tr>
</tbody></table>

### `DeleteByRange`<a name="ZH-CN_TOPIC_0000002008232696"></a>

<a name="table103969158136"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteByRange(int64_t startId, int64_t endId);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the vector data in the base library that corresponds to the ID range in the parameters.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int64_t startId</code>: The starting ID of the base library vectors to delete.<br><code>int64_t endId</code>: The ending ID of the base library vectors to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The IDs to delete must be IDs used by the base library addition API, and the ID must be in the range [<code>startId</code>, <code>endId</code>].</td></tr>
</tbody></table>

### `Search`<a name="ZH-CN_TOPIC_0000002044351689"></a>

<a name="table197566920146"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(const AscendIndexSearchParams&amp; params) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval and returns the IDs of the most similar <code>topK</code> features based on the input feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code>: The length must be greater than or equal to <code>n * topK</code>.</td></tr>
</tbody></table>

### `SearchWithMask`<a name="ZH-CN_TOPIC_0000002044510705"></a>

<a name="table777072291418"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval and returns the IDs of the most similar <code>topK</code> features based on the input feature vectors. <code>mask</code> is a 0 and 1 bit string. Each bit indicates whether the corresponding feature in the base library participates in distance calculation. 0 means it does not participate, and 1 means it does.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.<br><code>const std::vector&lt;uint8_t&gt;&amp; mask</code>: The feature base library mask.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code>: The length must be greater than or equal to <code>n * topK</code>. <code>mask</code>: The length must be greater than or equal to <code>n * ceil(ntotal/8)</code>, where <code>ntotal</code> is the number of base library features.</td></tr>
</tbody></table>

### `MultiSearch`<a name="ZH-CN_TOPIC_0000002008390988"></a>

<a name="table158666394146"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR MultiSearch(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, bool merge) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval across multiple <code>AscendIndexVStar</code> libraries and returns the IDs and distances of the most similar <code>topK</code> features based on the input feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</code>: Multiple <code>Index</code> instances to search.<br><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.<br><code>bool merge</code>: Specifies whether to merge the retrieval results across multiple <code>Index</code> instances.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code> must meet the following requirements. When <code>merge = true</code>, the length must be greater than or equal to <code>n * topK</code>. When <code>merge = false</code>, the length must be greater than or equal to <code>indexes.size() * n * topK</code>. <code>indexes</code> must meet the following requirement: <code>0 &lt; indexes.size() ≤ 150</code>.</td></tr>
</tbody></table>

### `MultiSearchWithMask`<a name="ZH-CN_TOPIC_0000002008232700"></a>

<a name="table141672058131413"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR MultiSearchWithMask(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask, bool merge);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval across multiple <code>AscendIndexVStar</code> libraries and returns the IDs and distances of the most similar <code>topK</code> features based on the input feature vectors. It also supports deciding whether the base library participates in distance calculation based on a <code>mask</code>. <code>mask</code> is a 0 and 1 bit string. Each bit indicates whether the corresponding feature in the base library participates in distance calculation. 0 means it does not participate, and 1 means it does.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</code>: Multiple <code>Index</code> instances to search.<br><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.<br><code>const std::vector&lt;uint8_t&gt;&amp; mask</code>: The feature base library mask.<br><code>bool merge</code>: Specifies whether to merge the retrieval results across multiple <code>Index</code> instances.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code> must meet the following requirements. When <code>merge = true</code>, the length must be greater than or equal to <code>n * topK</code>. When <code>merge = false</code>, the length must be greater than or equal to <code>indexes.size() * n * topK</code>. <code>mask</code>: The length must be greater than or equal to <code>n * ceil(ntotal_max/8)</code>, where <code>ntotal_max</code> is the number of base library features and is the maximum number of base library vectors among all <code>Index</code> instances. <code>indexes</code> must meet the following requirement: <code>0 &lt; indexes.size() ≤ 150</code>.</td></tr>
</tbody></table>

### `SetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002044351693"></a>

<a name="table4215111781514"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams&amp; params);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the hyperparameters used when an <code>AscendIndexVstar</code> instance performs retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVstarHyperParams&amp; params</code>: The retrieval hyperparameters. For details, see <code>AscendIndexVstarHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nProbeL1</code> ∈ (16, <code>nListL1</code>], <code>nProbeL1 % 8 == 0</code>. <code>nProbeL2</code> ∈ (16, <code>nProbeL1</code> * <code>nList2</code>], <code>nProbeL2 % 8 == 0</code>. <code>l3SegmentNum</code> ∈ (100, 5000], <code>l3SegmentNum % 8 == 0</code>.</td></tr>
</tbody></table>

### `GetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002044510709"></a>

<a name="table5860202961515"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams&amp; params) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the hyperparameters used during vector retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>AscendIndexVstarHyperParams&amp; params</code>: The retrieval hyperparameters. For details, see <code>AscendIndexVstarHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `GetDim`<a name="ZH-CN_TOPIC_0000002008390992"></a>

<a name="table6661184351519"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetDim(int&amp; dim) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the dimension used when the index is initialized.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>int&amp; dim</code>: The dimension of the <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `GetNTotal`<a name="ZH-CN_TOPIC_0000002008232704"></a>

<a name="table1919613597154"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetNTotal(uint64_t&amp; ntotal) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the number of base library vectors in the current index.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>uint64_t&amp; ntotal</code>: The total number of base library vectors in the current <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Reset`<a name="ZH-CN_TOPIC_0000002044351697"></a>

<a name="table19794117167"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Reset();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Resets the index and clears the saved index data.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you reset the index, the parameters that the user provided when initializing the index are retained.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000002008390996"></a>

<a name="table3792193711620"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVStar&amp; operator=(const AscendIndexVStar&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVStar&amp;</code>: An <code>AscendIndexVStar</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendIndexGreat`<a name="ZH-CN_TOPIC_0000002044829945"></a>

### Overview<a name="ZH-CN_TOPIC_0000002008751966"></a>

This self-developed vector retrieval algorithm provides approximate retrieval for high-dimensional large base libraries on the Ascend side and the Kunpeng side. It uses a self-developed retrieval strategy to retrieve the top `K` most similar vectors from the base library.

The vectors stored in the base library and the query vectors passed to each API must be normalized `float` values.

This algorithm does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you must acquire a lock before use. Otherwise, the retrieval API may fail. It also does not support sharing one Device across threads.

This algorithm is mainly designed for approximate fuzzy search in large base library scenarios, and its accuracy is lower than brute-force retrieval. In small base library scenarios, you are advised to increase the hyperparameter values appropriately to reduce the loss in accuracy.

> [!NOTE]
>
>- When you create an `Index` instance, set `params.dim` according to the actual situation.
>- The `Index` has two algorithm modes: `KMode`, which uses only the Kunpeng-side algorithm, and `AKMode`, which uses the Ascend plus Kunpeng algorithm. In `AKMode`, you must generate the corresponding operators in advance.
>- Ensure that `subSpaceDim` and `nlist` match the corresponding parameters used for codebook training.

### `AscendIndexGreat`<a name="ZH-CN_TOPIC_0000002044829953"></a>

<a name="table5404639201712"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat(const std::string&amp; mode, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; mode</code>: Specifies the algorithm mode.<br><code>const std::vector&lt;int&gt;&amp; deviceList</code>: The specified NPU-side device IDs.<br><code>bool verbose</code>: Specifies whether to enable the <code>verbose</code> option. When enabled, some operations provide additional print prompts. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>mode</code> supports only <code>KMode</code> and <code>AKMode</code>. For <code>deviceList</code>, use the <code>npu-smi</code> command to query the corresponding NPU IDs. Only one device ID is supported. After you create an <code>Index</code> instance with this constructor, you must first call <code>LoadIndex</code> to load the pre-saved <code>Index</code> instance from disk, and then you can perform other operations.</td></tr>
</tbody></table>

<a name="table72261454131719"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>explicit AscendIndexGreat(const AscendIndexGreatInitParams&amp; kModeInitParams);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">Initialization parameters required by the <code>Index</code>, specifically <code>kModeInitParams</code>. For details, see <code>AscendIndexGreatInitParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See the parameter descriptions and constraints in <code>AscendIndexGreatInitParams</code>.</td></tr>
</tbody></table>

<a name="table198261931819"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat(const AscendIndexVstarInitParams&amp; aModeInitParams, const AscendIndexGreatInitParams&amp; kModeInitParams);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">Initialization parameters required by the <code>Index</code>, specifically <code>aModeInitParams</code> and <code>kModeInitParams</code>. For details, see <code>AscendIndexVstarInitParams</code> and <code>AscendIndexGreatInitParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Refer to the parameter descriptions and constraints in <code>AscendIndexVstarInitParams</code> and <code>AscendIndexGreatInitParams</code>.<br>The <code>dim</code> values of <code>aModeInitParams</code> and <code>kModeInitParams</code> must be the same.</td></tr>
</tbody></table>

<a name="table32891532172215"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat(const AscendIndexGreat&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares this copy constructor as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexGreat&amp;</code>: A constant <code>AscendIndexGreat</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~AscendIndexGreat`<a name="ZH-CN_TOPIC_0000002013257524"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexGreat() = default;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The destructor of <code>AscendIndexGreat</code>. It destroys the <code>AscendIndexGreat</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000002008751990"></a>

<a name="table39961720122213"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat &amp;operator=(const AscendIndexGreat&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexGreat&amp;</code>: A constant <code>AscendIndexGreat</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Add`<a name="ZH-CN_TOPIC_0000002044950953"></a>

<a name="table11133547191811"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseRawData);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexGreat</code> base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseRawData</code>: The feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of <code>baseRawData</code> must be <code>dim * nTotal</code>. <code>nTotal</code> is the number of vectors to add to the base library, and <code>dim</code> is the dimension of each vector. The valid range of the total number of base library vectors is <code>10000 ≤ nTotal ≤ 1e8</code>. This algorithm does not support adding data again after the base library has been added. The <code>Add</code> API cannot be used together with the <code>AddWithIds</code> API.</td></tr>
</tbody></table>

### `AddWithIds`<a name="ZH-CN_TOPIC_0000002044829957"></a>

<a name="table2436200181918"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddWithIds (const std::vector&lt;float&gt;&amp; baseRawData, const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the AscendIndexGreat base index. When features are added through <code>AddWithIds</code>, the default IDs for the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseRawData</code>: Feature vectors to add to the base index.<br><code>const std::vector&lt;int64_t&gt;&amp; ids</code>: IDs of the feature vectors to add to the base index. IDs must be unique within the <code>Index</code> instance.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of the <code>baseRawData</code> array must be <code>dim * nTotal</code>. <code>nTotal</code> is the number of vectors to be added to the base index, and <code>dim</code> is the dimensionality of each vector. The total number of base vectors must satisfy <code>10000 ≤ nTotal ≤ 1e8</code>. The length of <code>ids</code> must be <code>nTotal</code>. Users must ensure the validity of <code>ids</code> according to their own business scenario. If duplicate IDs exist in the base index, the <code>label</code> in the search results cannot be mapped to a specific base vector. This algorithm does not support adding vectors after the base index has been built. The <code>AddWithIds</code> API cannot be used together with the <code>Add</code> API.</td></tr>
</tbody></table>

### `LoadIndex`<a name="ZH-CN_TOPIC_0000002008751978"></a>

<a name="table17789162191912"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads the <code>Index</code> structure from disk, including compressed, dimension-reduced feature vectors and codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; indexPath</code>: Path to load the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The file corresponding to <code>indexPath</code> must be a persisted file generated by calling <code>WriteIndex</code>, and the running user must have read permission for it. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

<a name="table98570373191"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and the original data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; aModeIndexPath</code>: Path to load the AMode index.<br><code>const std::string&amp; kModeIndexPath</code>: Path to load the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The files corresponding to <code>aModeIndexPath</code> and <code>kModeIndexPath</code> must be the persisted files generated by calling <code>WriteIndex</code>, and the running user must have read permission for them. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

### `WriteIndex`<a name="ZH-CN_TOPIC_0000002044950957"></a>

<a name="table84194504191"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>const std::string&amp; indexPath</code>: Path to write the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The user must ensure that the directory containing the <code>indexPath</code> file exists and that the running user has write permission for that directory. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

<a name="table14392122132014"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>const std::string&amp; aModeIndexPath</code>: Path to write the AMode index.<br><code>const std::string&amp; kModeIndexPath</code>: Path to write the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The user must ensure that the directories containing the <code>aModeIndexPath</code> and <code>kModeIndexPath</code> file paths exist and that the running user has write permission for those directories. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

### `AddCodeBooks`<a name="ZH-CN_TOPIC_0000002008751982"></a>

<a name="table339181620207"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooks(const std::string&amp; codeBooksPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads an already generated codebook into the <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; codeBooksPath</code>: Path to the generated codebook.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API can only be used when initializing an index in <code>AKMode</code>.<br>The user must ensure that the directory containing the <code>codeBooksPath</code> file exists, and the file content must be a valid codebook. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

### `Search`<a name="ZH-CN_TOPIC_0000002008910274"></a>

<a name="table537563852013"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(const AscendIndexSearchParams&amp; searchParams);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implements the AscendIndexGreat feature-vector search API. Based on the input feature vectors, it returns the distances and IDs of the most similar <code>topK</code> features.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">For the <code>searchParams</code> structure, see the <code>AscendIndexSearchParams</code> API.<br><code>size_t n</code>: Number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature-vector data.<br><code>int topK</code>: Number of most similar results to return.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>topK</code> ∈ (0, 4096]. <code>n</code> ∈ (0, 10000]. <code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>. <code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. <code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</td></tr>
</tbody></table>

### `SearchWithMask`<a name="ZH-CN_TOPIC_0000002044950961"></a>

<a name="table186956182018"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; searchParams, const std::vector&lt;uint8_t&gt;&amp; mask);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implements the AscendIndexGreat feature-vector search API. Based on the input feature vectors, it returns the distances and IDs of the most similar <code>topK</code> features. In addition, the user can input a <code>uint8</code> array to mask specific base-index IDs so that the feature vectors corresponding to those IDs are excluded from retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">For the <code>searchParams</code> structure, see the <code>AscendIndexSearchParams</code> API.<br><code>size_t n</code>: Number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature-vector data.<br><code>int topK</code>: Number of most similar results to return.<br><code>const std::vector&lt;uint8_t&gt;&amp; mask</code>: External filtering mask, in bits. 0 means the feature is filtered out; 1 means the feature is selected.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>topK</code> ∈ (0, 4096]. <code>n</code> ∈ (0, 10000]. <code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>. <code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. <code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. The total amount of data pointed to by <code>mask</code> must be greater than or equal to <code>n * ceil(nTotal / 8)</code>.</td></tr>
</tbody></table>

### `GetNTotal`<a name="ZH-CN_TOPIC_0000002044829965"></a>

<a name="table971712872115"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetNTotal (uint64_t&amp; nTotal) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the number of feature vectors that have been added to the AscendIndexGreat base index.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>uint64_t&amp; nTotal</code>: Number of feature vectors added to the base index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `GetDim`<a name="ZH-CN_TOPIC_0000002008751986"></a>

<a name="table113422226216"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetDim(int&amp; dim) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the dimensionality of the feature vectors added to the AscendIndexGreat base index.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>int&amp; dim</code>: Dimensionality of the feature vectors added to the base index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Reset`<a name="ZH-CN_TOPIC_0000002008910278"></a>

<a name="table1974793512118"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Reset();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clears the data stored in this <code>Index</code>, including compressed, dimension-reduced feature vectors and codebook data, while retaining the parameters entered when the user initialized the index.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `SetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002044950965"></a>

<a name="table1011347192118"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams&amp; params);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the hyperparameters used when searching this <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const AscendIndexHyperParams&amp; params</code>: Search hyperparameters. For details, see <code>AscendIndexHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `GetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002400547905"></a>

<a name="table749915518225"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetHyperSearchParams(AscendIndexHyperParams&amp; params) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the search hyperparameters used when searching this <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>AscendIndexHyperParams&amp; params</code>: Search hyperparameters. For details, see <code>AscendIndexHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendIndexMixSearchParams`<a name="ZH-CN_TOPIC_0000002008910258"></a>

### Overview<a name="ZH-CN_TOPIC_0000002045034929"></a>

The `AscendIndexMixSearchParams.h` file provides the structures required by `AscendIndexGreat` and `AscendIndexVStar`.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must acquire a lock before use; otherwise, the search API may cause exceptions. Sharing a single device across different threads is also not supported.

### `AscendIndexGreatInitParams`<a name="ZH-CN_TOPIC_0000002049404289"></a>

<a name="table17465519101616"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreatInitParams();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization parameter structure for KMode mode.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See <code>AscendIndexGreatInitParams</code> for default parameter values.</td></tr>
</tbody></table>

<a id="table10419189143817"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreatInitParams(int dim, int degree, int convPQM, int evaluationType, int expandingFactor);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization parameter structure for KMode mode.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>int dim</code>: Dimensionality of the feature vectors.<br><code>int degree</code>: Controls the fineness of the graph index during index construction. A larger value makes the graph index more fine-grained, requires more space, and yields higher retrieval accuracy.<br><code>int convPQM</code>: Number of PQ quantization vector segments.<br><code>int evaluationType</code>: Distance evaluation algorithm type; 0 represents IP and 1 represents L2.<br><code>int expandingFactor</code>: Number of neighbors connected when searching each layer during the initial graph-construction phase. Note that this is different from the retrieval-stage <code>expandingFactor</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {128, 256, 512, 1024}, default value: 256. <code>degree</code> ∈ [50, 100], default value: 50. <code>convPQM</code> must be at least 16, must be a multiple of 8, and must be divisible by <code>dim</code>; default value: 128. <code>evaluationType</code> ∈ {0, 1}, default value: 0. <code>expandingFactor</code> ∈ [200, 400], must be a multiple of 10; default value: 300.</td></tr>
</tbody></table>

### `AscendIndexVstarInitParams`<a name="ZH-CN_TOPIC_0000002013246410"></a>

<a name="table20955195613391"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVstarInitParams();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization parameter structure for Vstar mode.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See <code>AscendIndexVstarHyperParams</code> for default parameter values.</td></tr>
</tbody></table>

<a id="table899624214019"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVstarInitParams(int dim, int subSpaceDim, int nlist, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false, int64_t resourceSize = VSTAR_DEFAULT_MEM);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization parameter structure for Vstar mode.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>int dim</code>: Dimensionality of the feature vectors.<br><code>int subSpaceDim</code>: Dimensionality after the first dimensionality reduction.<br><code>int nlist</code>: Number of first-level clusters.<br><code>const std::vector&lt;int&gt;&amp; deviceList</code>: Specified NPU physical IDs.<br><code>bool verbose</code>: Whether to enable the <code>verbose</code> option. When enabled, some operations provide additional printed messages. Default value: <code>false</code>.<br><code>int64_t resourceSize</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>VSTAR_DEFAULT_MEM</code> defined in the header file, with a size of 128 MB. This parameter is determined jointly by the base index size and the <code>search</code> batch size.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {128, 256, 512, 1024}, default value: 1024.<br><code>subSpaceDim</code> ∈ {32, 64, 128}. <code>subSpaceDim</code> must be less than <code>dim</code>. Default value: 128.<br><code>nlist</code> ∈ {256, 512, 1024}. Default value: 1024.<br>For <code>deviceList</code>, use the <code>npu-smi</code> command to query the physical ID of the corresponding NPU card. Only one device ID is supported.<br><code>resourceSize</code> ∈ [128M, 2048M].</td></tr>
</tbody></table>

### `AscendIndexVstarHyperParams`<a name="ZH-CN_TOPIC_0000002013404694"></a>

<a name="table201855541164"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVstarHyperParams();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Hyperparameter structure for VSTAR mode.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See <code>AscendIndexVstarHyperParams</code> for default parameter values.</td></tr>
</tbody></table>

<a id="table42921559204019"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVstarHyperParams(int nProbeL1, int nProbeL2, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Hyperparameter structure for VSTAR mode.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>int nProbeL1</code>: Number of clusters searched in the first-stage retrieval.<br><code>int nProbeL2</code>: Number of clusters searched in the second-stage retrieval.<br><code>int l3SegmentNum</code>: Number of segments in the third-stage retrieval, that is, the number of data segments searched from <code>nProbeL2</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nProbeL1</code> ∈ [32, <code>nListL1</code>], and <code>nProbeL1</code> must be an integer multiple of 8. Default value: 72. <code>nProbeL2</code> ∈ (16, <code>nProbeL1</code> * <code>n</code>]; when <code>dim</code> is 1024, <code>n</code> is 16, and for other dimensions <code>n</code> is 32. <code>nProbeL2</code> must be an integer multiple of 8. Default value: 64. <code>l3SegmentNum</code> ∈ (100, 5000], and <code>l3SegmentNum</code> must be an integer multiple of 8. Default value: 512.</td></tr>
</tbody></table>

### `AscendIndexHyperParams`<a name="ZH-CN_TOPIC_0000002049325253"></a>

<a name="table93967711712"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexHyperParams();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Hyperparameter structure for GREAT retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See <code>AscendIndexHyperParams</code> for default parameter values.</td></tr>
</tbody></table>

<a id="table1334182412417"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexHyperParams(const std::string&amp; mode, const AscendIndexVstarHyperParams&amp; vstarHyperParam, int expandingFactor);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Hyperparameter structure for GREAT retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; mode</code>: Specifies the algorithm mode.<br><code>const AscendIndexVstarHyperParams&amp; vstarHyperParam</code>: For details, see <code>AscendIndexVstarHyperParams</code>.<br><code>int expandingFactor</code>: Number of neighbors searched at each layer during retrieval. Note that this differs from the <code>expandingFactor</code> used during graph construction.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>mode</code> ∈ {&quot;KMode&quot;, &quot;AKMode&quot;}. Default value: <code>AKMode</code>. <code>expandingFactor</code> ∈ [10, 200]. Default value: 150.</td></tr>
</tbody></table>

<a name="table88027219236"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexHyperParams(const std::string&amp; mode, int expandingFactor);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Hyperparameter structure for GREAT retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; mode</code>: Specifies the algorithm mode.<br><code>int expandingFactor</code>: Number of neighbors searched at each layer during retrieval. Note that this differs from the <code>expandingFactor</code> used during graph construction.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>mode</code> ∈ {&quot;KMode&quot;, &quot;AKMode&quot;}. Default value: <code>AKMode</code>. <code>expandingFactor</code> ∈ [10, 200]. Default value: 150.</td></tr>
</tbody></table>

### `AscendIndexSearchParams`<a name="ZH-CN_TOPIC_0000002044950949"></a>

<a name="table414612258177"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSearchParams(size_t n, std::vector&lt;float&gt;&amp; queryData, int topK, std::vector&lt;float&gt;&amp; dists, std::vector&lt;int64_t&gt;&amp; labels);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Search parameter structure for retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameters</td><td valign="middle"><code>size_t n</code>: Number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature-vector data.<br><code>int topK</code>: Number of most similar results to return.<br><code>std::vector&lt;float&gt;&amp; dists</code>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>topK</code> ∈ (0, 4096]. <code>n</code> ∈ (0, 10000]. <code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>. <code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. <code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</td></tr>
</tbody></table>

## `AscendIndexIVFFlat`<a name="ZH-CN_TOPIC_0000002478095516"></a>

### Overview<a name="ZH-CN_TOPIC_0000002510095475"></a>

`AscendIndexIVFFlat` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports only IP distance.

### `AscendIndexIVFFlat`<a name="ZH-CN_TOPIC_0000002509975505"></a>

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

### `~AscendIndexIVFFlat`<a name="ZH-CN_TOPIC_0000002477935546"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendIndexIVFFlat()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVFFlat</code>, which destroys the <code>AscendIndexIVFFlat</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000002484264062"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFFlat&amp;</code>: Constant <code>AscendIndexIVFFlat</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFFlat</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Currently only CPU clustering is supported, and <code>useKmeansPP</code> cannot be set to <code>true</code>.</td></tr>
</tbody></table>

## `AscendIndexIVFPQ`<a name="ZH-CN_TOPIC_0000002478095516"></a>

### Overview<a name="ZH-CN_TOPIC_0000002510095475"></a>

`AscendIndexIVFPQ` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports only L2 distance and, for performance reasons, only retrieval top-k values within 320.

### `AscendIndexIVFPQ`<a name="ZH-CN_TOPIC_0000002509975505"></a>

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

### `~AscendIndexIVFPQ`<a name="ZH-CN_TOPIC_0000002477935546"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendIndexIVFPQ()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVFPQ</code>, which destroys the <code>AscendIndexIVFPQ</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000002484264062"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFPQ&amp; operator=(const AscendIndexIVFPQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFPQ&amp;</code>: Constant <code>AscendIndexIVFPQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFPQ</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Setting <code>useKmeansPP</code> to <code>true</code> enables NPU clustering; otherwise CPU clustering is used.</td></tr>
</tbody></table>

### `remove_ids`<a name="ZH-CN_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void remove_ids(size_t n, const idx_t *ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Removes the trained vectors in <code>AscendIndexIVFPQ</code> corresponding to the provided index IDs, by calling the relevant APIs in <code>AscendIndexIVFPQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n</code>: Number of feature vectors to delete.<br><code>const idx_t *ids</code>: IDs of the feature vectors to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFPQ *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Reads trained data from the <code>IndexIVFPQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFPQ *index</code>: <code>IVFPQ</code> index, a type of index in the Faiss library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling this API, ensure that the data in <code>index</code> already has trained centroids and an inverted list, and that all parameters are complete.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(const faiss::IndexIVFPQ *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Saves the trained data into the <code>IndexIVFPQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFPQ *index</code>: <code>IVFPQ</code> index, a type of index in the Faiss library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling this API, ensure that the original vectors have been trained and added to the index, so that no empty centroids, codebooks, or inverted lists are read into <code>index</code>.</td></tr>
</tbody></table>

### `update`<a name="ZH-CN_TOPIC_0000002478095518"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; update(idx_t n, const float *x, idx_t *ids)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Batch-updates the vectors in the <code>AscendIndexIVFPQ</code> base index corresponding to <code>ids</code> to <code>x</code>. IDs that do not exist in the base index are not updated, and the list of missing IDs is returned.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to update.<br><code>float *x</code>: List of feature vectors to update.<br><code>idx_t *ids</code>: List of feature-vector IDs to update.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>std::vector&lt;idx_t&gt; noExistIds</code>: Returns the list of vector IDs that do not exist.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## `AscendIndexIVFRaBitQ`<a name="ZH-CN_TOPIC_0000002513157720"></a>

### Overview<a name="ZH-CN_TOPIC_0000002544797635"></a>

`AscendIndexIVFRaBitQ` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports L2 distance computation.

### `AscendIndexIVFRaBitQ`<a name="ZH-CN_TOPIC_0000002513317654"></a>

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

### `~AscendIndexIVFRaBitQ`<a name="ZH-CN_TOPIC_0000002544837623"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendIndexIVFRaBitQ()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVFRaBitQ</code>, which destroys the <code>AscendIndexIVFRaBitQ</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000002513157724"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFRaBitQ&amp; operator=(const AscendIndexIVFRaBitQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFRaBitQ&amp;</code>: Constant <code>AscendIndexIVFRaBitQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000002544797639"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFRaBitQ</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Setting <code>useKmeansPP</code> to <code>true</code> enables NPU clustering; otherwise CPU clustering is used. For precision issues, see floating-point computation precision issues.</td></tr>
</tbody></table>

### `remove_ids`<a name="ZH-CN_TOPIC_0000002513157728"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void remove_ids(size_t n, const idx_t* ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Removes the trained vectors in <code>AscendIndexIVFRaBitQ</code> corresponding to the provided index IDs, by calling the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n</code>: Number of feature vectors to delete.<br><code>const idx_t *ids</code>: IDs of the feature vectors to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000002557609263"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFRaBitQ *index)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Provides a CPU-side <code>IndexIVFRaBitQ</code> index, loads data from the trained index to the device side for subsequent retrieval, and calls the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFRaBitQ *index</code>: Trained CPU-side <code>IndexIVFRaBitQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The pointer <code>index</code> must be non-null, and it must point to a trained <code>IndexIVFRaBitQ</code> index. Before calling this API to read data, configure <code>AscendIndexIVFRaBitQConfig</code> and create an <code>AscendIndexIVFRaBitQ</code> object according to the normal procedure.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000002557689209"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFRaBitQ *index) const</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Provides a CPU-side <code>IndexIVFRaBitQ</code> index, downloads the trained data from the device side into the CPU index for persistence, and calls the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFRaBitQ *index</code>: Trained CPU-side <code>IndexIVFRaBitQ</code> index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The pointer <code>index</code> must be non-null. Before calling this API to persist data, create an <code>AscendIndexIVFRaBitQ</code> object and train it into the index according to the normal procedure.</td></tr>
</tbody></table>

### `update`<a name="ZH-CN_TOPIC_0000002566242121"></a>

<a name="table962730101715"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; update(idx_t n, float* x, idx_t* ids)</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Batch-updates the vectors in the <code>AscendIndexIVFRaBitQ</code> base index corresponding to <code>ids</code> to <code>x</code>. IDs that do not exist in the base index are not updated, and the list of missing IDs is returned.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to update.<br><code>float* x</code>: List of feature vectors to update.<br><code>idx_t *ids</code>: List of feature-vector IDs to update.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>std::vector&lt;idx_t&gt; noExistIds</code>: Returns the list of vector IDs that do not exist.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be non-null, and its length must be <code>n * dim</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>; otherwise, out-of-bounds read/write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## `AscendIndexIVFRaBitQConfig`<a name="ZH-CN_TOPIC_0000002544944511"></a>

`AscendIndexIVFRaBitQ` must use the corresponding `AscendIndexIVFRaBitQConfig` to initialize the relevant resources.

### `Member Overview`<a name="section4211138173219"></a>

<a name="table388535175015"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="210" align="center" valign="middle">useRandomOrthogonalMatrix</td><td valign="middle">bool</td><td valign="middle">Whether to use a random orthogonal matrix. Default: <code>true</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">needRefine</td><td valign="middle">bool</td><td valign="middle">Whether refinement is required. Default: <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">matrixSeed</td><td valign="middle">int</td><td valign="middle">Random seed used to generate the random orthogonal matrix. Default: 12345.</td></tr>
<tr><td width="210" align="center" valign="middle">refineAlpha</td><td valign="middle">float</td><td valign="middle">Refinement-related parameter. During retrieval, if the original plan is to retrieve the top <code>k</code>, refinement retrieves the top <code>k * refineAlpha</code> results first, and then takes the top <code>k</code> from them.<br>The default value is 2. A larger value gives higher recall but lower retrieval efficiency.</td></tr>
</tbody></table>

### `AscendIndexIVFRaBitQConfig`<a name="section6579185362314"></a>

>`Note:`
>`AscendIndexIVFRaBitQConfig` inherits from [AscendIndexIVFConfig](./approximate_retrieval.md#ascendindexivfconfig).

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
