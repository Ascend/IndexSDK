# `AscendIndex`<a id="en-us_TOPIC_0000001456375304"></a>

## Overview<a name="en-us_TOPIC_0000001506414937"></a>

AscendIndex is the base class of the `Index` implementations for most retrieval methods in the feature retrieval component. It sits on top of Faiss and defines interfaces for the other indexes in feature retrieval.

## `add`<a id="en-us_TOPIC_0000001506614985"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implements AscendIndex index creation and adds new feature vectors to the base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const float *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.<br><code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>.<br>Note:<br>The <code>add</code> interface cannot be used together with the <code>add_with_ids</code> interface. After you use the <code>add</code> interface, the <code>labels</code> in the search results may repeat. If your service has requirements for labels, you are advised to use the <code>add_with_ids</code> interface.</td></tr>
</tbody></table>

<a name="table17254342193617"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const uint16_t *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implements AscendIndex index creation and adds new feature vectors to the base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const uint16_t *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.<br><code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

## `add_with_ids`<a id="en-us_TOPIC_0000001456694864"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implements AscendIndex index creation and adds new feature vectors to the base library, with an ID for each base-library feature.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const float *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>. When the <code>filterable</code> filter switch is set to <code>true</code>, ensure that the timestamps in <code>ids</code> are positive.<br><code>ids</code> of type <code>uint64_t</code> contain <code>timestamp</code> of type <code>int32_t</code> and <code>cid</code> of type <code>uint8_t</code>, as shown below:<br> -----| cid | timestamp | ----- 14 | 8 | 32 | 10</td></tr>
</tbody></table>

<a name="table562574920111"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const uint16_t *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implements AscendIndex index creation and adds new feature vectors to the base library, with an ID for each base-library feature.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const uint16_t *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>. When the <code>filterable</code> filter switch is set to <code>true</code>, ensure that the timestamps in <code>ids</code> are positive. <code>ids</code> of type <code>uint64_t</code> contain <code>timestamp</code> of type <code>int32_t</code> and <code>cid</code> of type <code>uint8_t</code>, as shown below: -----| cid | timestamp | ----- 14 | 8 | 32 | 10</td></tr>
</tbody></table>

## `AscendIndex`<a name="en-us_TOPIC_0000001456695048"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndex(int dims, faiss::MetricType metric, AscendIndexConfig config)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndex</code>. It creates an <code>AscendIndex</code> with dimension <code>dims</code>. A single <code>Index</code> manages vectors with one fixed dimension. Device-side resources are set according to the values configured in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndex</code>.<br><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. Currently supported values are <code>faiss::MetricType::METRIC_L2</code> and <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.<br><code>AscendIndexConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> must be an integer in the range (0, 4096] and must be divisible by 16.</td></tr>
</tbody></table>

<a name="table161511529133912"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndex(const AscendIndex&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor of <code>AscendIndex</code> as deleted. Therefore, <code>AscendIndex</code> is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndex&amp;</code>: Constant <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table62621513124018"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndex();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndex</code>. It destroys the <code>AscendIndex</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getDeviceList`<a name="en-us_TOPIC_0000001506495857"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;int&gt; getDeviceList();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the device-side Ascend AI Processor configuration managed in <code>Index</code>. Derived classes provide the implementation. This class does not provide one and returns only an empty <code>vector&lt;int&gt;</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">Device-side Ascend AI Processor configuration managed in <code>Index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001506334661"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndex&amp; operator=(const AscendIndex&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of <code>AscendIndex</code> as deleted. Therefore, <code>AscendIndex</code> is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndex&amp;</code>: Constant <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reclaimMemory`<a name="en-us_TOPIC_0000001456695092"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual size_t reclaimMemory();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Reduces the memory occupied by the base library without changing the number of vectors in it. The implementation is inherited and provided by derived classes. This class does not provide an implementation.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">Size of the reclaimed memory, in bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `remove_ids`<a name="en-us_TOPIC_0000001456535000"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Removes the specified feature vectors from the base library in <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to be deleted. For details about usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">Number of deleted feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reserveMemory`<a name="en-us_TOPIC_0000001456375348"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void reserveMemory(size_t numVecs);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Abstract interface for reserving memory for the base library before it is built. The implementation is inherited and provided by derived classes. This class does not provide an implementation.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>size_t numVecs</code>: Number of vectors in the base library for which to reserve memory.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reset`<a name="en-us_TOPIC_0000001506414901"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Clears the base-library vectors of this <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `search`<a name="en-us_TOPIC_0000001506334641"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Feature-vector retrieval interface. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const float *x</code>: Feature-vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid distances with 65504 or -65504, depending on the metric.<br><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid labels with -1.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096.</td></tr>
<tr><td width="150" align="center" valign="middle">Note</td><td valign="middle">In scenarios that use the small-base-library brute-force algorithm, if performance drops when the base library and batch size are large, increase the <code>resources</code> parameter in <code>AscendIndexConfig</code>. The default value of the brute-force algorithm is 128 MB.</td></tr>
</tbody></table>
