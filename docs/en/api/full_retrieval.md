# Full Retrieval<a name="ZH-CN_TOPIC_0000001533164645"></a>

## `AscendIndex`<a id="ZH-CN_TOPIC_0000001456375304"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506414937"></a>

AscendIndex is the base class of the `Index` implementations for most retrieval methods in the feature retrieval component. It sits on top of Faiss and defines interfaces for the other indexes in feature retrieval.

### `add`<a id="ZH-CN_TOPIC_0000001506614985"></a>

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

### `add_with_ids`<a id="ZH-CN_TOPIC_0000001456694864"></a>

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

### `AscendIndex`<a name="ZH-CN_TOPIC_0000001456695048"></a>

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

### `getDeviceList`<a name="ZH-CN_TOPIC_0000001506495857"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;int&gt; getDeviceList();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the device-side Ascend AI Processor configuration managed in <code>Index</code>. Derived classes provide the implementation. This class does not provide one and returns only an empty <code>vector&lt;int&gt;</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">Device-side Ascend AI Processor configuration managed in <code>Index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001506334661"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndex&amp; operator=(const AscendIndex&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of <code>AscendIndex</code> as deleted. Therefore, <code>AscendIndex</code> is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndex&amp;</code>: Constant <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reclaimMemory`<a name="ZH-CN_TOPIC_0000001456695092"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual size_t reclaimMemory();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Reduces the memory occupied by the base library without changing the number of vectors in it. The implementation is inherited and provided by derived classes. This class does not provide an implementation.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">Size of the reclaimed memory, in bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `remove_ids`<a name="ZH-CN_TOPIC_0000001456535000"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Removes the specified feature vectors from the base library in <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to be deleted. For details about usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">Number of deleted feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reserveMemory`<a name="ZH-CN_TOPIC_0000001456375348"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void reserveMemory(size_t numVecs);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Abstract interface for reserving memory for the base library before it is built. The implementation is inherited and provided by derived classes. This class does not provide an implementation.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>size_t numVecs</code>: Number of vectors in the base library for which to reserve memory.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reset`<a name="ZH-CN_TOPIC_0000001506414901"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Clears the base-library vectors of this <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `search`<a name="ZH-CN_TOPIC_0000001506334641"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Feature-vector retrieval interface. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const float *x</code>: Feature-vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid distances with 65504 or -65504, depending on the metric.<br><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid labels with -1.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096.</td></tr>
<tr><td width="150" align="center" valign="middle">Note</td><td valign="middle">In scenarios that use the small-base-library brute-force algorithm, if performance drops when the base library and batch size are large, increase the <code>resources</code> parameter in <code>AscendIndexConfig</code>. The default value of the brute-force algorithm is 128 MB.</td></tr>
</tbody></table>

## `AscendIndexCluster`<a id="ZH-CN_TOPIC_0000001614744825"></a>

### Overview<a name="ZH-CN_TOPIC_0000001564586790"></a>

`AscendIndexCluster` requires [`Init`](#init) to initialize the specified resources. After initialization, it allocates a complete memory space to store the base library. After use, call [`Finalize`](#finalize) to release the resources.

`AscendIndexCluster` supports only the vector inner-product distance type in standard mode on Atlas Inference Series products. It depends on Flat and AICPU operators. For details, see [Flat](../user_guide.md#generating-operators) and [AICPU](../user_guide.md#generating-operators).

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `AddFeatures`<a name="ZH-CN_TOPIC_0000001614746533"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const float *features, const uint32_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Inserts <code>n</code> feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, this interface updates it.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to insert.<br><code>const float *features</code>: Feature vectors to insert. The length is <code>n</code> multiplied by the vector dimension <code>dim</code>.<br><code>const uint32_t *indices</code>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: The index of each feature must be in [0, <code>capacity</code> ), and <code>indices</code> must be continuous. <code>n</code>: Must be in (0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table772538154310"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const uint16_t *features, const int64_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Inserts <code>n</code> feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, this interface updates it.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to insert.<br><code>const uint16_t *features</code>: Feature vectors to insert. The length is <code>n</code> multiplied by the vector dimension <code>dim</code>.<br><code>const int64_t *indices</code>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: The index of each feature must be in [0, <code>capacity</code> ). <code>n</code>: Must be in (0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `AscendIndexCluster`<a name="ZH-CN_TOPIC_0000001564746410"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCluster();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexCluster</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table15621560282"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCluster(const AscendIndexCluster&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares this <code>Index</code> copy constructor as deleted. Therefore, the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexCluster&amp;</code>: <code>AscendIndexCluster</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~AscendIndexCluster`<a name="ZH-CN_TOPIC_0000002399598393"></a>

<a name="table179216322487"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexCluster() = default;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexCluster</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `ComputeDistanceByIdx`<a name="ZH-CN_TOPIC_0000002446061685"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num, const uint32_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle"><code>ComputeDistance</code> calculates the distance between the query vectors and all base-library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distance between the query vectors and the base-library vectors at the given indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the interface returns the mapped top-<code>k</code> results.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of query feature vectors.<br><code>const uint16_t *queries</code>: Query feature vectors. The valid length is <code>n * dim</code>, and <code>dim</code> must be the same as the dimension specified during initialization.<br><code>const int *num</code>: Number of base-library feature vectors to compare for each query. The length is <code>n</code>.<br><code>const uint32_t *indices</code>: Indices of the base-library feature vectors to compare. The number of base-library vectors to compare can differ for each query. Valid vector indices must be stored continuously from front to back, and the space usage must be padded according to the maximum <code>num</code>. The length of <code>indices</code> is <code>n * max(num)</code>.<br><code>unsigned int tableLen</code>: Mapping-table length. The default value is <code>0</code>, which means that no mapping is performed. Currently, the supported mapping-table length is <code>10000</code>.<br><code>const float *table</code>: Mapping-table pointer that points to valid mapped values of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>*table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distances between the query vectors and the selected base-library vectors. For each query, valid distances are recorded continuously from front to back, and the space usage is padded according to the maximum <code>num</code>. The total length is <code>n * max(num)</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: Must be in the range (0, <code>capacity</code> ]. <code>num</code>: User-specified. The length is <code>n</code>, and the <code>num</code> value for each query must be in [0, <code>ntotal</code>]. <code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ). Example parameter values: <code>n = 3</code>, <code>num[3] = {1, 3, 5}</code> means that the three queries compare against 1, 3, and 5 base-library vectors respectively. If <code>max(num) = 5</code>, then the space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>. When both <code>tableLen</code> and <code>table</code> meet the requirements, the interface maps the computed <code>distance</code> values.<br>First, normalize <code>distance</code> to a floating-point value <code>f1</code> in [0, 1]. Then multiply <code>f1</code> by <code>tableLen</code> and round it down to obtain an integer index in [0, <code>tableLen</code>]. Next, use the integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>. This completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be abstracted as <code>((CosDistance + 1) / 2) * tableLen</code>.</td></tr>
</tbody></table>

### `ComputeDistanceByThreshold`<a name="ZH-CN_TOPIC_0000001615066169"></a>

> This interface must be used together with [`AddFeatures(int n, const float *features, const uint32_t *indices);`](#addfeatures).

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByThreshold(const std::vector&lt;uint32_t&gt; &amp;queryIdxArr, uint32_t codeStartIdx, uint32_t codeNum, float threshold, bool aboveFilter, std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr, std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Calculates the distances between the queried feature vectors in the base library and the specified base-library feature vectors, then filters by threshold and returns the distances and labels that meet the conditions.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;uint32_t&gt; &amp;queryIdxArr</code>: Indices of the vectors to query in the base library.<br><code>uint32_t codeStartIdx</code>: Starting index of the base library vectors for distance calculation.<br><code>uint32_t codeNum</code>: Number of base-library vectors for distance calculation.<br><code>float threshold</code>: Threshold used for filtering. Distances smaller than the threshold are filtered out.<br><code>bool aboveFilter</code>: Reserved parameter.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr</code>: Two-dimensional array that returns the distances between each query vector and the base-library vectors that meet the threshold condition.<br><code>std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr</code>: Two-dimensional array that returns the indices of the base-library vectors that meet the threshold condition for each query vector.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The lengths of <code>queryIdxArr</code>, <code>resDistArr</code>, and <code>resIdxArr</code> must be the same, that is, <code>queryIdxArr.size() == resDistArr.size()</code>. <code>queryIdxArr.size()</code> must be greater than <code>0</code> and less than or equal to <code>ntotal</code>. <code>codeNum</code> must be greater than <code>0</code> and less than or equal to <code>ntotal</code>. <code>codeStartIdx + codeNum</code> must not exceed <code>ntotal</code> (the base-library size). <code>codeStartIdx</code> must be greater than or equal to <code>0</code> and less than or equal to <code>ntotal</code>.</td></tr>
</tbody></table>

### `Finalize`<a name="ZH-CN_TOPIC_0000001614906601"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle">void Finalize();</td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Releases feature-library management resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `GetFeatures`<a name="ZH-CN_TOPIC_0000002412742482"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, uint16_t *features, const int64_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Retrieves <code>n</code> feature vectors at the specified indices.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of base-library vectors to retrieve.<br><code>const int64_t *indices</code>: Indices corresponding to the feature vectors. The length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>uint16_t *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * vector dimension dim</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ), and <code>ntotal</code> can be obtained through the <code>GetNTotal</code> interface. <code>n</code>: Must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetNTotal`<a name="ZH-CN_TOPIC_0000002412582646"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int GetNTotal() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the theoretical maximum number of feature vectors in the current feature library. If the inserted feature-vector indices are continuous, <code>ntotal</code> is equal to the number of feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int ntotal</code>: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>int</code>: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Init`<a name="ZH-CN_TOPIC_0000001614866169"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initialization function of <code>AscendIndexCluster</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dim</code>: Dimension of the feature vectors managed by <code>AscendIndexCluster</code>.<br><code>int capacity</code>: Maximum base-library capacity. The interface allocates <code>capacity * dim * sizeof(fp16)</code> bytes of memory based on the value of <code>capacity</code>.<br><code>faiss::MetricType metricType</code>: Feature-distance category, including vector inner product, Euclidean distance, and cosine similarity.<br><code>const std::vector&lt;int&gt; &amp;deviceList</code>: Device-side resource configuration.<br><code>int64_t resourceSize</code>: Size of the preallocated memory pool on the device side, in bytes. This memory stores intermediate results during computation and is used to avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>-1</code>, which means <code>128 MB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> must be one of <code>{32, 64, 128, 256, 384, 512}</code>. <code>metricType</code>: <code>AscendIndexCluster</code> currently implements only vector inner-product distance, which means that only <code>faiss::MetricType::METRIC_INNER_PRODUCT</code> is supported. The maximum memory that can be allocated for the base library is <code>12,288,000,000</code> bytes, and the value range of <code>capacity</code> is [0, 12000000]. For example, for a base-library vector with 512 dimensions and the FP16 type, the maximum supported <code>capacity</code> is 12 million (<code>12288000000 / (512 * sizeof(fp_16))</code>). For base-library vectors with 256 dimensions and the FP16 type, even though the memory constraint supports a larger <code>capacity</code>, the maximum <code>capacity</code> can still be only 12 million. Only single-card configuration is supported. Multi-card configuration is not supported yet, so <code>deviceList.size()</code> must equal <code>1</code>. <code>resourceSize</code> can be <code>-1</code> or a value in [134217728, 4294967296], which is equivalent to <code>[128 MB, 4096 MB]</code>. This parameter is determined jointly by the base-library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</td></tr>
</tbody></table>

### `operator =`<a name="ZH-CN_TOPIC_0000001897100377"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCluster&amp; operator=(const AscendIndexCluster&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares this <code>Index</code> copy assignment operator as deleted, making the type non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexCluster&amp;</code>: <code>AscendIndexCluster</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `RemoveFeatures`<a name="ZH-CN_TOPIC_0000002446181741"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR RemoveFeatures(int n, const int64_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Removes <code>n</code> feature vectors at the specified indices from the vector library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to remove.<br><code>const int64_t *indices</code>: Indices corresponding to the feature vectors. The length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ), and <code>ntotal</code> can be obtained through the <code>GetNTotal</code> interface. <code>n</code>: Must be in [0, <code>capacity</code> ]. <code>indices</code> must be a non-null pointer, and its length must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `SearchByThreshold`<a name="ZH-CN_TOPIC_0000002446061689"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num, int64_t * indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the interface returns the mapped top-<code>k</code> results.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const uint16_t *queries</code>: Query feature vectors. The length is <code>n * dim</code>.<br><code>float threshold</code>: Threshold used for filtering. The interface does not restrict the value range. If you pass a mapping table, the interface first maps the distance to a score and then filters by <code>threshold</code>.<br><code>int topk</code>: Sorts the comparison distances between the query and the base library, then returns the top <code>k</code> results.<br><code>unsigned int tableLen</code>: Mapping-table length. The default value is <code>0</code>, which means that no mapping is performed. Currently, the supported mapping-table length is <code>10000</code>.<br><code>const float *table</code>: Mapping-table pointer that points to valid mapped values of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>*table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int *num</code>: Number of base-library vectors that meet the threshold condition for each query feature vector. The length is <code>n</code>.<br><code>int64_t *indices</code>: Indices of base-library vectors that meet the threshold condition. For each query, matching indices are recorded from front to back and the space is padded according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.<br><code>float *distances</code>: Distances between the base-library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: Must be in the range (0, <code>capacity</code> ]. <code>topk</code>: <code>k</code> must be in (0, 1024]. When both <code>tableLen</code> and <code>table</code> meet the requirements, the interface maps the computed <code>distance</code> values.<br>First, normalize <code>distance</code> to a floating-point value <code>f1</code> in [0, 1]. Then multiply <code>f1</code> by <code>tableLen</code> and round it down to obtain an integer index in [0, <code>tableLen</code>]. Next, use the integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>. This completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be abstracted as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `SetNTotal`<a name="ZH-CN_TOPIC_0000002412742486"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetNTotal(int n);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Provides an external way to adjust the <code>ntotal</code> count.<br>After base-library vectors are added, the <code>Index</code> internally updates <code>ntotal</code> according to the maximum inserted index. However, it does not record which areas in the range [0, <code>ntotal</code> ] are invalid space. Therefore, the <code>RemoveFeatures</code> operation does not change the value of <code>ntotal</code>. If you explicitly record the maximum base-library index after add and remove operations in the service layer, you can set <code>ntotal</code> manually. This can reduce the amount of work performed by the operators within a controllable range and improve interface performance.<br>For example, if you currently insert 100 vectors with base-library indices from 0 to 99, then <code>ntotal = 100</code>. If you delete the base-library vectors with indices from 80 to 90, the internal <code>ntotal</code> of <code>Index</code> remains unchanged and can only be set to a value in [ <code>ntotal</code>, <code>capacity</code> ]. If you then delete the base-library vectors with indices from 90 to 99, you can manually set <code>ntotal</code> to a value in [80, <code>capacity</code> ]. When you set it to <code>80</code>, the amount of base-library data participating in the comparison is effectively reduced by 20 vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Maximum base-library index plus 1, managed by the user in the service layer.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: Must be in the range [0, <code>capacity</code> ].</td></tr>
</tbody></table>

## `AscendIndexConfig`<a name="ZH-CN_TOPIC_0000001506414705"></a>

`AscendIndex` must use the corresponding `AscendIndexConfig` to initialize the relevant resources. `AscendIndexConfig` must configure the hardware resources and memory pool size used during retrieval.

> [!NOTE]
> The memory pool size unit is `Byte`. This parameter specifies the size of the preallocated memory pool on the device side. The memory pool stores the results of distance calculations on Ascend hardware. When the base library is large, you are advised to reserve a larger memory pool.

**Members<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="150" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="150" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device-side device IDs.</td></tr>
<tr><td width="150" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">Device-side memory pool size, in bytes. The default parameter is <code>INDEX_DEFAULT_MEM</code> in the header file.</td></tr>
<tr><td width="150" align="center" valign="middle">slim</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexConfig</code>. Indicates whether to increase memory dynamically.</td></tr>
<tr><td width="150" align="center" valign="middle">filterable</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexConfig</code>. Indicates whether to filter by ID.</td></tr>
<tr><td width="150" align="center" valign="middle">dBlockSize</td><td valign="middle">uint32_t</td><td valign="middle">Device-side block size configuration.</td></tr>
</tbody></table>

**API Description<a name="section1197816229504"></a>**

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexConfig()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Default constructor of <code>AscendIndexConfig</code>. The default <code>deviceList</code> is <code>0</code>, which means that the Ascend AI Processor with ID <code>0</code> on the NPU is used as the heterogeneous computing platform for <code>AscendFaiss</code> retrieval. The default resource-pool size is <code>32 MB</code> (<code>32*1024*1024</code> bytes).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table0786126165110"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexConfig</code>. It creates an <code>AscendIndexConfig</code> and sets device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, while also configuring the resource-pool size.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>INDEX_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Device-side block size configuration. It constrains the amount of data processed in one <code>tik</code> operator call and the size of vectors stored in each partition of the base-library shard. The default value of <code>DEFAULT_BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The maximum number is 64. The configured value of <code>resources</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

<a name="table23967285518"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexConfig(std::vector&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexConfig</code>. It creates an <code>AscendIndexConfig</code> and sets device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, while also configuring the resource-pool size.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>INDEX_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Device-side block size configuration. It constrains the amount of data processed in one <code>tik</code> operator call and the size of vectors stored in each partition of the base-library shard. The default value of <code>DEFAULT_BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. The maximum number is 64. The configured value of <code>resources</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

## `AscendIndexInt8`<a id="ZH-CN_TOPIC_0000001506495841"></a>

### Overview<a id="ZH-CN_TOPIC_0000001506495913"></a>

`AscendIndexInt8` is the base class of the indexes that use INT8 feature vectors in the feature retrieval component. It defines interfaces for other INT8 indexes in feature retrieval.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must lock before use, or the retrieval interface may raise exceptions. It also does not support sharing one device across different threads. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `add`<a name="ZH-CN_TOPIC_0000001506334825"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const int8_t *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

<a name="table6211414109"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const char *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const char *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

> [!NOTE]
>
>- The `add` interface cannot be used together with the `add_with_ids` interface.
>- After you use the `add` interface, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` interface.

### `add_with_ids`<a name="ZH-CN_TOPIC_0000001506614905"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const int8_t *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library and specifies the feature IDs.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const int8_t *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library. The IDs must be unique within the <code>Index</code> instance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

<a name="table38814511704"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const char *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library and specifies the feature IDs.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const char *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *ids</code>: IDs corresponding to the feature vectors to add to the base library. The IDs must be unique within the <code>Index</code> instance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

### `assign`<a name="ZH-CN_TOPIC_0000001506495721"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Feature-vector retrieval interface of <code>AscendIndexInt8</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const int8_t *x</code>: Feature-vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the length of <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be greater than <code>0</code> and less than <code>1e9</code>. <code>k</code> must be greater than <code>0</code> and less than or equal to <code>4096</code>. <code>n * k</code> must be less than <code>1e10</code>.</td></tr>
</tbody></table>

### `AscendIndexInt8`<a name="ZH-CN_TOPIC_0000001506614993"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config)`;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8</code>. It creates an <code>AscendIndexInt8</code> with dimension <code>dims</code>. The dimension of the vector set managed by a single <code>Index</code> is unique. Device-side resources are set according to the values configured in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexInt8</code>.<br><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndexInt8</code> when performing feature-vector similarity retrieval. Currently supported values are <code>faiss::MetricType::METRIC_L2</code> and <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.<br><code>AscendIndexInt8Config config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> must be an integer that is not smaller than 64 and not larger than 1024, and it must be divisible by 64.</td></tr>
</tbody></table>

<a name="table103312407520"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8(const AscendIndexInt8&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares this <code>Index</code> copy constructor as deleted. Therefore, the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8&amp;</code>: <code>AscendIndexInt8</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table1882220715614"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexInt8();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexInt8</code>. It destroys the <code>AscendIndexInt8</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getDeviceList`<a name="ZH-CN_TOPIC_0000001672982421"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;int&gt; getDeviceList() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Return the device-side Ascend AI Processor settings managed by <code>Index</code>. Subclasses inherit from it and implement it. This base class does not provide a corresponding implementation and returns only an empty <code>vector&lt;int&gt;</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The device-side Ascend AI Processor settings managed by <code>Index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getDim`<a name="ZH-CN_TOPIC_0000001690599922"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDim() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the dimension of the feature vector set managed by <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The dimension of the feature vector set managed by <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getNTotal`<a name="ZH-CN_TOPIC_0000001738718517"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::idx_t getNTotal() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the number of feature vectors that <code>AscendIndexInt8</code> has added to the base vector set.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of feature vectors that <code>AscendIndexInt8</code> has added to the base vector set.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getMetricType`<a name="ZH-CN_TOPIC_0000001738678653"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::MetricType getMetricType() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the distance metric type used by <code>AscendIndexInt8</code> when performing feature vector similarity retrieval.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The distance metric type used by <code>AscendIndexInt8</code> when performing feature vector similarity retrieval.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `isTrained`<a name="ZH-CN_TOPIC_0000001690759666"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>bool isTrained() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Determine whether <code>AscendIndexInt8</code> is trained.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The trained state of <code>AscendIndexInt8</code>. <code>true</code> means trained, and <code>false</code> means not trained.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `operator =`<a name="ZH-CN_TOPIC_0000001506414841"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8&amp; operator=(const AscendIndexInt8&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8&amp;</code>: A constant <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reclaimMemory`<a name="ZH-CN_TOPIC_0000001506615133"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual size_t reclaimMemory();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `remove_ids`<a name="ZH-CN_TOPIC_0000001456695088"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implement the interface for deleting the specified feature vectors from the base vector set in <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to delete. For details on usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of deleted feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reserveMemory`<a name="ZH-CN_TOPIC_0000001506615065"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void reserveMemory(size_t numVecs);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>size_t numVecs</code>: Number of base vectors for which to reserve memory.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `search`<a name="ZH-CN_TOPIC_0000001506414889"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implement the feature vector search interface for <code>AscendIndexInt8</code>, and return the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const int8_t *x</code>: Feature vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid distances with <code>65504</code> or <code>-65504</code> depending on the metric.<br><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid labels with <code>-1</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of the query feature vector data <code>x</code> should be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, <code>n</code> is greater than <code>0</code> and less than <code>1e9</code>. Here, <code>k</code> is greater than <code>0</code> and less than or equal to <code>4096</code>.</td></tr>
</tbody></table>

<a name="table88671631181418"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implement the feature vector search interface for <code>AscendIndexInt8</code>, and return the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const char *x</code>: Feature vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of the query feature vector data <code>x</code> should be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, <code>n</code> is greater than <code>0</code> and less than <code>1e9</code>. Here, <code>k</code> is greater than <code>0</code> and less than or equal to <code>4096</code>.</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000001456534956"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void train(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const int8_t *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `updateCentroids`<a name="ZH-CN_TOPIC_0000001506414833"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void updateCentroids(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const int8_t *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table2023134918146"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void updateCentroids(idx_t n, const char *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const char *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendIndexInt8Config`<a id="ZH-CN_TOPIC_0000001456854968"></a>

`AscendIndexInt8` requires the corresponding `AscendIndexInt8Config` to initialize the associated resources.

`Member Description`<a name="section1372191465013"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="150" align="center" valign="middle"><code>deviceList</code></td><td valign="middle"><code>std::vector&lt;int&gt;</code></td><td valign="middle">Device-side device ID list.</td></tr>
<tr><td width="150" align="center" valign="middle"><code>resourceSize</code></td><td valign="middle"><code>int64_t</code></td><td valign="middle">Preallocated memory pool size on the device side, in bytes.</td></tr>
</tbody></table>

`API Description`<a name="section135441937164218"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Config()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Default constructor of <code>AscendIndexInt8Config</code>. The default <code>deviceList</code> is <code>0</code>, which means Ascend AI Processor 0 on the NPU is used as the heterogeneous computing platform for AscendFaiss retrieval. The default resource pool size is used.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table012165162914"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Config(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8Config</code>. It creates an <code>AscendIndexInt8Config</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INDEX_INT8_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs, and the maximum number is 64. The configured <code>resources</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes).</td></tr>
</tbody></table>

<a name="table9202719152913"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Config(std::vector&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8Config</code>. It creates an <code>AscendIndexInt8Config</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resources</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INDEX_INT8_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs, and the maximum number is 64. The configured <code>resources</code> value must not exceed 16 \* 1024 MB (16 \* 1024 \* 1024 \* 1024 bytes).</td></tr>
</tbody></table>

## `AscendIndexInt8Flat`<a name="ZH-CN_TOPIC_0000001506334741"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506615033"></a>

`AscendIndexInt8Flat` stores `INT8` feature vectors and performs brute-force search.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval uses OMP internally for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexInt8Flat`<a name="ZH-CN_TOPIC_0000001456375168"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8Flat</code>. It creates an <code>AscendIndexInt8</code> with dimension <code>dims</code>. The dimension of the vector set managed by a single <code>Index</code> is unique. It configures device-side resources according to the values in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of the feature vector set managed by <code>AscendIndexInt8</code>.<br><code>faiss::MetricType metric</code>: Distance metric type used by <code>AscendIndex</code> when performing feature vector similarity retrieval.<br><code>AscendIndexInt8FlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {64, 128, 256, 384, 512, 768, 1024}. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</td></tr>
</tbody></table>

<a name="table08035919302"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8Flat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexScalarQuantizer *index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexInt8FlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. It must be a pointer of the <code>faiss::IndexScalarQuantizer</code> type generated by the <code>copyTo</code> interface of <code>AscendIndexInt8Flat</code>.</td></tr>
</tbody></table>

<a name="table11312020103012"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Flat(const faiss::IndexIDMap *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8Flat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexInt8FlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. It must be a pointer of the <code>faiss::IndexIDMap</code> type generated by the <code>copyTo</code> interface of <code>AscendIndexInt8Flat</code>.</td></tr>
</tbody></table>

<a name="table186285584308"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Flat(const AscendIndexInt8Flat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> copy constructor as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8Flat&amp;</code>: A constant <code>AscendIndexInt8Flat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table206471151315"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexInt8Flat();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexInt8Flat</code>. It destroys the <code>AscendIndexInt8Flat</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456375340"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIDMap* index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>index</code> to Ascend based on <code>AscendIndexInt8Flat</code>, and keep the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The dimension <code>d</code> parameter of the member index of this <code>Index</code> must be in the range {64, 128, 256, 384, 512, 768, 1024}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>.</td></tr>
</tbody></table>

<a name="table862731073217"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexScalarQuantizer* index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>index</code> to Ascend based on <code>AscendIndexInt8Flat</code>, and keep the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexScalarQuantizer* index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The dimension <code>d</code> parameter of the <code>Index</code> must be in the range {64, 128, 256, 384, 512, 768, 1024}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000001506334805"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexScalarQuantizer* index) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copy the retrieval resources of <code>AscendIndexInt8Flat</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexScalarQuantizer* index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The resources occupied by <code>Index</code> are freed by the user.</td></tr>
</tbody></table>

<a name="table1981952413329"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIDMap* index) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copy the retrieval resources of <code>AscendIndexInt8Flat</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The resources occupied by <code>Index</code> are freed by the user.</td></tr>
</tbody></table>

### `getBase`<a name="ZH-CN_TOPIC_0000001506334753"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getBase(int deviceId, std::vector&lt;int8_t&gt; &amp;xb) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the feature vectors managed by this <code>AscendIndexInt8Flat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;int8_t&gt; &amp;xb</code>: Base feature vectors stored by <code>AscendIndexInt8Flat</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

### `getBaseSize`<a name="ZH-CN_TOPIC_0000001506414709"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t getBaseSize(int deviceId) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the number of feature vectors managed by this <code>AscendIndexInt8Flat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of feature vectors on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

### `getIdxMap`<a name="ZH-CN_TOPIC_0000001506495853"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the feature vector IDs managed by this <code>AscendIndexInt8Flat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;idx_t&gt; &amp;idxMap</code>: Base feature vector IDs stored by <code>AscendIndexInt8Flat</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

### `operator =`<a name="ZH-CN_TOPIC_0000001506414909"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8Flat&amp; operator=(const AscendIndexInt8Flat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8Flat&amp;</code>: A constant <code>AscendIndexInt8Flat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `reset`<a name="ZH-CN_TOPIC_0000001506495889"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Clear the base vectors in this <code>AscendIndexInt8Flat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `search_with_masks`<a name="ZH-CN_TOPIC_0000001456694912"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implement the feature vector search interface for <code>AscendIndexInt8</code>, and return the distances and IDs of the <code>k</code> most similar features based on the input feature vectors and the <code>mask</code>. The mask is a <code>0</code>/<code>1</code> bit string. Each bit indicates whether the corresponding feature in the base vector set participates in distance computation. <code>1</code> means participate, and <code>0</code> means not participate.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const int8_t* x</code>: Feature vector data.<br><code>idx_t k</code>: Number of most similar results to return.<br><code>const void* mask</code>: Base vector set filter mask.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value range of <code>n</code> is <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>. <code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and their lengths should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>mask</code> must be a non-null pointer, and the length of the passed mask must be <code>ntotal / 8 * n</code> (<code>ntotal</code> is the number of vectors in the base vector set). The mask is set in the order of the base vector set. If <code>remove_ids</code> is called before this interface, the order of base vectors changes. Therefore, call <code>getIdxMap</code> first to obtain the IDs of the base vectors, and then set the mask. This interface requires the base vector set to be stored on a single device. Otherwise, the filtering result may be incorrect.</td></tr>
</tbody></table>

### `setPageSize`<a name="ZH-CN_TOPIC_0000002007453769"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setPageSize(uint16_t pageBlockNum);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Set the number of base-vector blocks that this <code>AscendIndexInt8Flat</code> computes consecutively in one <code>search</code> call.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>uint16_t pageBlockNum</code>: Number of base-vector blocks to compute consecutively in one call. If you do not set this parameter, the default is to compute 16 blocks consecutively at a time. The size of one block is determined by <code>blockSize</code> in <code>AscendIndexInt8FlatConfig</code>. The larger the value, the more memory <code>search</code> uses.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>pageBlockNum</code> is <code>0 &lt; pageBlockNum ≤ 144</code>. This interface is mainly used for large base vector set scenarios and for performance tuning of the <code>search</code> interface. The larger the value, the more preallocated memory configured by <code>resourceSize</code> in <code>AscendIndexInt8FlatConfig</code> it consumes. You are advised to request enough preallocated memory first and then use this interface to tune parameters.</td></tr>
</tbody></table>

## `AscendIndexInt8FlatConfig`<a name="ZH-CN_TOPIC_0000001456535040"></a>

`AscendIndexInt8Flat` requires the corresponding `AscendIndexInt8FlatConfig` to initialize the associated resources.

`Member Description`<a name="section1372191465013"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="150" align="center" valign="middle"><code>dIndexMode</code></td><td valign="middle"><code>Int8IndexMode</code></td><td valign="middle">Configures the INT8 retrieval mode for the <code>Index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle"><code>dBlockSize</code></td><td valign="middle"><code>uint32_t</code></td><td valign="middle">Configures the device-side <code>blockSize</code>.</td></tr>
</tbody></table>

`API Description`<a name="section136272015172914"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8FlatConfig(uint32_t blockSize =BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8FlatConfig</code>. It creates an <code>AscendIndexInt8FlatConfig</code>, configures the device-side <code>blockSize</code>, and configures the INT8 retrieval mode.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>uint32_t blockSize</code>: Configures the device-side <code>blockSize</code>. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.<br><code>Int8IndexMode indexMode</code>: Configures the INT8 retrieval mode for the <code>Index</code>. The default value is <code>DEFAULT_MODE</code>.<br> <code>DEFAULT_MODE</code>: Default mode. <code>PIPE_SEARCH_MODE</code>: This mode is optimized for scenarios where the batch is greater than or equal to <code>128</code>. When you use this mode, you are advised to set <code>resourceSize</code> to at least <code>1324 MB</code>. <code>WITHOUT_NORM_MODE</code>: This mode is not supported at this time.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The set of valid <code>blockSize</code> values is <code>{16384, 32768, 65536, 131072, 262144}</code>. In <code>PIPE_SEARCH_MODE</code>, <code>AscendIndexInt8Flat</code> supports only <code>METRIC_L2</code>.</td></tr>
</tbody></table>

<a name="table1258103643012"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8FlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8FlatConfig</code>. It creates an <code>AscendIndexInt8FlatConfig</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>. It also configures the device-side <code>blockSize</code> and the INT8 retrieval mode.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INT8_FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br><code>uint32_t blockSize</code>: Configures the device-side <code>blockSize</code>. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.<br><code>Int8IndexMode indexMode</code>: Configures the INT8 retrieval mode for the <code>Index</code>. The default value is <code>DEFAULT_MODE</code>.<br> <code>DEFAULT_MODE</code>: Default mode. <code>PIPE_SEARCH_MODE</code>: This mode is optimized for scenarios where the batch is greater than or equal to <code>128</code>. When you use this mode, you are advised to set <code>resourceSize</code> to at least <code>1324 MB</code>. <code>WITHOUT_NORM_MODE</code>: This mode is not supported at this time.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs, and the maximum number is 64. The configured <code>resourceSize</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes). When the batch is greater than or equal to <code>96</code>, you are advised to set <code>resourceSize</code> to at least <code>2 * 1024 MB</code> to improve algorithm performance. The set of valid <code>blockSize</code> values is <code>{16384, 32768, 65536, 131072, 262144}</code>. In <code>PIPE_SEARCH_MODE</code>, <code>AscendIndexInt8Flat</code> supports only <code>METRIC_L2</code>.</td></tr>
</tbody></table>

<a name="table8629135217302"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8FlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8FlatConfig</code>. It creates an <code>AscendIndexInt8FlatConfig</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>. It also configures the device-side <code>blockSize</code> and the INT8 retrieval mode.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INT8_FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br><code>uint32_t blockSize</code>: Configures the device-side <code>blockSize</code>. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.<br><code>Int8IndexMode indexMode</code>: Configures the INT8 retrieval mode for the <code>Index</code>. The default value is <code>DEFAULT_MODE</code>.<br> <code>DEFAULT_MODE</code>: Default mode. <code>PIPE_SEARCH_MODE</code>: This mode is optimized for scenarios where the batch is greater than or equal to <code>128</code>. When you use this mode, you are advised to set <code>resourceSize</code> to at least <code>1324 MB</code>. <code>WITHOUT_NORM_MODE</code>: This mode is not supported at this time.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs, and the maximum number is 64. The configured <code>resourceSize</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes). When the batch is greater than or equal to <code>96</code>, you are advised to set <code>resourceSize</code> to at least <code>2 * 1024 MB</code> to improve algorithm performance. The set of valid <code>blockSize</code> values is <code>{16384, 32768, 65536, 131072, 262144}</code>. In <code>PIPE_SEARCH_MODE</code>, <code>AscendIndexInt8Flat</code> supports only <code>METRIC_L2</code>.</td></tr>
</tbody></table>

## `AscendIndexFlat`<a id="ZH-CN_TOPIC_0000001506334757"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506334829"></a>

`AscendIndexFlat` is the most basic feature retrieval algorithm. It stores FP16 floating-point feature vectors and performs brute-force search.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval uses OMP internally for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> `AscendIndexFlat` supports online operator conversion for L2 and IP distances. If the environment variable `MX_INDEX_USE_ONLINEOP` is set to `1` (set it with `export MX_INDEX_USE_ONLINEOP=1`), the operator is converted and called online. To use online operators, the application must explicitly call `(void)aclFinalize()` at the end. You also need to include the header file `#include "acl/acl.h"`.

### `AscendIndexFlat`<a name="ZH-CN_TOPIC_0000001456375308"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexFlat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>.</td></tr>
</tbody></table>

<a name="table1735274911381"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlat(const faiss::IndexIDMap *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexFlat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>.</td></tr>
</tbody></table>

<a name="table142416323911"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexFlat</code>. It creates an <code>AscendIndexFlat</code> with dimension <code>dims</code>. The dimension of the vector set managed by a single <code>Index</code> is unique. It configures device-side resources according to the values in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of the feature vector set managed by <code>AscendIndex</code>.<br><code>faiss::MetricType metric</code>: Distance metric type used by <code>AscendIndexFlat</code> when performing feature vector similarity retrieval.<br><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</td></tr>
</tbody></table>

<a name="table5169814143913"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlat(const AscendIndexFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> copy constructor as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexFlat&amp;</code>: A constant <code>AscendIndexFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table04891725153918"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexFlat();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexFlat</code>. It destroys the <code>AscendIndexFlat</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456535180"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>Index</code> to Ascend based on <code>AscendIndexFlat</code>, clear the current base vector set in <code>AscendIndexFlat</code>, and keep the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>.</td></tr>
</tbody></table>

<a name="table525914213409"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIDMap *index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>index</code> to Ascend based on <code>AscendIndexFlat</code>, clear the current base vector set in <code>AscendIndexFlat</code>, and keep the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexIDMap</code> pointer. Otherwise, the program may crash or the function may become unavailable. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000001456535148"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexFlat *index) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexFlat</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user must free the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

<a name="table154531752144016"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIDMap *index) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexFlat</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The user must free the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

### `getBase`<a name="ZH-CN_TOPIC_0000001456375236"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getBase(int deviceId, char* xb) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vectors managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>char* xb</code>: The base library feature vectors stored by <code>AscendIndexFlat</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.<br><code>xb</code> must be a non-null pointer, and its length must be <code>dims * BaseSize * sizeof(float32)</code> bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>BaseSize</code> is the return value of <code>getBaseSize</code>.</td></tr>
</tbody></table>

### `getBaseSize`<a name="ZH-CN_TOPIC_0000001456854956"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t getBaseSize(int deviceId) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the number of feature vectors managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of feature vectors on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

### `getIdxMap`<a name="ZH-CN_TOPIC_0000001506334785"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vector IDs managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;idx_t&gt; &amp;idxMap</code>: The base library feature vector IDs stored by <code>AscendIndexFlat</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001506495701"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlat&amp; operator=(const AscendIndexFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexFlat&amp;</code>: A constant <code>AscendIndexFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `search_with_masks`<a name="ZH-CN_TOPIC_0000001810529650"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The feature vector query API of <code>AscendIndexFlat</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. <code>mask</code> is a bit string of <code>0</code>s and <code>1</code>s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. <code>1</code> means participate, and <code>0</code> means do not participate.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of query feature vectors.<br><code>const float *x</code>: Feature vector data.<br><code>idx_t k</code>: The number of most similar results to return.<br><code>const void *mask</code>: Feature library mask.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>mask</code> must be a non-null pointer, and its length must be <code>n * ceil(ntotal / 8)</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>ntotal</code> is the number of base library features. <code>mask</code> is set according to the order of the base library. If you call <code>remove_ids</code> to delete feature vectors before calling this API, the order of the base library features changes. First call <code>getIdxMap</code> to obtain the IDs of the base library features, and then set <code>mask</code>. To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect.</td></tr>
</tbody></table>

<a name="table0628133121511"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The feature vector query API of <code>AscendIndexFlat</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. <code>mask</code> is a bit string of <code>0</code>s and <code>1</code>s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. <code>1</code> means participate, and <code>0</code> means do not participate.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of query feature vectors.<br><code>const uint16_t *x</code>: Feature vector data.<br><code>idx_t k</code>: The number of most similar results to return.<br><code>const void *mask</code>: Feature library mask.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>mask</code> must be a non-null pointer, and its length must be <code>n * ceil(ntotal / 8)</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>ntotal</code> is the number of base library features. <code>mask</code> is set according to the order of the base library. If you call <code>remove_ids</code> to delete feature vectors before calling this API, the order of the base library features changes. First call <code>getIdxMap</code> to obtain the IDs of the base library features, and then set <code>mask</code>. To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect.</td></tr>
</tbody></table>

## `AscendIndexFlatConfig`<a name="ZH-CN_TOPIC_0000001456375216"></a>

`AscendIndexFlat` requires the corresponding `AscendIndexFlatConfig` to initialize the corresponding resources.

**API Description**<a name="section140920164419"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The default constructor of <code>AscendIndexFlatConfig</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table46951722104415"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatConfig</code>. It creates an <code>AscendIndexFlatConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>4194304</code> and the batch size is greater than or equal to <code>16</code>, use the following recommendations.<br>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_L2</code>, the recommended value is <code>1024 MB</code>. When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_INNER_PRODUCT</code>, the recommended value is <code>1280 MB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</td></tr>
</tbody></table>

<a name="table842319354444"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatConfig</code>. It creates an <code>AscendIndexFlatConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>4194304</code> and the batch size is greater than or equal to <code>16</code>, use the following recommendations.<br>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_L2</code>, the recommended value is <code>1024 MB</code>. When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_INNER_PRODUCT</code>, the recommended value is <code>1280 MB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</td></tr>
</tbody></table>

## `AscendIndexFlatL2`<a name="ZH-CN_TOPIC_0000001456375424"></a>

### Overview<a name="ZH-CN_TOPIC_0000001877955534"></a>

`AscendIndexFlatL2` is a brute-force feature retrieval algorithm that stores FP16 floating-point values and uses the L2 distance.

It supports multithreaded concurrent calls. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> The `AscendIndexFlatL2` algorithm supports online operator conversion. If the environment variable `MX_INDEX_USE_ONLINEOP` is set to `1` (`export MX_INDEX_USE_ONLINEOP=1`), it converts the operators online and calls them. To use online operators, the user must explicitly call `(void)aclFinalize()` at the end of the application. The header file `#include "acl/acl.h"` is required.

### `AscendIndexFlatL2`<a name="ZH-CN_TOPIC_0000001506495761"></a>

<a name="zh-cn_topic_0000001294312541_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatL2</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexFlatL2 *index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is <code>{32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 3584, 4096}</code>. The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be <code>faiss::MetricType::METRIC_L2</code>.</td></tr>
</tbody></table>

<a name="zh-cn_topic_0000001294591937_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexFlatL2</code>. It creates an <code>AscendIndexFlatL2</code> with dimension <code>dims</code>. The dimension of a vector set managed by one <code>Index</code> is unique. It then sets Device-side resources according to the values configured in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: The dimension of a set of feature vectors managed by <code>AscendIndexFlatL2</code>.<br><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 4096, 3584}</td></tr>
</tbody></table>

<a name="zh-cn_topic_0000001247793230_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2(const AscendIndexFlatL2&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexFlatL2&amp;</code>: A constant <code>AscendIndexFlatL2</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="zh-cn_topic_0000001294312453_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexFlatL2()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The destructor of <code>AscendIndexFlatL2</code>. It destroys the <code>AscendIndexFlatL2</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456375400"></a>

<a name="zh-cn_topic_0000001248112146_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies an existing <code>index</code> to Ascend based on <code>AscendIndexFlat</code>, clears the current base library of <code>AscendIndexFlatL2</code>, and keeps the existing Device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is <code>{64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3584}</code>. The value range of the total number of base library vectors is <code>0 &lt;= n &lt; 1e9</code>. The <code>metric_type</code> parameter must be <code>faiss::MetricType::METRIC_L2</code>.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000001456535052"></a>

<a name="zh-cn_topic_0000001247793178_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexFlatL2</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user must free the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001456695116"></a>

<a name="zh-cn_topic_0000001294432513_table7235918388"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlatL2&amp; operator=(const AscendIndexFlatL2&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexFlatL2&amp;</code>: A constant <code>AscendIndexFlatL2</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendIndexSQ`<a name="ZH-CN_TOPIC_0000001506614969"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456695120"></a>

`AscendIndexSQ` performs Scalar Quantization on the input vectors.

The vectors stored in the base library and the query vectors of each API must be normalized float values.

It supports multithreaded concurrent calls. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexSQ`<a name="ZH-CN_TOPIC_0000001506614933"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSQ(const faiss::IndexScalarQuantizer* index, AscendIndexSQConfig config = AscendIndexSQConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexSQ</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexScalarQuantizer* index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexSQConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is <code>{64, 128, 256, 384, 512, 768}</code>. The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. The <code>sq.qtype</code> parameter supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</td></tr>
</tbody></table>

<a name="table207325212487"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSQ(const faiss::IndexIDMap* index, AscendIndexSQConfig config = AscendIndexSQConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexSQ</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIDMap* index</code>: CPU-side <code>Index</code> resource.<br><code>AscendIndexSQConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the dimension parameter <code>d</code> of the member index is <code>{64, 128, 256, 384, 512, 768}</code>. The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. The <code>sq.qtype</code> parameter supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</td></tr>
</tbody></table>

<a name="table1132217014918"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexSQConfig config = AscendIndexSQConfig());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexSQ</code>. It creates an <code>AscendIndex</code> with dimension <code>dims</code>. The dimension of a vector set managed by one <code>Index</code> is unique. It then sets Device-side resources according to the values configured in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: The dimension of a set of feature vectors managed by <code>AscendIndexSQ</code>.<br><code>faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit</code>: Currently, only <code>ScalarQuantizer::QuantizerType::QT_8bit</code> is supported.<br><code>faiss::MetricType metric</code>: The distance metric type used by <code>AscendIndex</code> when it performs feature vector similarity retrieval.<br><code>AscendIndexSQConfig config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {64, 128, 256, 384, 512, 768}. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</td></tr>
</tbody></table>

<a name="table16655810104919"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSQ(const AscendIndexSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexSQ&amp;</code>: An <code>AscendIndexSQ</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table17704194534915"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexSQ();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The destructor of <code>AscendIndexSQ</code>. It destroys the <code>AscendIndexSQ</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `copyFrom`<a name="ZH-CN_TOPIC_0000001506615037"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexScalarQuantizer* index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies an existing <code>index</code> to Ascend based on <code>AscendIndexSQ</code>, clears the current base library of <code>AscendIndexSQ</code>, and keeps the existing Device-side resource configuration of <code>AscendIndexSQ</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexScalarQuantizer* index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is <code>{64, 128, 256, 384, 512, 768}</code>. The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. The <code>sq.qtype</code> parameter supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</td></tr>
</tbody></table>

<a name="table853716365015"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIDMap* index);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies an existing <code>index</code> to Ascend based on <code>AscendIndexSQ</code>, clears the current base library of <code>AscendIndexSQ</code>, and keeps the existing Device-side resource configuration of <code>AscendIndexSQ</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The value range of the dimension parameter <code>d</code> of the member index is <code>{64, 128, 256, 384, 512, 768}</code>. The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. The <code>sq.qtype</code> parameter supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</td></tr>
</tbody></table>

### `copyTo`<a name="ZH-CN_TOPIC_0000001456695084"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexScalarQuantizer* index) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexSQ</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexScalarQuantizer* index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user must free the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

<a name="table817201512500"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIDMap* index) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexSQ</code> to the CPU side.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The user must free the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

### `getBase`<a name="ZH-CN_TOPIC_0000001456694928"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getBase(int deviceId, char* xb) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vectors managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>char* xb</code>: The base library feature vectors stored by <code>AscendIndexSQ</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID. <code>xb</code> must be a non-null pointer, and its length must be <code>dims * BaseSize * sizeof(uint8_t)</code> bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>BaseSize</code> is the return value of <code>getBaseSize</code>.</td></tr>
</tbody></table>

### `getBaseSize`<a name="ZH-CN_TOPIC_0000001456854788"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t getBaseSize(int deviceId) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the number of feature vectors managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of feature vectors on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

### `getIdxMap`<a name="ZH-CN_TOPIC_0000001456375152"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt;&amp; idxMap) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vector IDs managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;idx_t&gt; &amp;idxMap</code>: The base library feature vector IDs stored by <code>AscendIndexSQ</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001456375300"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSQ&amp; operator=(const AscendIndexSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexSQ&amp;</code>: An <code>AscendIndexSQ</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `search_with_filter`<a name="ZH-CN_TOPIC_0000001810589742"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The feature vector query API of <code>AscendIndexSQ</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It also provides CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array of length <code>n * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four values of each filter, that is, 128 bits, represent the corresponding CID. The last two values represent the left-closed timestamp interval, that is, [<code>x</code>, <code>y</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of query feature vectors.<br><code>const float *x</code>: Feature vector data.<br><code>idx_t k</code>: The number of most similar results to return.<br><code>const void *filters</code>: Filter conditions.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>filters</code> must be a non-null pointer to a <code>uint32_t</code> array of length <code>n * 6</code>. Otherwise, out-of-bounds read errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### `search_with_masks`<a name="ZH-CN_TOPIC_0000001456694932"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The feature vector query API of <code>AscendIndexSQ</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. <code>mask</code> is a bit string of <code>0</code>s and <code>1</code>s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. <code>1</code> means participate, and <code>0</code> means do not participate.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of query feature vectors.<br><code>const float *x</code>: Feature vector data.<br><code>idx_t k</code>: The number of most similar results to return.<br><code>const void *mask</code>: Feature library mask.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>mask</code> must be a non-null pointer, and its length must be <code>n * ceil(ntotal / 8)</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>ntotal</code> is the number of base library features. <code>mask</code> is set according to the order of the base library. If you call <code>remove_ids</code> to delete feature vectors before calling this API, the order of the base library features changes. First call <code>getIdxMap</code> to obtain the IDs of the base library features, and then set <code>mask</code>. To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect.</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000001506414905"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Trains the quantizer on <code>AscendIndexSQ</code>. This API inherits the interface from <code>AscendFaiss</code> and provides the concrete implementation. **Note that you must train the <code>Index</code> before you call <code>add</code>.**</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of feature vectors in the training set.<br><code>const float *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. Training collects the data distribution. A small training set may affect query accuracy.</td></tr>
</tbody></table>

## `AscendIndexSQConfig`<a name="ZH-CN_TOPIC_0000001456375392"></a>

`AscendIndexSQ` requires the corresponding `AscendIndexSQConfig` to initialize its resources.

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexSQConfig()</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The default constructor of <code>AscendIndexSQConfig</code>. The default <code>deviceList</code> is <code>0</code>, which means the first Ascend AI Processor of the NPU is selected as the heterogeneous computing platform for <code>AscendFaiss</code> retrieval. The default resource pool size is used.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table108621239568"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexSQConfig</code>. It creates an <code>AscendIndexSQConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>SQ_DEFAULT_MEM</code> defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>10,000,000</code> and the batch size is greater than or equal to <code>16</code>, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Configures the <code>blockSize</code> on the Device side. It constrains the amount of data processed in a single <code>tik</code> operator execution and the size of vectors stored in each shard of the base library. The default value is <code>16384 * 16 = 262144</code>. This value affects the maximum number of <code>Index</code> objects that can be created and retrieval performance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>. The valid values of <code>blockSize</code> are <code>{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}</code>.</td></tr>
</tbody></table>

<a name="table1735412445711"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexSQConfig</code>. It creates an <code>AscendIndexSQConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.<br><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>SQ_DEFAULT_MEM</code> defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>10,000,000</code> and the batch size is greater than or equal to <code>16</code>, you are advised to set it to <code>1024 MB</code>.<br><code>uint32_t blockSize</code>: Configures the <code>blockSize</code> on the Device side. It constrains the amount of data processed in a single <code>tik</code> operator execution and the size of vectors stored in each shard of the base library. The default value is <code>16384 * 16 = 262144</code>. This value affects the maximum number of <code>Index</code> objects that can be created and retrieval performance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>. The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>. The valid values of <code>blockSize</code> are <code>{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}</code>.</td></tr>
</tbody></table>

## `IndexIL`<a name="ZH-CN_TOPIC_0000001506414825"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456535188"></a>

`IndexIL` is a feature management abstract class based on a contiguous memory allocation mechanism. It serves retrieval algorithms that use indices as labels. To use it, you must inherit from it and implement all interfaces.

The vectors stored in the base library and the query vectors of each API must be normalized FP16 floating-point values. (`IL` stands for "Indices as Labels".)

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, the user must lock before use. Otherwise, the retrieval APIs may raise exceptions. It also does not support sharing a Device across different threads.

### `AddFeatures`<a name="ZH-CN_TOPIC_0000001506414693"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Inserts <code>n</code> feature vectors with specified indices into the feature library. If a feature vector already exists at an index, this insertion is equivalent to an update.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: The number of feature vectors to insert.<br><code>const float16_t *features</code>: Feature vectors, with a length of <code>n * vector dimension dim</code>.<br><code>const idx_t *indices</code>: The index values corresponding to the feature vectors, with a length of <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The return status of the call. For details, see the reference for API return values.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The input parameters are constrained by the implementation class. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### `IndexIL`<a name="ZH-CN_TOPIC_0000001456695020"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>IndexIL();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>IndexIL</code>. It creates a feature management object.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~IndexIL`<a name="ZH-CN_TOPIC_0000001506334781"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~IndexIL();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The destructor of <code>IndexIL</code>. It destroys the feature management object.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Finalize`<a name="ZH-CN_TOPIC_0000001456375356"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR Finalize() = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Releases the feature library management resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The return status of the call. For details, see the reference for API return values.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `GetFeatures`<a name="ZH-CN_TOPIC_0000001506495833"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the feature vectors for <code>n</code> specified index values.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: The number of feature vectors to obtain.<br><code>const idx_t *indices</code>: The index values to query, with a length of <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float16_t *features</code>: The feature vectors corresponding to the queried indices, with a length of <code>n * vector dimension dim</code>. The user must allocate memory before the call and ensure that the memory size is correct.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The return status of the call. For details, see the reference for API return values.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The input parameters are constrained by the implementation class. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetNTotal`<a name="ZH-CN_TOPIC_0000001456535092"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual int GetNTotal() const = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the maximum occupied space of the current feature library vectors.<br>Feature vectors are inserted starting from index <code>0</code>. If the inserted feature vector indices are continuous, <code>ntotal</code> equals the number of feature vectors. Otherwise, <code>ntotal</code> equals the maximum inserted index value plus <code>1</code>. For performance reasons, the operator batches memory operations and, by default, treats the space at and before the maximum index position as valid base library vectors and includes it in the calculation. The user must use this API to obtain the total number of base library entries recorded inside the <code>Index</code>, and then allocate the corresponding memory space to pass parameters to the corresponding functional APIs. For details, see the specific API.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>int ntotal</code>: See the description.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Init`<a name="ZH-CN_TOPIC_0000001506334657"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initializes feature library parameters and allocates base library memory resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dim</code>: Feature vector dimension.<br><code>AscendMetricType metricType</code>: Feature distance type, including inner product, Euclidean distance, and cosine similarity.<br><code>int capacity</code>: Maximum base library capacity. The allocated memory size is <code>capacity * dim * sizeof(float)</code> bytes.<br><code>int resourceSize</code>: Preallocates Device-side cache resources. When a retrieval API is called, it can use these resources directly instead of calling <code>aclrtmalloc</code> to allocate memory, which improves performance. The default value is <code>-1</code>, which means the cache resource is allocated with the default size of <code>128 MB</code>. You can configure the actual size more precisely based on the retrieval workload and Device-side resource usage.<br>For example, if the query batch size is <code>64</code>, the base library contains 1,000,000 vectors, and one FP32 value occupies 4 bytes, set <code>resourceSize</code> to <code>64 * 1000000 * 4 = 256,000,000</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The implementation class constrains the input parameters.</td></tr>
</tbody></table>

### RemoveFeatures API<a name="ZH-CN_TOPIC_0000001456534932"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR RemoveFeatures(int n, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Deletes the feature vectors with the specified indices from the vector library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to delete.<br><code>const idx_t *indices</code>: Indices of the feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The implementation class constrains the input parameters. <code>indices</code> must be a non-null pointer, and its length must be <code>n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### SetNTotal API<a name="ZH-CN_TOPIC_0000001456375256"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR SetNTotal(int n) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Provides an interface for adjusting the <code>ntotal</code> count externally.<br>After base library vectors are added, the <code>Index</code> internally updates the <code>ntotal</code> value according to the largest inserted index, but it does not record which regions in the range [0, <code>ntotal</code> ] are invalid. Therefore, the <code>RemoveFeatures</code> operation does not change the <code>ntotal</code> value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set <code>ntotal</code> manually. This reduces the operator workload within a controllable range and improves interface performance.<br>For example, if 100 vectors are inserted and the base library indices range from 0 to 99, <code>ntotal = 100</code>. If you delete the base library entries with indices from 80 to 90, the <code>ntotal</code> value inside <code>Index</code> remains unchanged and can only be set to a value in [ <code>ntotal</code>, <code>capacity</code> ]. If you then delete the base library entries with indices from 90 to 99, you can manually set <code>ntotal</code> to a value in [80, <code>capacity</code> ]. When you set it to <code>80</code>, the amount of base library data involved in comparison decreases by 20 vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Maximum base library index managed by the service side, plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The implementation class constrains the input parameters.</td></tr>
</tbody></table>

## IndexILFlat<a name="ZH-CN_TOPIC_0000001506614925"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506414785"></a>

`IndexILFlat` inherits from `IndexIL` and is a pure Device-side retrieval solution. It uses resources such as the Ascend AI Processor and AI Core to enable each API. The program must be compiled on the Host into a binary, and then the binary and related runtime dependencies are deployed to the Device for execution. `IndexILFlat` uses the [Init](#init) interface to initialize the specified resources. After initialization, it allocates a contiguous block of memory to store the base library. After use, call the [Finalize](#finalize) interface to release the resources.

`IndexILFlat` currently receives only functional and performance maintenance on Atlas Inference Series products. The base library and query vectors must be normalized by the user, and the interfaces currently support only the inner product distance. For details, see [IndexILFlat](#indexilflat). Successful execution of this algorithm depends on the OM file of the TIK operator. In a pure-Device scenario, ensure that the deployed OM file is generated from the Index SDK deliverable and has not been tampered with.

Multithreaded concurrent calls are supported. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

### AddFeatures API<a name="ZH-CN_TOPIC_0000001456854852"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Inserts <code>n</code> feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, the API updates it.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to insert.<br><code>const float16_t *features</code>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>const idx_t *indices</code>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>capacity</code> ). <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### ComputeDistance API<a name="ZH-CN_TOPIC_0000001456535116"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the distances between <code>n</code> feature vectors and all feature vectors in the base library. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: External memory that stores the distances between query vectors and base library vectors. The total length should be <code>n * nTotalPad</code> (<code>ntotalPad</code> is <code>(*ntotal + 15) / 16 * 16</code>, that is, <code>ntotal</code> rounded up to a multiple of 16).</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>distances</code>: The required buffer length is <code>n * ntotalPad</code> (<code>ntotalPad</code> is <code>(*ntotal + 15) / 16 * 16</code>, that is, the result of rounding <code>ntotal</code> up to a multiple of 16. The valid comparison distances for each query are stored in the first <code>ntotal</code> positions, and the padded data has no practical meaning). If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>queries</code> and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### ComputeDistanceByIdx API<a name="ZH-CN_TOPIC_0000001456694920"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Similar to <code>ComputeDistance</code>, except that <code>ComputeDistance</code> calculates the distances between query vectors and all base library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distances between query vectors and the base library vectors at the given indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The valid length is <code>n * dim</code>, and <code>dim</code> must match the dimension specified during initialization.<br><code>const int *num</code>: Number of base library feature vectors to compare for each query. The length is <code>n</code>.<br><code>const idx_t *indices</code>: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum <code>num</code> value. The length of <code>indices</code> is <code>n * max(num)</code>.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum <code>num</code> value. The total length is <code>n * max(num)</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>num</code>: User-specified length <code>n</code>, and each <code>num</code> value must be in [0, <code>ntotal</code>]. <code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ). For example, if <code>n = 3</code> and <code>num[3] = {1, 3, 5}</code>, the three queries compare with 1, 3, and 5 base library vectors respectively. Since <code>max(num) = 5</code>, the storage space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example, <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### ComputeDistanceByThreshold API<a name="ZH-CN_TOPIC_0000001506615117"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds threshold filtering on top of <code>ComputeDistance</code> and returns only the distances that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), <code>distances</code> contains the mapped results after threshold filtering.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>float16_t *queries</code>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>float threshold</code>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int *num</code>: Number of base library vectors that meet the threshold condition for each query, with length <code>n</code>.<br><code>idx_t *indices</code>: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.<br><code>float *distances</code>: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>topk</code>: The value must be in [0, 1024]. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### Finalize API<a name="ZH-CN_TOPIC_0000001506414845"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Finalize() override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Releases feature library management resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### GetFeatures API<a name="ZH-CN_TOPIC_0000001456854992"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the feature vectors with the specified indices for <code>n</code> entries.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of base library vectors to get.<br><code>const idx_t *indices</code>: Indices corresponding to the <code>n</code> base library vectors to get.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float16_t *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ), and you can get <code>ntotal</code> by calling <code>GetNTotal</code>. <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### GetNTotal API<a name="ZH-CN_TOPIC_0000001456375336"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int GetNTotal() const override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the theoretical maximum number of feature vectors in the current feature library. If the feature vector indices are inserted consecutively, <code>ntotal</code> is equal to the number of feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int ntotal</code>: The theoretical maximum number of feature vectors, that is, the maximum base library index plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>int ntotal</code>: See the description.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### IndexILFlat API<a name="ZH-CN_TOPIC_0000001456694872"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>IndexILFlat();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>IndexILFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table194381755582"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>IndexILFlat(const IndexILFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor of <code>IndexILFlat</code> as deleted. Therefore, <code>IndexILFlat</code> is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const IndexILFlat&amp;</code>: <code>IndexILFlat</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~IndexILFlat` API<a name="ZH-CN_TOPIC_0000001456375172"></a>

<a name="table11904175418"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~IndexILFlat();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>IndexILFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### Init API<a name="ZH-CN_TOPIC_0000001456375212"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize = -1) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initializes feature library parameters and allocates base library memory resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dim</code>: Feature vector dimension.<br><code>AscendMetricType metricType</code>: Feature distance type, including inner product, Euclidean distance, and cosine similarity.<br><code>int capacity</code>: Maximum base library capacity. The API allocates <code>capacity * dim * sizeof(fp16)</code> bytes of memory based on the <code>capacity</code> value.<br><code>int64_t resourceSize</code>: Preallocates Device-side cache resources. When a retrieval API is called, it can use these resources directly instead of calling the <code>aclrtmalloc</code> interface to allocate memory, which improves performance.<br>The default value is <code>-1</code>, which means that the cache resource is allocated with the default size of <code>128 MB</code>. You can configure the actual size more precisely based on the retrieval workload and Device-side resource usage.<br>For example, if the query batch size is <code>64</code>, the base library contains 1,000,000 vectors, and one FP32 value occupies 4 bytes, set <code>resourceSize</code> to <code>64 * 1000000 * 4 = 256,000,000</code> bytes. Note that the maximum cache resource supported by the interface is <code>4 GB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {32, 64, 128, 256, 384, 512, 1024}. <code>metricType</code>: <code>IndexILFlat</code> currently implements only the inner product distance, so it supports only <code>AscendMetricType::ASCEND_METRIC_INNER_PRODUCT</code>. <code>capacity</code>: The maximum memory that the API can allocate for the base library is <code>12,288,000,000</code> bytes, and the allowed range of <code>capacity</code> is (0, 12000000]. For example, for a base library vector set with 512 dimensions and the FP16 type, the maximum supported <code>capacity</code> is 12 million (<code>12288000000 / (512 * sizeof(fp16))</code>). For a base library vector set with 256 dimensions and the FP16 type, <code>capacity</code> can still be set to at most 12 million, even though the memory limit supports a larger value. <code>resourceSize</code> can be set to <code>-1</code> or any value in [134217728, 4294967296], in bytes, which is equivalent to <code>[128 MB, 4096 MB]</code>. This parameter is determined jointly by the base library size and the search batch size. When the base library contains at least 10 million vectors and the batch size is at least 16, you are advised to set it to <code>1024 MB</code>.</td></tr>
</tbody></table>

### `operator =` API<a name="ZH-CN_TOPIC_0000001897140809"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>IndexILFlat&amp; operator=(const IndexILFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares this <code>Index</code> assignment operator as deleted. Therefore, the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const IndexILFlat&amp;</code>: <code>IndexILFlat</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### RemoveFeatures API<a name="ZH-CN_TOPIC_0000001506414837"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR RemoveFeatures(int n, const idx_t *indices) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Deletes the feature vectors with the specified indices from the vector library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to delete.<br><code>const idx_t *indices</code>: Indices of the feature vectors. The length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ), and you can get <code>ntotal</code> by calling <code>GetNTotal</code>. <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>indices</code> must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### Search API<a name="ZH-CN_TOPIC_0000001456854856"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the indices and corresponding distances of the <code>topk</code> base library vectors that are closest to the query vectors. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>int topk</code>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: External memory. It stores the cosine distances corresponding to the <code>topk * n</code> base library feature vectors that are most similar to the query. The length is <code>n * topk</code>.<br><code>idx_t *indices</code>: External memory. It returns the indices corresponding to the <code>topk</code> base library vectors that are most similar to the query. The length is <code>n * topk</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>topk</code>: The value must be in (0, 1024]. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### SearchByThreshold API<a name="ZH-CN_TOPIC_0000001456694892"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The length is <code>n * dim</code>.<br><code>float threshold</code>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.<br><code>int topk</code>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int *num</code>: Number of base library vectors that meet the threshold condition for each query. The length is <code>n</code>.<br><code>idx_t* indices</code>: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.<br><code>float *distances</code>: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>topk</code>: The value must be in (0, 1024]. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### SetNTotal API<a name="ZH-CN_TOPIC_0000001456854892"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetNTotal(int n) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Provides an interface for adjusting the <code>ntotal</code> count externally.<br>After base library vectors are added, the <code>Index</code> internally updates the <code>ntotal</code> value according to the largest inserted index, but it does not record which regions in the range [0, <code>ntotal</code> ] are invalid. Therefore, the <code>RemoveFeatures</code> operation does not change the <code>ntotal</code> value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set <code>ntotal</code> manually. This reduces the operator workload within a controllable range and improves interface performance.<br>For example, if 100 vectors are inserted and the base library indices range from 0 to 99, <code>ntotal = 100</code>. If you delete the base library entries with indices from 80 to 90, the <code>ntotal</code> value inside <code>Index</code> remains unchanged and can only be set to a value in [ <code>ntotal</code>, <code>capacity</code> ]. If you then delete the base library entries with indices from 90 to 99, you can manually set <code>ntotal</code> to a value in [80, <code>capacity</code> ]. When you set it to <code>80</code>, the amount of base library data involved in comparison decreases by 20 vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Maximum base library index managed by the service side, plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in [0, <code>capacity</code> ].</td></tr>
</tbody></table>

## AscendIndexILFlat<a name="ZH-CN_TOPIC_0000002514896041"></a>

### Overview<a name="ZH-CN_TOPIC_0000002482656058"></a>

`AscendIndexILFlat` is the standard-mode scenario of `ILFlat`. You need to use `Init` to initialize the corresponding resources. After initialization, it allocates a contiguous block of memory to store the base library. After use, call the `Finalize` interface to release the resources.

`AscendIndexILFlat` supports only Atlas Inference Series products and only the inner product distance type in the standard deployment mode. `AscendIndexILFlat` depends on the Flat and AICPU operators. For details, see [Flat](../user_guide.md#generating-operators) and [AICPU](../user_guide.md#generating-operators).

Multithreaded concurrent calls are supported. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

### AddFeatures API<a name="ZH-CN_TOPIC_0000002514776041"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const float *features);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds <code>n</code> feature vectors to the feature library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to insert.<br><code>const float *features</code>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>features</code> must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table392463914228"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddFeatures(int n, const float16_t *features);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds <code>n</code> feature vectors to the feature library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to insert.<br><code>const float16_t *features</code>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>features</code> must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### AscendIndexILFlat API<a name="ZH-CN_TOPIC_0000002516511133"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexILFlat();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexILFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table161511529133912"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexILFlat(const AscendIndexILFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor of <code>AscendIndexILFlat</code> as deleted. Therefore, <code>AscendIndexILFlat</code> is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexILFlat&amp;</code>: <code>AscendIndexILFlat</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table62621513124018"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexILFlat();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexILFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### ComputeDistance API<a name="ZH-CN_TOPIC_0000002482736032"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the distances between <code>n</code> feature vectors and all feature vectors in the base library. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: External memory. It stores the distances between query vectors and base library vectors. The total length should be <code>n * nTotalPad</code> (<code>ntotalPad</code> is <code>(*ntotal + 15) / 16 * 16</code>, that is, <code>ntotal</code> rounded up to a multiple of 16).</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The recommended value should be in (0, <code>capacity</code> ]. <code>distances</code>: The required buffer length is <code>n * ntotalPad</code> (<code>ntotalPad</code> is <code>(*ntotal + 15) / 16 * 16</code>, that is, the result of rounding <code>ntotal</code> up to a multiple of 16. The valid comparison distances for each query are stored in the first <code>ntotal</code> positions, and the padded data has no practical meaning). If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>queries</code> and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table17574555124816"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistance(int n, const float *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the distances between <code>n</code> feature vectors and all feature vectors in the base library. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float *queries</code>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: External memory. It stores the distances between query vectors and base library vectors. The total length should be <code>n * nTotalPad</code> (<code>ntotalPad</code> is <code>(*ntotal + 15) / 16 * 16</code>, that is, <code>ntotal</code> rounded up to a multiple of 16).</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The recommended value should be in (0, <code>capacity</code> ]. <code>distances</code>: The required buffer length is <code>n * ntotalPad</code> (<code>ntotalPad</code> is <code>(*ntotal + 15) / 16 * 16</code>, that is, the result of rounding <code>ntotal</code> up to a multiple of 16. The valid comparison distances for each query are stored in the first <code>ntotal</code> positions, and the padded data has no practical meaning). If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>queries</code> and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### ComputeDistanceByIdx API<a name="ZH-CN_TOPIC_0000002514896043"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const float *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle"><code>ComputeDistance</code> calculates the distances between query vectors and all base library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distances between query vectors and the base library vectors at the specified indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float *queries</code>: Feature vectors to query. The valid length is <code>n * dim</code>, and <code>dim</code> must match the dimension specified during initialization.<br><code>const int *num</code>: Number of base library feature vectors to compare for each query. The length is <code>n</code>.<br><code>const idx_t *indices</code>: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum <code>num</code> value. The length of <code>indices</code> is <code>n * max(num)</code>. If the input is on the host, <code>indices</code> is a host pointer. If the input is on the device, <code>indices</code> is a device pointer.<br><code>MEMORY_TYPE memoryType</code>: Policy for where the input and output are stored. The default is <code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>. The available policies are as follows:<br><code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>: input on the host and output on the host. <code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE</code>: input on the device and output on the device. <code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST</code>: input on the device and output on the host. <code>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE</code>: input on the host and output on the device.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum <code>num</code> value. The total length is <code>n * max(num)</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>num</code>: User-specified, with length <code>n</code>, and each query&#x27;s <code>num</code> value must be in [0, <code>ntotal</code>]. <code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ). For example, if <code>n = 3</code> and <code>num[3] = {1, 3, 5}</code>, the three queries compare with 1, 3, and 5 base library vectors respectively. Since <code>max(num) = 5</code>, the storage space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example, <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. When selecting a <code>memoryType</code> storage policy, <code>queries</code> and <code>distances</code> must be pointers to the corresponding location, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table93703718308"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle"><code>ComputeDistance</code> calculates the distances between query vectors and all base library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distances between query vectors and the base library vectors at the specified indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The valid length is <code>n * dim</code>, and <code>dim</code> must match the dimension specified during initialization.<br><code>const int *num</code>: Number of base library feature vectors to compare for each query. The length is <code>n</code>.<br><code>const idx_t *indices</code>: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum <code>num</code> value. The length of <code>indices</code> is <code>n * max(num)</code>. If the input is on the host, <code>indices</code> is a host pointer. If the input is on the device, <code>indices</code> is a device pointer.<br><code>MEMORY_TYPE memoryType</code>: Policy for where the input and output are stored. The default is <code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>. The available policies are as follows:<br><code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>: input on the host and output on the host. <code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE</code>: input on the device and output on the device. <code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST</code>: input on the device and output on the host. <code>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE</code>: input on the host and output on the device.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum <code>num</code> value. The total length is <code>n * max(num)</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>num</code>: User-specified, with length <code>n</code>, and each query&#x27;s <code>num</code> value must be in [0, <code>ntotal</code>]. <code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ). For example, if <code>n = 3</code> and <code>num[3] = {1, 3, 5}</code>, the three queries compare with 1, 3, and 5 base library vectors respectively. Since <code>max(num) = 5</code>, the storage space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example, <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. When selecting a <code>memoryType</code> storage policy, <code>queries</code> and <code>distances</code> must be pointers to the corresponding location, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### Finalize API<a name="ZH-CN_TOPIC_0000002482656060"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void Finalize();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Releases feature library management resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### GetFeatures API<a name="ZH-CN_TOPIC_0000002484074790"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, float *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the host.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of base library vectors to get.<br><code>const idx_t *indices</code>: Indices corresponding to the feature vectors, with length <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ), and you can get <code>ntotal</code> by calling <code>GetNTotal</code>. <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table018415716495"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the host.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of base library vectors to get.<br><code>const idx_t *indices</code>: Indices corresponding to the feature vectors, with length <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float16_t *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ), and you can get <code>ntotal</code> by calling <code>GetNTotal</code>. <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### GetFeaturesOnDevice API<a name="ZH-CN_TOPIC_0000002516516843"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeaturesOnDevice (int n, float16_t *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the Device.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of base library vectors to get.<br><code>const idx_t *indices</code>: Indices corresponding to the feature vectors, with length <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float16_t *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension. Device-side pointer.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ), and you can get <code>ntotal</code> by calling <code>GetNTotal</code>. <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table15312115612410"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeaturesOnDevice (int n, float *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the Device.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of base library vectors to get.<br><code>const idx_t *indices</code>: Indices corresponding to the feature vectors, with length <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension. Device-side pointer.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ), and you can get <code>ntotal</code> by calling <code>GetNTotal</code>. <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### GetNTotal API<a name="ZH-CN_TOPIC_0000002514776043"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int GetNTotal() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the theoretical maximum number of feature vectors in the current feature library. If the feature vector indices are inserted consecutively, <code>ntotal</code> is equal to the number of feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int ntotal</code>: The theoretical maximum number of feature vectors, that is, the maximum base library index plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>int</code>: The theoretical maximum number of feature vectors, that is, the maximum base library index plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### Init API<a name="ZH-CN_TOPIC_0000002482736034"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initialization function of <code>AscendIndexILFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dim</code>: Dimension of the feature vectors managed by <code>AscendIndexILFlat</code>.<br><code>int capacity</code>: Maximum base library capacity. The API allocates <code>capacity * dim * sizeof(fp16)</code> bytes of memory based on the <code>capacity</code> value.<br><code>faiss::MetricType metricType</code>: Feature distance type, including inner product, Euclidean distance, and cosine similarity.<br><code>const std::vector&lt;int&gt; &amp;deviceList</code>: Device-side resource configuration.<br><code>int64_t resourceSize</code>: Device-side preset memory pool size, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>-1</code>, which means <code>128 MB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {32, 64, 128, 256, 384, 512}. <code>metricType</code>: <code>AscendIndexILFlat</code> currently implements only the inner product distance, so it supports only <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>. <code>capacity</code>: The maximum memory that the API can allocate for the base library is <code>12,288,000,000</code> bytes, and the allowed range of <code>capacity</code> is [0, 12000000]. For example, for a base library vector set with 512 dimensions and the FP16 type, the maximum supported <code>capacity</code> is 12 million (<code>12288000000 / (512 * sizeof(fp16))</code>). For a base library vector set with 256 dimensions and the FP16 type, <code>capacity</code> can still be set to at most 12 million, even though the memory limit supports a larger value. Only single-card configuration is supported. Multi-card configuration is not supported yet, and <code>deviceList.size() == 1</code> must hold. <code>resourceSize</code> can be set to <code>-1</code> or any value in [134217728, 4294967296], which is equivalent to <code>[128 MB, 4096 MB]</code>. This parameter is determined jointly by the base library size and the search batch size. When the base library contains at least 10 million vectors and the batch size is at least 16, you are advised to set it to <code>1024 MB</code>.</td></tr>
</tbody></table>

### `operator =` API<a name="ZH-CN_TOPIC_0000002482794858"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexILFlat&amp; operator=(const AscendIndexILFlat &amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares this <code>Index</code> assignment operator as deleted. Therefore, the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexILFlat &amp;</code>: <code>AscendIndexILFlat</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### RemoveFeatures API<a name="ZH-CN_TOPIC_0000002482917750"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR RemoveFeatures(int n, const idx_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Deletes the feature vectors with the specified indices from the vector library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to delete.<br><code>const idx_t *indices</code>: Indices of the feature vectors. The length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ), and you can get <code>ntotal</code> by calling <code>GetNTotal</code>. <code>n</code>: The value must be in [0, <code>capacity</code> ]. <code>indices</code> must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### Search API<a name="ZH-CN_TOPIC_0000002514896045"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the indices and corresponding distances of the <code>topk</code> base library vectors that are closest to the query vectors. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>int topk</code>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: External memory. It stores the cosine distances corresponding to the <code>topk * n</code> base library feature vectors that are most similar to the query. The length is <code>n * topk</code>.<br><code>idx_t *indices</code>: External memory. It returns the indices corresponding to the <code>topk</code> base library vectors that are most similar to the query. The length is <code>n * topk</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>topk</code>: The value must be in (0, 1024]. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table838713119461"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Returns the indices and corresponding distances of the <code>topk</code> base library vectors that are closest to the query vectors. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float *queries</code>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>int topk</code>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: External memory. It stores the cosine distances corresponding to the <code>topk * n</code> base library feature vectors that are most similar to the query. The length is <code>n * topk</code>.<br><code>idx_t *indices</code>: External memory. It returns the indices corresponding to the <code>topk</code> base library vectors that are most similar to the query. The length is <code>n * topk</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>topk</code>: The value must be in (0, 1024]. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### SearchByThreshold API<a name="ZH-CN_TOPIC_0000002482656062"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float *queries</code>: Feature vectors to query. The length is <code>n * dim</code>.<br><code>float threshold</code>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.<br><code>int topk</code>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int *num</code>: Number of base library vectors that meet the threshold condition for each query. The length is <code>n</code>.<br><code>idx_t *indices</code>: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.<br><code>float *distances</code>: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>topk</code>: The value must be in (0, 1024]. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table910711421721"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const float16_t *queries</code>: Feature vectors to query. The length is <code>n * dim</code>.<br><code>float threshold</code>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.<br><code>int topk</code>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.<br><code>unsigned int tableLen</code>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.<br><code>const float *table</code>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int *num</code>: Number of base library vectors that meet the threshold condition for each query. The length is <code>n</code>.<br><code>idx_t* indices</code>: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.<br><code>float *distances</code>: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>topk</code>: The value must be in (0, 1024]. If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values.<br>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in [0, 1]. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in [0, <code>tableLen</code>]. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

### SetNTotal API<a name="ZH-CN_TOPIC_0000002514776045"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetNTotal(int n);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Provides an interface for adjusting the <code>ntotal</code> count externally.<br>After base library vectors are added, the <code>Index</code> internally updates the <code>ntotal</code> value according to the largest inserted index, but it does not record which regions in the range [0, <code>ntotal</code> ] are invalid. Therefore, the <code>RemoveFeatures</code> operation does not change the <code>ntotal</code> value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set <code>ntotal</code> manually. This reduces the operator workload within a controllable range and improves interface performance.<br>For example, if 100 vectors are inserted and the base library indices range from 0 to 99, <code>ntotal = 100</code>. If you delete the base library entries with indices from 80 to 90, the <code>ntotal</code> value inside <code>Index</code> remains unchanged and can only be set to a value in [ <code>ntotal</code>, <code>capacity</code> ]. If you then delete the base library entries with indices from 90 to 99, you can manually set <code>ntotal</code> to a value in [80, <code>capacity</code> ]. When you set it to <code>80</code>, the amount of base library data involved in comparison decreases by 20 vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Maximum base library index managed by the service side, plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: The value must be in [0, <code>capacity</code> ].</td></tr>
</tbody></table>

### UpdateFeatures API<a name="ZH-CN_TOPIC_0000002516314733"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR UpdateFeatures (int n, const float16_t *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Updates <code>n</code> feature vectors with the specified indices in the feature library. If a feature vector does not exist at an index, the API adds it. If a feature vector already exists at an index, the API updates it.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to insert.<br><code>const float16_t *features</code>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>const idx_t *indices</code>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ). <code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

<a name="table19567183517113"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR UpdateFeatures(int n, const float *features, const idx_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Updates <code>n</code> feature vectors with the specified indices in the feature library. If a feature vector does not exist at an index, the API adds it. If a feature vector already exists at an index, the API updates it.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to insert.<br><code>const float *features</code>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.<br><code>const idx_t *indices</code>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: Each feature index must be in [0, <code>ntotal</code> ). <code>n</code>: The value must be in (0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>
