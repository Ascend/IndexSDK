# `AscendIndexCluster`<a id="en-us_TOPIC_0000001614744825"></a>

## Overview<a name="en-us_TOPIC_0000001564586790"></a>

`AscendIndexCluster` requires [`Init`](#init) to initialize the specified resources. After initialization, it allocates a complete memory space to store the base library. After use, call [`Finalize`](#finalize) to release the resources.

`AscendIndexCluster` supports only the vector inner-product distance type in standard mode on Atlas Inference Series products. It depends on Flat and AICPU operators. For details, see [Flat](../../05_user_guide.md#flat) and [AICPU](../../05_user_guide.md#aicpu).

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `AddFeatures`<a name="en-us_TOPIC_0000001614746533"></a>

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

## `AscendIndexCluster`<a name="en-us_TOPIC_0000001564746410"></a>

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

## `~AscendIndexCluster`<a name="en-us_TOPIC_0000002399598393"></a>

<a name="table179216322487"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexCluster() = default;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexCluster</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `ComputeDistanceByIdx`<a name="en-us_TOPIC_0000002446061685"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num, const uint32_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle"><code>ComputeDistance</code> calculates the distance between the query vectors and all base-library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distance between the query vectors and the base-library vectors at the given indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the interface returns the mapped top-<code>k</code> results.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of query feature vectors.<br><code>const uint16_t *queries</code>: Query feature vectors. The valid length is <code>n * dim</code>, and <code>dim</code> must be the same as the dimension specified during initialization.<br><code>const int *num</code>: Number of base-library feature vectors to compare for each query. The length is <code>n</code>.<br><code>const uint32_t *indices</code>: Indices of the base-library feature vectors to compare. The number of base-library vectors to compare can differ for each query. Valid vector indices must be stored continuously from front to back, and the space usage must be padded according to the maximum <code>num</code>. The length of <code>indices</code> is <code>n * max(num)</code>.<br><code>unsigned int tableLen</code>: Mapping-table length. The default value is <code>0</code>, which means that no mapping is performed. Currently, the supported mapping-table length is <code>10000</code>.<br><code>const float *table</code>: Mapping-table pointer that points to valid mapped values of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>*table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distances between the query vectors and the selected base-library vectors. For each query, valid distances are recorded continuously from front to back, and the space usage is padded according to the maximum <code>num</code>. The total length is <code>n * max(num)</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: Must be in the range (0, <code>capacity</code> ]. <code>num</code>: User-specified. The length is <code>n</code>, and the <code>num</code> value for each query must be in [0, <code>ntotal</code>]. <code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ). Example parameter values: <code>n = 3</code>, <code>num[3] = {1, 3, 5}</code> means that the three queries compare against 1, 3, and 5 base-library vectors respectively. If <code>max(num) = 5</code>, then the space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>. When both <code>tableLen</code> and <code>table</code> meet the requirements, the interface maps the computed <code>distance</code> values.<br>First, normalize <code>distance</code> to a floating-point value <code>f1</code> in [0, 1]. Then multiply <code>f1</code> by <code>tableLen</code> and round it down to obtain an integer index in [0, <code>tableLen</code>]. Next, use the integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>. This completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be abstracted as <code>((CosDistance + 1) / 2) * tableLen</code>.</td></tr>
</tbody></table>

## `ComputeDistanceByThreshold`<a name="en-us_TOPIC_0000001615066169"></a>

> This interface must be used together with [`AddFeatures(int n, const float *features, const uint32_t *indices);`](#addfeatures).

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR ComputeDistanceByThreshold(const std::vector&lt;uint32_t&gt; &amp;queryIdxArr, uint32_t codeStartIdx, uint32_t codeNum, float threshold, bool aboveFilter, std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr, std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Calculates the distances between the queried feature vectors in the base library and the specified base-library feature vectors, then filters by threshold and returns the distances and labels that meet the conditions.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;uint32_t&gt; &amp;queryIdxArr</code>: Indices of the vectors to query in the base library.<br><code>uint32_t codeStartIdx</code>: Starting index of the base library vectors for distance calculation.<br><code>uint32_t codeNum</code>: Number of base-library vectors for distance calculation.<br><code>float threshold</code>: Threshold used for filtering. Distances smaller than the threshold are filtered out.<br><code>bool aboveFilter</code>: Reserved parameter.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr</code>: Two-dimensional array that returns the distances between each query vector and the base-library vectors that meet the threshold condition.<br><code>std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr</code>: Two-dimensional array that returns the indices of the base-library vectors that meet the threshold condition for each query vector.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The lengths of <code>queryIdxArr</code>, <code>resDistArr</code>, and <code>resIdxArr</code> must be the same, that is, <code>queryIdxArr.size() == resDistArr.size()</code>. <code>queryIdxArr.size()</code> must be greater than <code>0</code> and less than or equal to <code>ntotal</code>. <code>codeNum</code> must be greater than <code>0</code> and less than or equal to <code>ntotal</code>. <code>codeStartIdx + codeNum</code> must not exceed <code>ntotal</code> (the base-library size). <code>codeStartIdx</code> must be greater than or equal to <code>0</code> and less than or equal to <code>ntotal</code>.</td></tr>
</tbody></table>

## `Finalize`<a name="en-us_TOPIC_0000001614906601"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle">void Finalize();</td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Releases feature-library management resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `GetFeatures`<a name="en-us_TOPIC_0000002412742482"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatures(int n, uint16_t *features, const int64_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Retrieves <code>n</code> feature vectors at the specified indices.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of base-library vectors to retrieve.<br><code>const int64_t *indices</code>: Indices corresponding to the feature vectors. The length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>uint16_t *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * vector dimension dim</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ), and <code>ntotal</code> can be obtained through the <code>GetNTotal</code> interface. <code>n</code>: Must be in [0, <code>capacity</code> ]. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `GetNTotal`<a name="en-us_TOPIC_0000002412582646"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int GetNTotal() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the theoretical maximum number of feature vectors in the current feature library. If the inserted feature-vector indices are continuous, <code>ntotal</code> is equal to the number of feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int ntotal</code>: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>int</code>: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `Init`<a name="en-us_TOPIC_0000001614866169"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initialization function of <code>AscendIndexCluster</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dim</code>: Dimension of the feature vectors managed by <code>AscendIndexCluster</code>.<br><code>int capacity</code>: Maximum base-library capacity. The interface allocates <code>capacity * dim * sizeof(fp16)</code> bytes of memory based on the value of <code>capacity</code>.<br><code>faiss::MetricType metricType</code>: Feature-distance category, including vector inner product, Euclidean distance, and cosine similarity.<br><code>const std::vector&lt;int&gt; &amp;deviceList</code>: Device-side resource configuration.<br><code>int64_t resourceSize</code>: Size of the preallocated memory pool on the device side, in bytes. This memory stores intermediate results during computation and is used to avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>-1</code>, which means <code>128 MB</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> must be one of <code>{32, 64, 128, 256, 384, 512}</code>. <code>metricType</code>: <code>AscendIndexCluster</code> currently implements only vector inner-product distance, which means that only <code>faiss::MetricType::METRIC_INNER_PRODUCT</code> is supported. The maximum memory that can be allocated for the base library is <code>12,288,000,000</code> bytes, and the value range of <code>capacity</code> is [0, 12000000]. For example, for a base-library vector with 512 dimensions and the FP16 type, the maximum supported <code>capacity</code> is 12 million (<code>12288000000 / (512 * sizeof(fp_16))</code>). For base-library vectors with 256 dimensions and the FP16 type, even though the memory constraint supports a larger <code>capacity</code>, the maximum <code>capacity</code> can still be only 12 million. Only single-card configuration is supported. Multi-card configuration is not supported yet, so <code>deviceList.size()</code> must equal <code>1</code>. <code>resourceSize</code> can be <code>-1</code> or a value in [134217728, 4294967296], which is equivalent to <code>[128 MB, 4096 MB]</code>. This parameter is determined jointly by the base-library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</td></tr>
</tbody></table>

## `operator =`<a name="en-us_TOPIC_0000001897100377"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCluster&amp; operator=(const AscendIndexCluster&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares this <code>Index</code> copy assignment operator as deleted, making the type non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexCluster&amp;</code>: <code>AscendIndexCluster</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `RemoveFeatures`<a name="en-us_TOPIC_0000002446181741"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR RemoveFeatures(int n, const int64_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Removes <code>n</code> feature vectors at the specified indices from the vector library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to remove.<br><code>const int64_t *indices</code>: Indices corresponding to the feature vectors. The length is <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ), and <code>ntotal</code> can be obtained through the <code>GetNTotal</code> interface. <code>n</code>: Must be in [0, <code>capacity</code> ]. <code>indices</code> must be a non-null pointer, and its length must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `SearchByThreshold`<a name="en-us_TOPIC_0000002446061689"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num, int64_t * indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the interface returns the mapped top-<code>k</code> results.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to query.<br><code>const uint16_t *queries</code>: Query feature vectors. The length is <code>n * dim</code>.<br><code>float threshold</code>: Threshold used for filtering. The interface does not restrict the value range. If you pass a mapping table, the interface first maps the distance to a score and then filters by <code>threshold</code>.<br><code>int topk</code>: Sorts the comparison distances between the query and the base library, then returns the top <code>k</code> results.<br><code>unsigned int tableLen</code>: Mapping-table length. The default value is <code>0</code>, which means that no mapping is performed. Currently, the supported mapping-table length is <code>10000</code>.<br><code>const float *table</code>: Mapping-table pointer that points to valid mapped values of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>*table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int *num</code>: Number of base-library vectors that meet the threshold condition for each query feature vector. The length is <code>n</code>.<br><code>int64_t *indices</code>: Indices of base-library vectors that meet the threshold condition. For each query, matching indices are recorded from front to back and the space is padded according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.<br><code>float *distances</code>: Distances between the base-library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: Must be in the range (0, <code>capacity</code> ]. <code>topk</code>: <code>k</code> must be in (0, 1024]. When both <code>tableLen</code> and <code>table</code> meet the requirements, the interface maps the computed <code>distance</code> values.<br>First, normalize <code>distance</code> to a floating-point value <code>f1</code> in [0, 1]. Then multiply <code>f1</code> by <code>tableLen</code> and round it down to obtain an integer index in [0, <code>tableLen</code>]. Next, use the integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>. This completes the mapping and stores <code>score</code> in <code>distance</code>.<br>The index mapping formula can be abstracted as <code>((CosDistance + 1) / 2) * tableLen</code>. <code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `SetNTotal`<a name="en-us_TOPIC_0000002412742486"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetNTotal(int n);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Provides an external way to adjust the <code>ntotal</code> count.<br>After base-library vectors are added, the <code>Index</code> internally updates <code>ntotal</code> according to the maximum inserted index. However, it does not record which areas in the range [0, <code>ntotal</code> ] are invalid space. Therefore, the <code>RemoveFeatures</code> operation does not change the value of <code>ntotal</code>. If you explicitly record the maximum base-library index after add and remove operations in the service layer, you can set <code>ntotal</code> manually. This can reduce the amount of work performed by the operators within a controllable range and improve interface performance.<br>For example, if you currently insert 100 vectors with base-library indices from 0 to 99, then <code>ntotal = 100</code>. If you delete the base-library vectors with indices from 80 to 90, the internal <code>ntotal</code> of <code>Index</code> remains unchanged and can only be set to a value in [ <code>ntotal</code>, <code>capacity</code> ]. If you then delete the base-library vectors with indices from 90 to 99, you can manually set <code>ntotal</code> to a value in [80, <code>capacity</code> ]. When you set it to <code>80</code>, the amount of base-library data participating in the comparison is effectively reduced by 20 vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Maximum base-library index plus 1, managed by the user in the service layer.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code>: Must be in the range [0, <code>capacity</code> ].</td></tr>
</tbody></table>
