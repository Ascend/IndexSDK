# `AscendIndexIVFSP`<a name="en-us_TOPIC_0000001635576081"></a>

## Overview<a name="en-us_TOPIC_0000001635815481"></a>

The Ascend-native IVFSP retrieval algorithm uses an in-house matrix approximation strategy to compress feature vectors before storing them in the base library. It then uses an in-house inverted-list strategy to select the base-library entries most likely to contain the ground truth. Finally, it uses an in-house retrieval strategy on the filtered base library to obtain the top K vector results.

`AscendIndexIVFSP` supports only standard mode scenarios and Atlas Inference Series products.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `add`<a name="en-us_TOPIC_0000001585895568"></a>

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

## `add_with_ids`<a name="en-us_TOPIC_0000001586055512"></a>

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

## `AscendIndexIVFSP`<a name="en-us_TOPIC_0000001585736168"></a>

> [!NOTE]
>
> - Before you pass parameter `config` to the function, set the values of `conf.handleBatch`, `conf.nprobe`, and `conf.searchListSize` according to the actual situation. For field descriptions, see [Common Parameters](./06_AscendIndexIVFSPConfig.md#en-us_TOPIC_0000001635696057).
> - The values of `conf.handleBatch` and `conf.searchListSize` must be consistent with the `nprobe handle batch` and `search list size` values used when generating the [IVFSP](../../05_user_guide.md#ivfsp) service operator model file.
> - `conf.filterable`, inherited from [AscendIndexConfig](../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig) false by default. If you want to use the `search_with_filter()` API, set `conf.filterable = true`. Setting `conf.filterable` to `true` stores extra information on the NPU card and consumes more NPU-side memory.

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

## `loadAllData API`<a id="en-us_TOPIC_0000001585736172"></a>

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

## `operator=`<a name="en-us_TOPIC_0000001635975413"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSP&amp; operator=(const AscendIndexIVFSP&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSP&amp;:</code> A constant <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `remove_ids`<a name="en-us_TOPIC_0000001635576085"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implement the API for deleting the specified feature vectors from the base vector set in <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel:</code> Feature vectors to delete. For details about the usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">The number of deleted feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reset`<a name="en-us_TOPIC_0000001635815485"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clear the base vectors in this <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `saveAllData`<a name="en-us_TOPIC_0000001635696053"></a>

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

## `search`<a name="en-us_TOPIC_0000001635815489"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implement the feature vector search API for <code>AscendIndexIVFSP</code>, and return the IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n:</code> Number of query feature vectors.<br><code>const float *x:</code> Feature vector data.<br><code>idx_t k:</code> Number of most similar results to return.<br><code>const SearchParameters *params:</code> Optional Faiss parameter. The default value is <code>nullptr</code>, and this parameter is currently unsupported.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances:</code> Distance values between the query vectors and the top <code>k</code> nearest vectors. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid distances with 65504 or <code>-65504</code>.<br><code>idx_t *labels:</code> IDs of the top <code>k</code> nearest vectors to the query. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid labels with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of the query feature vector data <code>x</code> should be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The value range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096.</td></tr>
</tbody></table>

## `search_with_filter`<a name="en-us_TOPIC_0000001585736176"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Feature vector search API for <code>AscendIndexIVFSP</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It also provides CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array with a length of <code>n * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four numbers of each filter, which are 128 bits, represent the corresponding CID. The last two numbers represent the left-closed timestamp range, that is, [<code>x</code>, <code>y</code>).</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n:</code> Number of query feature vectors.<br><code>const float *x:</code> Feature vector data.<br><code>idx_t k:</code> Number of most similar results to return.<br><code>const void *filters:</code> Filter conditions.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances:</code> Distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels:</code> IDs of the top <code>k</code> nearest vectors to the query.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value range of <code>n</code> is <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096. <code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and their lengths should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>filters</code> must be a non-null pointer, and its length must be a <code>uint32_t</code> array of <code>n * 6</code>. Otherwise, out-of-bounds reads may occur and cause the program to crash.</td></tr>
</tbody></table>

## `setNumProbes`<a name="en-us_TOPIC_0000001635576089"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setNumProbes(int nprobes);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Set the total number of candidate buckets used during search.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobes:</code> <code>nprobe</code> count of <code>AscendIndexIVFSP</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobes</code> must be a multiple of 16 and satisfy <code>0 &lt; nprobes &lt;= nlist</code>.</td></tr>
</tbody></table>

## `setVerbose`<a name="en-us_TOPIC_0000001586055516"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setVerbose(bool verbose);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Set whether to print the progress of adding feature vectors to the base vector set.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>bool verbose:</code> Whether to print the progress of adding feature vectors to the base vector set.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `trainCodeBook`<a name="en-us_TOPIC_0000002148530670"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void trainCodeBook(const AscendIndexCodeBookInitParams &amp;codeBookInitParams) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">IVFSP codebook training API. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable <code>OMP_NUM_THREADS=4</code> to speed it up.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">const AscendIndexCodeBookInitParams &amp;codeBookInitParams: Initialization parameters required for codebook training.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See the <code>AscendIndexCodeBookInitParams</code> API.</td></tr>
</tbody></table>

## `addCodeBook`<a name="en-us_TOPIC_0000002148372594"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void addCodeBook(const char *codeBookPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Add a trained codebook.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">const char *codeBookPath: Codebook path.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The file corresponding to <code>codeBookPath</code> should be the codebook file produced by <code>trainCodeBook</code>, and the process user must have read permission for it. The file must not be a symbolic link.</td></tr>
</tbody></table>

## `AscendIndexCodeBookInitParams`<a name="en-us_TOPIC_0000002183731529"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum, std::string codeBookOutputDir, std::string learnDataPath, bool verbose);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization structure for IVFSP codebook training.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Values</td><td valign="middle"><code>int numIter:</code> Number of training iterations. The default value is 1.<br><code>int device:</code> Logical device ID. The default value is 0.<br><code>float ratio:</code> Sampling rate of the original samples used for training. The default value is <code>1.0</code>.<br><code>int batchSize:</code> Train with batches of size <code>batchSize</code>. This value must match <code>&lt;batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The default value is 32768.<br><code>int codeNum:</code> Operate on at most <code>codeNum</code> samples at a time when updating the codebook. This value must be a power of two and must match <code>&lt;codebook_batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The default value is 32768.<br><code>std::string codeBookOutputDir:</code> Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.<br><code>std::string learnDataPath:</code> Path to the original feature file used for training. The file supports the bin and npy formats. For bin files, the storage order is row-major and the data type is <code>float32</code>.<br><code>bool verbose:</code> Whether to enable additional output. The default value is <code>true</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Constraints</td><td valign="middle"><code>numIter</code> ∈ (0, 20]. <code>ratio</code> ∈ (0, 1.0]. <code>batchSize</code> ∈ (0, 32768]. <code>codeNum</code> ∈ (0, 32768]. When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner. Before you run codebook training, refer to the <code>IVFSP</code> operator model file generation instructions.</td></tr>
</tbody></table>

## `trainCodeBookFromMem`<a name="en-us_TOPIC_0000002257319034"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">IVFSP codebook training API. Training data is loaded from memory. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable <code>OMP_NUM_THREADS=4</code> to speed it up.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams: Initialization parameters required for codebook training.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Constraints</td><td valign="middle">For details about <code>AscendIndexCodeBookInitFromMemParams</code>, see <code>AscendIndexCodeBookInitFromMemParams</code>.</td></tr>
</tbody></table>

## `AscendIndexCodeBookInitFromMemParams`<a name="en-us_TOPIC_0000002291969193"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCodeBookInitFromMemParams (int numIter, int device, float ratio, int batchSize, int codeNum,bool verbose,std::string codeBookOutputDir,const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initialization structure for IVFSP codebook training. Training data is loaded from memory.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Values</td><td valign="middle"><code>int numIter:</code> Number of training iterations. The default value is 1.<br><code>int device:</code> Logical device ID. The default value is 0.<br><code>float ratio:</code> Sampling rate of the original samples used for training. The default value is <code>1.0</code>.<br><code>int batchSize:</code> Train with batches of size <code>batchSize</code>. This value must match <code>&lt;batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The value must be greater than 0, and the default value is 32768.<br><code>int codeNum:</code> Operate on at most <code>codeNum</code> samples at a time when updating the codebook. This value must be a power of two and must match <code>&lt;codebook_batch_size&gt;</code> in the <code>IVFSP</code> training operator model file generation section. The value must be greater than 0, and the default value is 32768.<br><code>std::string codeBookOutputDir:</code> Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.<br><code>bool verbose:</code> Whether to enable additional output. The default value is <code>true</code>.<br><code>const float *memLearnData:</code> Pointer to in-memory data. The default value is a null pointer.<br><code>size_t memLearnDataSize:</code> Length of the in-memory data. The default value is 0.<br><code>bool isTrainAndAdd:</code> Whether to add the codebook directly to the <code>Index</code> after training. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter Constraints</td><td valign="middle"><code>numIter</code> ∈ (0, 20]. <code>ratio</code> ∈ (0, 1.0]. <code>memLearnDataSize % dim == 0</code>. <code>memLearnDataSize &lt;= 25G</code>. When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner. Before you run codebook training, refer to the <code>IVFSP</code> operator model file generation instructions. When <code>isTrainAndAdd</code> is <code>true</code>, the trained codebook is added directly to the <code>Index</code> and is not written to disk. When <code>isTrainAndAdd</code> is <code>false</code>, the codebook is saved to <code>codeBookOutputDir</code>, and you must call <code>addCodeBook</code> manually. <code>memLearnDataSize</code> must be the actual length of the <code>memLearnData</code> pointer. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>
