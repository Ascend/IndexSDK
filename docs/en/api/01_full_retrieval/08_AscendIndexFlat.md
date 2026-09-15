# `AscendIndexFlat`<a id="en-us_TOPIC_0000001506334757"></a>

## Overview<a name="en-us_TOPIC_0000001506334829"></a>

`AscendIndexFlat` is the most basic feature retrieval algorithm. It stores FP16 floating-point feature vectors and performs brute-force search.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval uses OMP internally for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> `AscendIndexFlat` supports online operator conversion for L2 and IP distances. If the environment variable `MX_INDEX_USE_ONLINEOP` is set to `1` (set it with `export MX_INDEX_USE_ONLINEOP=1`), the operator is converted and called online. To use online operators, the application must explicitly call `(void)aclFinalize()` at the end. You also need to include the header file `#include "acl/acl.h"`.

## `AscendIndexFlat`<a name="en-us_TOPIC_0000001456375308"></a>

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

## `copyFrom`<a name="en-us_TOPIC_0000001456535180"></a>

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

## `copyTo`<a name="en-us_TOPIC_0000001456535148"></a>

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

## `getBase`<a name="en-us_TOPIC_0000001456375236"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getBase(int deviceId, char* xb) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vectors managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>char* xb</code>: The base library feature vectors stored by <code>AscendIndexFlat</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.<br><code>xb</code> must be a non-null pointer, and its length must be <code>dims * BaseSize * sizeof(float32)</code> bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>BaseSize</code> is the return value of <code>getBaseSize</code>.</td></tr>
</tbody></table>

## `getBaseSize`<a name="en-us_TOPIC_0000001456854956"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t getBaseSize(int deviceId) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the number of feature vectors managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of feature vectors on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

## `getIdxMap`<a name="en-us_TOPIC_0000001506334785"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vector IDs managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;idx_t&gt; &amp;idxMap</code>: The base library feature vector IDs stored by <code>AscendIndexFlat</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001506495701"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexFlat&amp; operator=(const AscendIndexFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexFlat&amp;</code>: A constant <code>AscendIndexFlat</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `search_with_masks`<a name="en-us_TOPIC_0000001810529650"></a>

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
