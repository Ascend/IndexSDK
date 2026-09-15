# `AscendIndexSQ`<a name="en-us_TOPIC_0000001506614969"></a>

## Overview<a name="en-us_TOPIC_0000001456695120"></a>

`AscendIndexSQ` performs Scalar Quantization on the input vectors.

The vectors stored in the base library and the query vectors of each API must be normalized float values.

It supports multithreaded concurrent calls. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

## `AscendIndexSQ`<a name="en-us_TOPIC_0000001506614933"></a>

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

## `copyFrom`<a name="en-us_TOPIC_0000001506615037"></a>

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

## `copyTo`<a name="en-us_TOPIC_0000001456695084"></a>

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

## `getBase`<a name="en-us_TOPIC_0000001456694928"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getBase(int deviceId, char* xb) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vectors managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>char* xb</code>: The base library feature vectors stored by <code>AscendIndexSQ</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID. <code>xb</code> must be a non-null pointer, and its length must be <code>dims * BaseSize * sizeof(uint8_t)</code> bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>BaseSize</code> is the return value of <code>getBaseSize</code>.</td></tr>
</tbody></table>

## `getBaseSize`<a name="en-us_TOPIC_0000001456854788"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t getBaseSize(int deviceId) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the number of feature vectors managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of feature vectors on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

## `getIdxMap`<a name="en-us_TOPIC_0000001456375152"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt;&amp; idxMap) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Gets the feature vector IDs managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int deviceId</code>: Device-side device ID.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;idx_t&gt; &amp;idxMap</code>: The base library feature vector IDs stored by <code>AscendIndexSQ</code> on <code>deviceId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceId</code> must be a valid device ID.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001456375300"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSQ&amp; operator=(const AscendIndexSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexSQ&amp;</code>: An <code>AscendIndexSQ</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `search_with_filter`<a name="en-us_TOPIC_0000001810589742"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The feature vector query API of <code>AscendIndexSQ</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It also provides CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array of length <code>n * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four values of each filter, that is, 128 bits, represent the corresponding CID. The last two values represent the left-closed timestamp interval, that is, [<code>x</code>, <code>y</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of query feature vectors.<br><code>const float *x</code>: Feature vector data.<br><code>idx_t k</code>: The number of most similar results to return.<br><code>const void *filters</code>: Filter conditions.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>filters</code> must be a non-null pointer to a <code>uint32_t</code> array of length <code>n * 6</code>. Otherwise, out-of-bounds read errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## `search_with_masks`<a name="en-us_TOPIC_0000001456694932"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The feature vector query API of <code>AscendIndexSQ</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. <code>mask</code> is a bit string of <code>0</code>s and <code>1</code>s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. <code>1</code> means participate, and <code>0</code> means do not participate.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of query feature vectors.<br><code>const float *x</code>: Feature vector data.<br><code>idx_t k</code>: The number of most similar results to return.<br><code>const void *mask</code>: Feature library mask.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>mask</code> must be a non-null pointer, and its length must be <code>n * ceil(ntotal / 8)</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>ntotal</code> is the number of base library features. <code>mask</code> is set according to the order of the base library. If you call <code>remove_ids</code> to delete feature vectors before calling this API, the order of the base library features changes. First call <code>getIdxMap</code> to obtain the IDs of the base library features, and then set <code>mask</code>. To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect.</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000001506414905"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Trains the quantizer on <code>AscendIndexSQ</code>. This API inherits the interface from <code>AscendFaiss</code> and provides the concrete implementation. **Note that you must train the <code>Index</code> before you call <code>add</code>.**</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of feature vectors in the training set.<br><code>const float *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>. <code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. Training collects the data distribution. A small training set may affect query accuracy.</td></tr>
</tbody></table>
