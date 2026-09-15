# `AscendIndexIVFSQT`<a name="en-us_TOPIC_0000001456375224"></a>

## Overview<a name="en-us_TOPIC_0000001506615005"></a>

The `AscendIndexIVFSQT` class contains the three-stage retrieval `IVFSQ` algorithm with dimensionality reduction. You need to pass two parameters to specify the dimensions before and after dimensionality reduction, and the original dimension must be divisible by the reduced dimension. It is suitable for scenarios with a base vector set on the order of 10 million.

You need to generate the operators required for three-stage retrieval according to the `IVFSQT` operator generation method.

This type provides fuzzy clustering. Before bucket assignment, use the `threshold` parameter to control the degree of fuzziness. Set the `threshold` value according to the base vector set capacity and the available memory size. A `threshold` that is too large can cause insufficient memory and lead to failure. For Atlas 200/300/500 inference product environments, you are advised to set it to [1.0, 1.1]. For Atlas inference series product environments, you are advised to set it to [1.0, 1.5]. For search, you are advised to use `batch size = 65536`.

The workflow is: 1. Construct the `Index` object. 2. Train the data. 3. Add the data. 4. Update the data. 5. Search the data. 6. Destroy the `Index` object. After `update`, adding data is no longer supported. If you need to search new data, destroy the original `Index` object and use the workflow again from the beginning.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval internally uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

## `AscendIndexIVFSQT`<a name="en-us_TOPIC_0000001506495685"></a>

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

## `~AscendIndexIVFSQT`<a name="en-us_TOPIC_0000001456854984"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSQT();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor for <code>AscendIndexIVFSQT</code>. It destroys the <code>AscendIndexIVFSQT</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `copyFrom`<a name="en-us_TOPIC_0000001456695060"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>index</code> to Ascend based on <code>AscendIndexIVFSQT</code>, while preserving the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer.<br><code>index-&gt;d</code> ∈ {256}. <code>index-&gt;sq.d</code> ∈ {32, 64, 128}. The dimension of <code>index</code> must be greater than the dimension of <code>index-&gt;sq</code>, and it must be divisible by the dimension of <code>index-&gt;sq</code>. Do not call this API on an updated object.</td></tr>
</tbody></table>

## `copyTo`<a name="en-us_TOPIC_0000001506495825"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy the retrieval resources of <code>AscendIndexIVFSQT</code> to the CPU side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user frees the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

## `fineTune`<a name="en-us_TOPIC_0000001456694860"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void fineTune(size_t n, const float *x);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Fine-tune and optimize the centroids to avoid uneven bucket assignment.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n:</code> Number of feature vectors.<br><code>const float *x:</code> Feature vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `getFuzzyK`<a name="en-us_TOPIC_0000001456855008"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getFuzzyK() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Get the maximum value used when a vector is assigned to buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>int:</code> Maximum value used when a vector is assigned to buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getListCodesAndIds`<a name="en-us_TOPIC_0000001687739112"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the feature vectors and corresponding IDs for a specific <code>nlistId</code> in the current <code>AscendIndexIVFSQT</code> <code>nlist</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId:</code> Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;uint8_t&gt;&amp; codes:</code> Feature vectors at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.<br><code>std::vector&lt;ascend_idx_t&gt;&amp; ids:</code> Feature vector IDs at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `getListLength`<a name="en-us_TOPIC_0000001735977797"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>uint32_t getListLength(int listId) const override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the length for a specific <code>nlistId</code> in the current <code>AscendIndexIVFSQT</code> <code>nlist</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId:</code> Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Length at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `getLowerBound`<a name="en-us_TOPIC_0000001506614885"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getLowerBound() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the threshold for second-level clustering.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Threshold for second-level clustering.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getMergeThres`<a name="en-us_TOPIC_0000001506615073"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getMergeThres() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Get the threshold for merging sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Threshold for merging sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getQMax`<a name="en-us_TOPIC_0000001456535208"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>float getQMax() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the maximum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Maximum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getQMin`<a name="en-us_TOPIC_0000001506615029"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>float getQMin() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Return the minimum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Minimum feature vector value.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getThreshold`<a name="en-us_TOPIC_0000001506334633"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>float getThreshold() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Get the threshold used to determine whether a vector is assigned to multiple buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>float:</code> Threshold used to determine whether a vector is assigned to multiple buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001506615085"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQT&amp; operator=(const AscendIndexIVFSQT&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSQT&amp;</code>: An <code>AscendIndexIVFSQT</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `remove_ids`<a name="en-us_TOPIC_0000001506615053"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes base library features by ID.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: The feature vectors to delete. For details about usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">The number of deleted feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version.</td></tr>
</tbody></table>

## `reset`<a name="en-us_TOPIC_0000001506334789"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Resets the index and clears the feature data.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Do not continue using this object after you call this API.</td></tr>
</tbody></table>

## `setAddTotal`<a name="en-us_TOPIC_0000001456375316"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setAddTotal(size_t addTotal);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the total number of base library vectors to add. The default value is 100000000. You must set <code>PreciseMemControl</code> to <code>true</code> first.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t addTotal</code>: The total number of base library vectors to add.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `setFuzzyK`<a name="en-us_TOPIC_0000001456534940"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setFuzzyK(int value);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the maximum value for each vector when it is assigned to a bucket.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int value</code>: The maximum value for each vector when it is assigned to a bucket. You are advised to keep it at the default value 3.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>value</code> is (0, 10].</td></tr>
</tbody></table>

## `setLowerBound`<a name="en-us_TOPIC_0000001506334777"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setLowerBound(int lowerBound);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the threshold for second-level clustering. The default value is 32.<br>If the number of elements in a first-level clustering bucket is greater than <code>lowerBound</code>, second-level clustering is performed. Otherwise, the original state is retained.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int lowerBound</code>: The threshold for second-level clustering.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `setMemoryLimit`<a name="en-us_TOPIC_0000001506614917"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setMemoryLimit(float memoryLimit);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the Host memory limit. The default value is 32, in <code>GB</code>. You must set <code>PreciseMemControl</code> to <code>true</code> first.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>float memoryLimit</code>: The memory limit.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `setMergeThres`<a name="en-us_TOPIC_0000001456694900"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setMergeThres(int mergeThres);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the threshold for merging sub-buckets. The default value is 5.<br>If the number of elements in a sub-bucket after second-level clustering is smaller than <code>mergeThres</code>, merge the elements of that sub-bucket into other sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int mergeThres</code>: The threshold for merging sub-buckets.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `setNumProbes`<a name="en-us_TOPIC_0000001736410013"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setNumProbes(int nprobes) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the <code>nprobe</code> value of the current <code>AscendIndexIVFSQT</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobes</code>: The <code>nprobe</code> value of <code>AscendIndexIVFSQT</code>. You are advised to keep it at the default value 64.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobes</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. <code>l2Probe</code> ≥ <code>nprobes</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, and <code>l2Probe</code> ≤ <code>nprobes * 64</code>. <code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}. For details about <code>l2Probe</code> and <code>l3SegmentNum</code>, see <code>setSearchParams</code>. <code>setNumProbes</code> is expected to be removed in September 2025. Use <code>setSearchParams</code> instead.</td></tr>
</tbody></table>

## `setPreciseMemControl`<a name="en-us_TOPIC_0000001506334681"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setPreciseMemControl(bool preciseMemControl);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Specifies whether to precisely limit the memory size on the Host side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>bool preciseMemControl</code>: The default value is <code>false</code>, which disables precise memory limiting on the Host side. <code>true</code> enables it.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is not supported in the current version. Do not call it.</td></tr>
</tbody></table>

## `setSearchParams`<a name="en-us_TOPIC_0000002052679693"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the parameters that affect retrieval accuracy and performance.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobe</code>: The <code>nprobe</code> value of <code>AscendIndexIVFSQT</code>. You are advised to keep it at the default value 64.<br><code>int l2Probe</code>: The number of sub-buckets selected during second-stage retrieval. The default value is 48.<br><code>int l3SegmentNum</code>: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is 96.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobe</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. <code>l2Probe</code> ≥ <code>nprobe</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, and <code>l2Probe</code> ≤ <code>nprobe * 64</code>.<br><code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}.</td></tr>
</tbody></table>

## `setSortMode`<a name="en-us_TOPIC_0000002165943965"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setSortMode(int mode);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the <code>topk</code> sorting mode. Mode 0 means approximate sorting. Mode 1 means exact sorting.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int mode</code>: The <code>topk</code> sorting mode.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">You must call this API before the <code>Search</code> API. <code>mode</code> supports only 0 or 1, and the default is 0. Mode 0: Approximate sorting truncates part of the <code>topk</code> results to improve performance. Mode 1: Exact sorting improves retrieval accuracy at the cost of some performance.</td></tr>
</tbody></table>

## `setThreshold`<a name="en-us_TOPIC_0000001456854808"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void setThreshold(float value);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the threshold for determining whether a vector is assigned to multiple buckets. The default value is <code>1.0</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>float value</code>: The threshold for determining whether a vector is assigned to multiple buckets. You are advised to set it in the range [1.0, 1.5]. Because the Device side has a memory limit, once memory usage reaches the limit, the OOM mechanism is triggered and kills the process. You can check the Device-side memory limit data first (<code>/sys/fs/cgroup/memory/usermemory/memory.limit_in_bytes</code>) to estimate the size of the base library to add. If memory is tight, you are advised to keep the parameter in the range [1.0, 1.1].</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>value</code> is [0, <code>fuzzyK</code> - 1]. For the valid range of <code>fuzzyK</code>, see the <code>getFuzzyK</code> API.</td></tr>
</tbody></table>

## `setUseCpuUpdate`<a name="en-us_TOPIC_0000002167379329"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>setUseCpuUpdate(int numThreads);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Specifies whether to use the CPU for update.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int numThreads</code>: The number of CPU cores used for update. The default value is the current number of CPU cores.<br>If the current CPU has more than 96 cores: if the current core count is smaller than the input <code>numThreads</code>, set <code>numThreads</code> to 96; if <code>96 &lt; numThreads &lt;=</code> the current core count, set <code>numThreads</code> to 96; if <code>numThreads &lt;= 96</code>, keep the input value. If the current CPU has 96 cores or fewer: if the current core count is smaller than the input <code>numThreads</code> and <code>numThreads &lt;= 96</code>, set <code>numThreads</code> to the current core count; if <code>0 &lt; numThreads &lt;=</code> the current core count, keep the input value.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>numThreads</code> must be greater than 0. Configure it before you use <code>update</code>.</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000001456375352"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Trains <code>AscendIndexIVFSQT</code>. This class inherits the relevant APIs in <code>AscendIndexIVFSQ</code> and provides concrete implementations.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: The number of feature vectors in the training set.<br><code>const float *x</code>: Feature vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering. A training set that is too small may affect query accuracy. The valid range of <code>n</code> here is <code>nlist ≤ n ≤ 7,000,000</code>. The pointer <code>x</code> must be a non-null pointer, and its length must be <code>dimIn * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `update`<a name="en-us_TOPIC_0000001506414869"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void update(bool cleanData = true);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">This is the second stage of three-stage retrieval. After all base library data has been added and before <code>search</code> is called, this API trains sub-bucket centers and assigns vectors to buckets according to those centers.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>cleanData</code>: Specifies whether to clear intermediate data. The default value is <code>true</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">You only need to call this API once in a full retrieval workflow.</td></tr>
</tbody></table>

## `updateTParams`<a name="en-us_TOPIC_0000001456854936"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void updateTParams(int l2Probe, int l3SegmentNum);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Passes in the parameters required for three-stage retrieval during testing.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int l2Probe</code>: The number of sub-buckets selected during second-stage retrieval. The default value is 48.<br><code>int l3SegmentNum</code>: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is 96.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nprobe</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. <code>l2Probe</code> ≥ <code>nprobe</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, and <code>l2Probe</code> ≤ <code>nprobe * 64</code>.<br><code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}. For details about the <code>nprobe</code> setting, see <code>setSearchParams</code>. <code>updateTParams</code> is expected to be removed in September 2026. Use <code>setSearchParams</code> instead.</td></tr>
</tbody></table>
