# `IndexIL`<a name="en-us_TOPIC_0000001506414825"></a>

## Overview<a name="en-us_TOPIC_0000001456535188"></a>

`IndexIL` is a feature management abstract class based on a contiguous memory allocation mechanism. It serves retrieval algorithms that use indices as labels. To use it, you must inherit from it and implement all interfaces.

The vectors stored in the base library and the query vectors of each API must be normalized FP16 floating-point values. (`IL` stands for "Indices as Labels".)

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, the user must lock before use. Otherwise, the retrieval APIs may raise exceptions. It also does not support sharing a Device across different threads.

## `AddFeatures`<a name="en-us_TOPIC_0000001506414693"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Inserts <code>n</code> feature vectors with specified indices into the feature library. If a feature vector already exists at an index, this insertion is equivalent to an update.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: The number of feature vectors to insert.<br><code>const float16_t *features</code>: Feature vectors, with a length of <code>n * vector dimension dim</code>.<br><code>const idx_t *indices</code>: The index values corresponding to the feature vectors, with a length of <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The return status of the call. For details, see the reference for API return values.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The input parameters are constrained by the implementation class. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## `IndexIL`<a name="en-us_TOPIC_0000001456695020"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>IndexIL();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>IndexIL</code>. It creates a feature management object.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `~IndexIL`<a name="en-us_TOPIC_0000001506334781"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~IndexIL();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">The destructor of <code>IndexIL</code>. It destroys the feature management object.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `Finalize`<a name="en-us_TOPIC_0000001456375356"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR Finalize() = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Releases the feature library management resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The return status of the call. For details, see the reference for API return values.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `GetFeatures`<a name="en-us_TOPIC_0000001506495833"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the feature vectors for <code>n</code> specified index values.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: The number of feature vectors to obtain.<br><code>const idx_t *indices</code>: The index values to query, with a length of <code>n</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float16_t *features</code>: The feature vectors corresponding to the queried indices, with a length of <code>n * vector dimension dim</code>. The user must allocate memory before the call and ensure that the memory size is correct.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The return status of the call. For details, see the reference for API return values.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The input parameters are constrained by the implementation class. <code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## `GetNTotal`<a name="en-us_TOPIC_0000001456535092"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual int GetNTotal() const = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the maximum occupied space of the current feature library vectors.<br>Feature vectors are inserted starting from index <code>0</code>. If the inserted feature vector indices are continuous, <code>ntotal</code> equals the number of feature vectors. Otherwise, <code>ntotal</code> equals the maximum inserted index value plus <code>1</code>. For performance reasons, the operator batches memory operations and, by default, treats the space at and before the maximum index position as valid base library vectors and includes it in the calculation. The user must use this API to obtain the total number of base library entries recorded inside the <code>Index</code>, and then allocate the corresponding memory space to pass parameters to the corresponding functional APIs. For details, see the specific API.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>int ntotal</code>: See the description.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `Init`<a name="en-us_TOPIC_0000001506334657"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initializes feature library parameters and allocates base library memory resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dim</code>: Feature vector dimension.<br><code>AscendMetricType metricType</code>: Feature distance type, including inner product, Euclidean distance, and cosine similarity.<br><code>int capacity</code>: Maximum base library capacity. The allocated memory size is <code>capacity * dim * sizeof(float)</code> bytes.<br><code>int resourceSize</code>: Preallocates Device-side cache resources. When a retrieval API is called, it can use these resources directly instead of calling <code>aclrtmalloc</code> to allocate memory, which improves performance. The default value is <code>-1</code>, which means the cache resource is allocated with the default size of <code>128 MB</code>. You can configure the actual size more precisely based on the retrieval workload and Device-side resource usage.<br>For example, if the query batch size is <code>64</code>, the base library contains 1,000,000 vectors, and one FP32 value occupies 4 bytes, set <code>resourceSize</code> to <code>64 * 1000000 * 4 = 256,000,000</code> bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The implementation class constrains the input parameters.</td></tr>
</tbody></table>

## RemoveFeatures API<a name="en-us_TOPIC_0000001456534932"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR RemoveFeatures(int n, const idx_t *indices) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Deletes the feature vectors with the specified indices from the vector library.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of feature vectors to delete.<br><code>const idx_t *indices</code>: Indices of the feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The implementation class constrains the input parameters. <code>indices</code> must be a non-null pointer, and its length must be <code>n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</td></tr>
</tbody></table>

## SetNTotal API<a name="en-us_TOPIC_0000001456375256"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual APP_ERROR SetNTotal(int n) = 0;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Provides an interface for adjusting the <code>ntotal</code> count externally.<br>After base library vectors are added, the <code>Index</code> internally updates the <code>ntotal</code> value according to the largest inserted index, but it does not record which regions in the range [0, <code>ntotal</code> ] are invalid. Therefore, the <code>RemoveFeatures</code> operation does not change the <code>ntotal</code> value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set <code>ntotal</code> manually. This reduces the operator workload within a controllable range and improves interface performance.<br>For example, if 100 vectors are inserted and the base library indices range from 0 to 99, <code>ntotal = 100</code>. If you delete the base library entries with indices from 80 to 90, the <code>ntotal</code> value inside <code>Index</code> remains unchanged and can only be set to a value in [ <code>ntotal</code>, <code>capacity</code> ]. If you then delete the base library entries with indices from 90 to 99, you can manually set <code>ntotal</code> to a value in [80, <code>capacity</code> ]. When you set it to <code>80</code>, the amount of base library data involved in comparison decreases by 20 vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Maximum base library index managed by the service side, plus 1.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status. For details, see the interface return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The implementation class constrains the input parameters.</td></tr>
</tbody></table>
