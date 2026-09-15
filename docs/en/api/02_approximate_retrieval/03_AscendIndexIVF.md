# `AscendIndexIVF`<a name="en-us_TOPIC_0000001456375220"></a>

## Overview<a name="en-us_TOPIC_0000001506334721"></a>

`AscendIndexIVF` serves as the base class of IVF-based indexes in the feature retrieval component and defines APIs for other IVF indexes in feature retrieval.

For IVF algorithms, the linear scaling on the Atlas 300I Duo inference card depends on the proportion of distance-computation workload in the entire search process. Compared with other computation types, only the distance-computation workload can be evenly distributed across multiple compute units. Therefore, scaling is better in large-batch and large-`nprobe` scenarios, and worse in small-batch and small-`nprobe` scenarios.

> [!NOTE]
> IVF algorithms should follow the rule `nlist * 2MB + resourceSize < NPU-side memory` to avoid memory allocation failures at runtime. For example, if the memory on the NPU card is 64 GB, `nlist` should be smaller than 32768. Since `32768 * 2MB = 64GB`, runtime may exceed the NPU memory size. This limit exists because the current retrieval service prioritizes large-page memory, and the allocation granularity of large-page memory is 2 MB. When every bucket in `nlist` contains data, the hardware allocates memory aligned to the 2 MB granularity. `resourceSize` is the shared memory size specified by the user in `AscendIndexIVFConfig`, and the default value is 128 MB.

## `AscendIndexIVF`<a name="en-us_TOPIC_0000001506414821"></a>

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

## `~AscendIndexIVF`<a name="en-us_TOPIC_0000001506334765"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexIVF();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexIVF</code>. It destroys the <code>AscendIndexIVF</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `copyFrom`<a name="en-us_TOPIC_0000001506334601"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVF* index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle"><code>AscendIndexIVF</code> copies data from an existing <code>index</code> to Ascend and retains the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVF* index</code>: CPU-side index resource.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The <code>probe</code> value of this <code>index</code> must be greater than 0 and less than or equal to <code>nlist</code>.</td></tr>
</tbody></table>

## `copyTo`<a name="en-us_TOPIC_0000001506615113"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVF* index) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies the retrieval resources of <code>AscendIndexIVF</code> to the CPU side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIVF* index</code>: CPU-side index resource.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The resources occupied by <code>Index</code> are released by the user.</td></tr>
</tbody></table>

## `getNumLists`<a name="en-us_TOPIC_0000001506614893"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getNumLists() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the current <code>nlist</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>nlist</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getNumProbes`<a name="en-us_TOPIC_0000001456534948"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getNumProbes() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the current <code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getListCodesAndIds`<a name="en-us_TOPIC_0000001456854940"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the feature vectors and corresponding IDs at a specific <code>nlistId</code> in the current <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId</code>: Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;uint8_t&gt;&amp; codes</code>: Feature vectors at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.<br><code>std::vector&lt;ascend_idx_t&gt;&amp; ids</code>: IDs of the feature vectors at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 ≤ listId &lt; nlist</code>.</td></tr>
</tbody></table>

## `getListLength`<a name="en-us_TOPIC_0000001506614973"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual uint32_t getListLength(int listId) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Returns the length of a specific <code>nlistId</code> in the current <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int listId</code>: Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Length of the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 ≤ listId &lt; nlist</code>.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001506495837"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVF&amp; operator=(const AscendIndexIVF&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment constructor of this index as deleted. Therefore, it is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVF&amp;</code>: Constant <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reclaimMemory`<a name="en-us_TOPIC_0000001506615049"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t reclaimMemory() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Reduces the memory occupied by the base library without changing the number of base-library entries. This API inherits from <code>AscendIndex</code> and provides a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Amount of memory reduced, in bytes.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reserveMemory`<a name="en-us_TOPIC_0000001506334617"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reserveMemory(size_t numVecs) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Abstract API that reserves memory for the base library before the base library is built. This API inherits from <code>AscendIndex</code> and provides a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t numVecs</code>: Number of base-library vectors for which to reserve memory.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">In a single-card environment: <code>0 &lt; numVecs ≤ 2e8</code>. In a multi-card environment: <code>0 &lt; numVecs ≤ 1e9</code> (<code>numVecs</code> divided by the number of cards must be smaller than <code>2e8</code>). Exceeding the limit throws an exception and stops the program.</td></tr>
</tbody></table>

## `reset`<a name="en-us_TOPIC_0000001506414685"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void reset() override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clears the base-library vectors of this <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `setNumProbes`<a name="en-us_TOPIC_0000001506614937"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void setNumProbes(int nprobes);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the current <code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int nprobes</code>: <code>nprobe</code> value of <code>AscendIndexIVF</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 &lt; nprobes ≤ nlist</code>.</td></tr>
</tbody></table>
