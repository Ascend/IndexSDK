# `AscendIndexMixSearchParams`<a name="en-us_TOPIC_0000002008910258"></a>

## Overview<a name="en-us_TOPIC_0000002045034929"></a>

The `AscendIndexMixSearchParams.h` file provides the structures required by `AscendIndexGreat` and `AscendIndexVStar`.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must acquire a lock before use; otherwise, the search API may cause exceptions. Sharing a single device across different threads is also not supported.

## `AscendIndexGreatInitParams`<a name="en-us_TOPIC_0000002049404289"></a>

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

## `AscendIndexVstarInitParams`<a name="en-us_TOPIC_0000002013246410"></a>

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

## `AscendIndexVstarHyperParams`<a name="en-us_TOPIC_0000002013404694"></a>

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

## `AscendIndexHyperParams`<a name="en-us_TOPIC_0000002049325253"></a>

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

## `AscendIndexSearchParams`<a name="en-us_TOPIC_0000002044950949"></a>

<a name="table414612258177"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexSearchParams(size_t n, std::vector&lt;float&gt;&amp; queryData, int topK, std::vector&lt;float&gt;&amp; dists, std::vector&lt;int64_t&gt;&amp; labels);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Search parameter structure for retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameters</td><td valign="middle"><code>size_t n</code>: Number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature-vector data.<br><code>int topK</code>: Number of most similar results to return.<br><code>std::vector&lt;float&gt;&amp; dists</code>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>topK</code> ∈ (0, 4096]. <code>n</code> ∈ (0, 10000]. <code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>. <code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. <code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</td></tr>
</tbody></table>
