# `AscendIndexCagra`<a name="en-us_TOPIC_0000002513157730"></a>

## Overview<a name="en-us_TOPIC_0000002544797645"></a>

`AscendIndexCagra` builds an FP32/L2 graph and searches it with RaBitQ on a single device from the Ascend 950 PR series. The unified API owns construction, base quantization and per-call query preparation. The external-graph API remains available for compatibility.

## Unified index API

Include `faiss/ascend/AscendIndexCagra.h`. All buffers are host buffers. Methods returning `APP_ERROR` return zero on success.

| Method | Contract |
| --- | --- |
| `Init(int dim, const std::vector<int>& devices, const AscendIndexCagraConfig& config = {})` | Configure an empty index. One nonnegative device ID is required. |
| `Train(int n, const float* data)` | Build the full graph and quantize the base. Requires finite FP32 `data[n][dim]`, `M<n<=1e9` and sufficient host/device memory. Failed training preserves the original index state. |
| `Add(int n, const float* data)` | Build an empty index, without incremental insertion. Returns APP_ERR_INVALID_PARAM on a trained index and preserves its data; Reset before rebuilding. Repeated Train remains a no-op. |
| `Search(int n, const float* queries, int k, float* distances, uint32_t* labels)` | Quantize this request and search. Requires `1<=n<=4096`, `k in {32,64,128,256,512,1024}`, `k<=N` and output buffers of at least `n*k` elements. Different k values may be used on the same index. |
| `Reset()` | Release graph, dataset, quantization and operator caches while retaining the unified configuration. |
| `GetKnnGraph(std::vector<uint32_t>& graph) const` | Copy the trained graph to a host vector of N*G IDs. Failure preserves the destination. |
| `SaveGraphToFile(const std::string& filePath) const` | Export the trained graph as N*G row-major native-endian uint32_t values without a header, vectors or quantization data. Overwrites an existing file. Rejects empty paths or embedded NULs; the parent directory must exist. Returns APP_ERR_INDEX_NOT_INIT before training and a nonzero code on file errors. A failed write may leave a partial file; the index remains usable. |
| `IsTrained() const` / `GetSize() const` | Return readiness / trained row count. An empty index has size zero. |

`AscendIndexCagraConfig` contains `graphDegree=64` (64 or 128), `intermediateDegree=128` (G<=M<=128 and M<N), `maxIterations=20` (>0), `terminationThreshold=0.0001` (finite, [0,1]), `guaranteeConnectivity=true` and `verbose=false`. Both build and search support dimensions {64,128,256,512,1024,1536,3072}. RaBitQ distances are returned without FP32 reranking.

Generate graph-build models and the seven query-page models matching N, D, G and every requested k. Search models are loaded lazily and cached per k; changing k does not rebuild the graph. Base quantization belongs to Train/Add, and query quantization belongs to Search. Construction configuration belongs to AscendIndexCagraConfig; graph retrieval and file export use the same Index. The legacy external-graph APIs below remain available; do not mix their Init/Add calling mode with unified Init.

After installing the operator package, run `python3 cagra_generate_model.py -data_base N -d D --intermediate-degree 128 -degree G -topK K --cores 56 -t Ascend950PR`. This generates six graph-build models followed by seven search-page models by default, using one N/D/G configuration. The generation command supports G=64/128 and G<=M<=128 with M<N.

## `AscendIndexCagra`<a name="en-us_TOPIC_0000002513317664"></a>

<a name="table_cagra_ctor"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCagra();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor of <code>AscendIndexCagra</code>, which creates an <code>Index</code> instance for Cagra retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table_cagra_delete"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexCagra(const AscendIndexCagra&amp;) = delete; AscendIndexCagra&amp; operator=(const AscendIndexCagra&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Disables the copy constructor and copy assignment operator, making <code>AscendIndexCagra</code> non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## Init<a name="en-us_TOPIC_0000002513317665"></a>

<a name="table_cagra_init"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Init(int dim, int graphDegree, int dataNum, int topK, const std::vector&lt;int&gt;&amp; deviceList);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Initializes <code>AscendIndexCagra</code> and configures the vector dimensionality, graph degree, base library size, retrieval <code>topK</code> value, and device list.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dim</code>: Dimensionality of the feature vectors.<br><code>int graphDegree</code>: Degree of the neighbor graph, that is, the number of neighbors of each node.<br><code>int dataNum</code>: Number of feature vectors in the base library.<br><code>int topK</code>: Number of nearest neighbors returned by retrieval.<br><code>const std::vector&lt;int&gt;&amp; deviceList</code>: Device-side device ID list.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: 0 indicates success, and a non-zero value indicates failure.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {64, 128, 256, 512, 1024, 1536, 3072}.<br><code>graphDegree</code> ∈ {64, 128}.<br><code>graphDegree &lt; dataNum &lt;= 1e9</code>, subject to available host and device memory.<br><code>topK</code> ∈ {32,64,128,256,512,1024}, topK≤dataNum.<br><code>deviceList</code> currently supports only one device.<br>Before calling other APIs, you must call <code>Init</code> for initialization.</td></tr>
</tbody></table>

## Add<a name="en-us_TOPIC_0000002513317666"></a>

<a name="table_cagra_add"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds the neighbor graph and base library feature data to the <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const uint32_t* graph</code>: Neighbor graph data. The size is <code>dataNum * graph_degree</code>.<br><code>const uint32_t* hash</code>: Retained for compatibility and required to be non-null. Its contents are not read. The SDK allocates visited state internally, and the kernel clears it for each query. This parameter cannot supply previsited nodes or filtering conditions.<br><code>const float* data</code>: Base library feature vector data. The size is <code>dataNum * dim</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: 0 indicates success, and a non-zero value indicates failure.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling <code>Add</code>, you must call <code>Init</code>.</td></tr>
</tbody></table>

### Device memory requirements

Let N=dataNum, D=dim and G=graphDegree. Major persistent device buffers require approximately `4ND` bytes for FP32 vectors, `4NG` bytes for the graph, `N*(6*ceil(D/8)+16)` bytes for base quantization codes, and `64*ceil(N/32)*4` bytes for visited state. Train/Add create the base codes in unified mode; legacy mode creates them in QuantizeData.

Visited state requires about 8 MB at N=1,000,000 and 8 GB (7.45 GiB) at N=1,000,000,000. The numeric Init limit does not guarantee that the dataset fits. Choose the dataset size using available memory, including query buffers, workspace and temporary peak allocations.

## QuantizeData<a name="en-us_TOPIC_0000002513317667"></a>

<a name="table_cagra_quantize"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR QuantizeData(int n, const float* queryData, int ntotal, const float* baseData);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Quantizes and encodes the query vectors and base library vectors, including random orthogonal transformation, centroid calculation, and bit quantization. The quantized data is used for subsequent retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of query vectors.<br><code>const float* queryData</code>: Query feature vector data. The size is <code>n * dim</code>.<br><code>int ntotal</code>: Number of base library vectors.<br><code>const float* baseData</code>: Base library feature vector data. The size is <code>ntotal * dim</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: 0 indicates success, and a non-zero value indicates failure.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling <code>QuantizeData</code>, you must call <code>Init</code> and <code>Add</code>. <code>n</code> ∈ (0, 4096].</td></tr>
</tbody></table>

## Search<a name="en-us_TOPIC_0000002513317668"></a>

<a name="table_cagra_search"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs approximate nearest neighbor retrieval and returns the distances and labels of the <code>topK</code> nearest neighbors of each query vector.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int n</code>: Number of query vectors.<br><code>const float* queryData</code>: Query feature vector data. The size is <code>n * dim</code>.<br><code>int topK</code>: Number of nearest neighbors returned by retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float* dists</code>: Retrieval result distances. The size is <code>n * topK</code>, arranged in query order.<br><code>uint32_t* labels</code>: Retrieval result labels. The size is <code>n * topK</code>, arranged in query order.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: 0 indicates success, and a non-zero value indicates failure.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling <code>Search</code>, you must call <code>Init</code>, <code>Add</code>, and <code>QuantizeData</code>.<br><code>topK</code> must be the same as the <code>topK</code> set in <code>Init</code>.<br><code>n</code> ∈ (0, 4096].<br><code>topK</code> ∈ {32,64,128,256,512,1024}, topK≤dataNum.</td></tr>
</tbody></table>

TopK supports exactly {32,64,128,256,512,1024} and must not exceed dataNum. Search uses RaBitQ distances without FP32 reranking. Internal capacities 64/128/256 use bitonic sorting, and 512/1024 use radix selection and sorting. Legacy Init fixes TopK; unified Init allows a different k on each Search. Matching static OM files are required in both modes. Rebuild the Host library, reinstall the operator package and regenerate search OM files after updating.
