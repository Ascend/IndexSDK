# `AscendIndexCagra`<a name="en-us_TOPIC_0000002513157730"></a>

## Overview<a name="en-us_TOPIC_0000002544797645"></a>

`AscendIndexCagra` is a graph retrieval algorithm based on Cagra. It constructs a neighbor graph to achieve efficient approximate nearest neighbor search.

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
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {64, 128, 256, 512}.<br><code>graphDegree</code> ∈ {64, 128, 256, 512}.<br><code>dataNum</code> ∈ (0, 1e9].<br><code>topK</code> ∈ (0, 4096].<br><code>deviceList</code> currently supports only one device.<br>Before calling other APIs, you must call <code>Init</code> for initialization.</td></tr>
</tbody></table>

## Add<a name="en-us_TOPIC_0000002513317666"></a>

<a name="table_cagra_add"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds the neighbor graph, hash table, and base library feature data to the <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const uint32_t* graph</code>: Neighbor graph data. The size is <code>dataNum * graph_degree</code>.<br><code>const uint32_t* hash</code>: Hash table data, used to mark accesses during retrieval. The size is <code>dataNum * 2</code>.<br><code>const float* data</code>: Base library feature vector data. The size is <code>dataNum * dim</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: 0 indicates success, and a non-zero value indicates failure.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling <code>Add</code>, you must call <code>Init</code>.</td></tr>
</tbody></table>

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
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Before calling <code>Search</code>, you must call <code>Init</code>, <code>Add</code>, and <code>QuantizeData</code>.<br><code>topK</code> must be the same as the <code>topK</code> set in <code>Init</code>.<br><code>n</code> ∈ (0, 4096].<br><code>topK</code> ∈ (0, 4096].</td></tr>
</tbody></table>
