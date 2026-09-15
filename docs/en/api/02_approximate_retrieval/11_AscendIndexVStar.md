# `AscendIndexVStar`<a name="en-us_TOPIC_0000002044351677"></a>

## Overview<a name="en-us_TOPIC_0000002044510693"></a>

Ascend's self-developed vector retrieval algorithm provides approximate retrieval for high-dimensional large base libraries on the Ascend side. It uses a self-developed matrix approximation strategy to compress feature vectors before storing them in the base library, and then uses a self-developed retrieval strategy to retrieve the top `K` most similar vectors from the base library.

The vectors stored in the base library and the query vectors passed to each API must be normalized `float` values.

This algorithm does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you must acquire a lock before use. Otherwise, the retrieval API may fail. It also does not support sharing one Device across threads.

This algorithm is mainly designed for approximate fuzzy search in large base library scenarios, and its accuracy is lower than brute-force retrieval. In small base library scenarios, you are advised to increase the hyperparameter values appropriately to reduce the loss in accuracy.

## `AscendIndexVStar`<a name="en-us_TOPIC_0000002044513265"></a>

> [!NOTE]
>
>- When you create an `Index` instance, set `params.dim` according to the actual situation.
>- `params.subSpaceDim` and `params.nlist` should match the corresponding parameters used for codebook training.

<a name="table13851535141118"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>explicit AscendIndexVStar(const AscendIndexVstarInitParams&amp; params);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexVStar</code>. It creates an <code>Index</code> with the corresponding dimension based on the values configured in <code>params</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVstarInitParams&amp; params</code>: The constructor parameters. For details, see <code>AscendIndexVstarInitParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">For details, see <code>AscendIndexVstarInitParams</code>.</td></tr>
</tbody></table>

<a name="table11631734281"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVStar(const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexVStar</code>. It creates an <code>Index</code> with an unknown input data dimension and unknown hyperparameters based on <code>deviceList</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;int&gt;&amp; deviceList</code>: Device-side device IDs.<br><code>bool verbose</code>: Specifies whether to enable the <code>verbose</code> option. When enabled, some operations provide additional print prompts. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>deviceList</code> must contain valid device IDs. Currently, only one device is supported. After you create an <code>Index</code> instance with this constructor, you must first call <code>LoadIndex</code> to load the pre-saved <code>Index</code> instance from disk, and then you can perform other operations.</td></tr>
</tbody></table>

<a name="table8937623141615"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVStar(const AscendIndexVStar&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares this copy constructor as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVStar&amp;</code>: An <code>AscendIndexVStar</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `LoadIndex`<a name="en-us_TOPIC_0000002008232688"></a>

<a name="table950712481817"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; indexPath, AscendIndexVStar* indexVStar = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads an existing index from disk into the Device.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; indexPath</code>: The data file path.<br><code>AscendIndexVStar* indexVStar</code>: Used only in the <code>MultiSearch</code> scenario so that all <code>Index</code> instances share the codebook of the first <code>Index</code> instance.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the directory that contains <code>indexPath</code> exists and that the user who runs the process has read permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links. <code>indexVStar</code> must not be a null pointer in the <code>MultiSearch</code> scenario. It must be a null pointer in the single-<code>Index</code> scenario. If a valid <code>Index</code> pointer is used in the single-<code>Index</code> scenario, the original <code>Index</code> codebook is replaced by the codebook of the parameter <code>Index</code> instance.</td></tr>
</tbody></table>

## `WriteIndex`<a name="en-us_TOPIC_0000002044351681"></a>

<a name="table29774016915"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the index to disk.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; indexPath</code>: The file path where the data is saved.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the directory that contains <code>indexPath</code> exists and that the user who runs the process has write permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links. If the file already exists, it is overwritten. In this case, the user who runs the process must be the owner of the file.</td></tr>
</tbody></table>

## `AddCodeBooksByIndex`<a name="en-us_TOPIC_0000002044510697"></a>

<a name="table81089131197"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooksByIndex(AscendIndexVStar&amp; indexVStar);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">In a multi-<code>Index</code> retrieval scenario, this API loads the codebook of the input <code>Index</code> instance into the current <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>AscendIndexVStar&amp; indexVStar</code>: An <code>Index</code> instance with the codebook already populated.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API is used only in the <code>MultiSearch</code> scenario.</td></tr>
</tbody></table>

## `AddCodeBooksByPath`<a name="en-us_TOPIC_0000002008390980"></a>

<a name="table1523424814919"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooksByPath(const std::string&amp; codeBooksPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads a codebook into the current <code>Index</code> from the codebook path.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; codeBooksPath</code>: The codebook data file path.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the directory that contains <code>codeBooksPath</code> exists and that the user who runs the process has read permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links.</td></tr>
</tbody></table>

## `Add`<a name="en-us_TOPIC_0000002008232692"></a>

<a name="table18288921121213"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseData);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Builds the <code>AscendIndexVStar</code> base library and adds new feature vectors to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseData</code>: The feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of <code>baseData</code> must be <code>n * dim</code>, where <code>n</code> is the number of vectors to add to the base library and <code>dim</code> is the vector dimension. <code>n</code> must be in the range [10000, 1e8].<br>This API does not set IDs. The default ID range of the base library is [<code>ntotal</code>, <code>ntotal</code> + <code>n</code>), where <code>ntotal</code> is the number of vectors already in the <code>Index</code>, and <code>n</code> is the number of vectors to add to the base library.</td></tr>
</tbody></table>

> [!NOTE]
>
>- The `Add` API cannot be used together with the `AddWithIds` API.
>- After you use the `Add` API, the labels in the `Search` results may be duplicated. If your business logic requires labels, you are advised to use the [AddWithIds API](#addwithids).

## `AddWithIds`<a name="en-us_TOPIC_0000002044351685"></a>

<a name="table32483414124"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddWithIds(const std::vector&lt;float&gt;&amp; baseData, const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Builds the <code>AscendIndexVStar</code> base library and adds new feature vectors to the base library. This API allows the user to specify the IDs of the base library vectors to add.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseData</code>: The feature vectors to add to the base library.<br><code>const std::vector&lt;int64_t&gt;&amp; ids</code>: The array of IDs to map to the base library vectors to add.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of <code>baseData</code> must be <code>n * dim</code>, where <code>n</code> is the number of vectors to add to the base library and <code>dim</code> is the vector dimension. The length of <code>ids</code> must be <code>n</code>. Based on your own business scenario, ensure that <code>ids</code> are valid. If duplicate IDs exist in the base library, the <code>label</code> in the retrieval results cannot correspond to a specific base library vector. <code>n</code> must be in the range [10000, 1e8].</td></tr>
</tbody></table>

## `DeleteByIds`<a name="en-us_TOPIC_0000002044510701"></a>

<a name="table1284884631210"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteByIds(const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the vector data in the base library that corresponds to the IDs in the parameter array.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;int64_t&gt;&amp; ids</code>: The array of vector IDs to delete from the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The IDs in <code>ids</code> must be IDs used by the base library addition API.</td></tr>
</tbody></table>

## `DeleteById`<a name="en-us_TOPIC_0000002008390984"></a>

<a name="table9845165841212"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteById(int64_t id);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the vector data in the base library that corresponds to the parameter ID.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int64_t id</code>: The ID of the base library vector to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The ID must be an ID used by the base library addition API.</td></tr>
</tbody></table>

## `DeleteByRange`<a name="en-us_TOPIC_0000002008232696"></a>

<a name="table103969158136"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteByRange(int64_t startId, int64_t endId);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Deletes the vector data in the base library that corresponds to the ID range in the parameters.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int64_t startId</code>: The starting ID of the base library vectors to delete.<br><code>int64_t endId</code>: The ending ID of the base library vectors to delete.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The IDs to delete must be IDs used by the base library addition API, and the ID must be in the range [<code>startId</code>, <code>endId</code>].</td></tr>
</tbody></table>

## `Search`<a name="en-us_TOPIC_0000002044351689"></a>

<a name="table197566920146"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(const AscendIndexSearchParams&amp; params) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval and returns the IDs of the most similar <code>topK</code> features based on the input feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code>: The length must be greater than or equal to <code>n * topK</code>.</td></tr>
</tbody></table>

## `SearchWithMask`<a name="en-us_TOPIC_0000002044510705"></a>

<a name="table777072291418"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval and returns the IDs of the most similar <code>topK</code> features based on the input feature vectors. <code>mask</code> is a 0 and 1 bit string. Each bit indicates whether the corresponding feature in the base library participates in distance calculation. 0 means it does not participate, and 1 means it does.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.<br><code>const std::vector&lt;uint8_t&gt;&amp; mask</code>: The feature base library mask.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code>: The length must be greater than or equal to <code>n * topK</code>. <code>mask</code>: The length must be greater than or equal to <code>n * ceil(ntotal/8)</code>, where <code>ntotal</code> is the number of base library features.</td></tr>
</tbody></table>

## `MultiSearch`<a name="en-us_TOPIC_0000002008390988"></a>

<a name="table158666394146"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR MultiSearch(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, bool merge) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval across multiple <code>AscendIndexVStar</code> libraries and returns the IDs and distances of the most similar <code>topK</code> features based on the input feature vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</code>: Multiple <code>Index</code> instances to search.<br><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.<br><code>bool merge</code>: Specifies whether to merge the retrieval results across multiple <code>Index</code> instances.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code> must meet the following requirements. When <code>merge = true</code>, the length must be greater than or equal to <code>n * topK</code>. When <code>merge = false</code>, the length must be greater than or equal to <code>indexes.size() * n * topK</code>. <code>indexes</code> must meet the following requirement: <code>0 &lt; indexes.size() ≤ 150</code>.</td></tr>
</tbody></table>

## `MultiSearchWithMask`<a name="en-us_TOPIC_0000002008232700"></a>

<a name="table141672058131413"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR MultiSearchWithMask(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask, bool merge);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs feature vector retrieval across multiple <code>AscendIndexVStar</code> libraries and returns the IDs and distances of the most similar <code>topK</code> features based on the input feature vectors. It also supports deciding whether the base library participates in distance calculation based on a <code>mask</code>. <code>mask</code> is a 0 and 1 bit string. Each bit indicates whether the corresponding feature in the base library participates in distance calculation. 0 means it does not participate, and 1 means it does.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</code>: Multiple <code>Index</code> instances to search.<br><code>const AscendIndexSearchParams&amp; params</code>: The retrieval parameters. For details, see <code>AscendIndexSearchParams</code>.<br><code>size_t n</code>: The number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature vector data.<br><code>int topK</code>: The number of most similar results to return.<br><code>const std::vector&lt;uint8_t&gt;&amp; mask</code>: The feature base library mask.<br><code>bool merge</code>: Specifies whether to merge the retrieval results across multiple <code>Index</code> instances.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: The distance values between the query vectors and the closest <code>topK</code> vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: The IDs of the closest <code>topK</code> vectors.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>n</code> ∈ (0, 10000]. Ensure that <code>n * dim * sizeof(float)</code> is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. <code>queryData</code>: The length must be greater than or equal to <code>n * dim</code>. <code>topK</code> ∈ (0, 4096]. <code>dists</code> and <code>labels</code> must meet the following requirements. When <code>merge = true</code>, the length must be greater than or equal to <code>n * topK</code>. When <code>merge = false</code>, the length must be greater than or equal to <code>indexes.size() * n * topK</code>. <code>mask</code>: The length must be greater than or equal to <code>n * ceil(ntotal_max/8)</code>, where <code>ntotal_max</code> is the number of base library features and is the maximum number of base library vectors among all <code>Index</code> instances. <code>indexes</code> must meet the following requirement: <code>0 &lt; indexes.size() ≤ 150</code>.</td></tr>
</tbody></table>

## `SetHyperSearchParams`<a name="en-us_TOPIC_0000002044351693"></a>

<a name="table4215111781514"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams&amp; params);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the hyperparameters used when an <code>AscendIndexVstar</code> instance performs retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVstarHyperParams&amp; params</code>: The retrieval hyperparameters. For details, see <code>AscendIndexVstarHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>nProbeL1</code> ∈ (16, <code>nListL1</code>], <code>nProbeL1 % 8 == 0</code>. <code>nProbeL2</code> ∈ (16, <code>nProbeL1</code> * <code>nList2</code>], <code>nProbeL2 % 8 == 0</code>. <code>l3SegmentNum</code> ∈ (100, 5000], <code>l3SegmentNum % 8 == 0</code>.</td></tr>
</tbody></table>

## `GetHyperSearchParams`<a name="en-us_TOPIC_0000002044510709"></a>

<a name="table5860202961515"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams&amp; params) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the hyperparameters used during vector retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>AscendIndexVstarHyperParams&amp; params</code>: The retrieval hyperparameters. For details, see <code>AscendIndexVstarHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `GetDim`<a name="en-us_TOPIC_0000002008390992"></a>

<a name="table6661184351519"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetDim(int&amp; dim) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the dimension used when the index is initialized.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>int&amp; dim</code>: The dimension of the <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `GetNTotal`<a name="en-us_TOPIC_0000002008232704"></a>

<a name="table1919613597154"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetNTotal(uint64_t&amp; ntotal) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the number of base library vectors in the current index.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>uint64_t&amp; ntotal</code>: The total number of base library vectors in the current <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `Reset`<a name="en-us_TOPIC_0000002044351697"></a>

<a name="table19794117167"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Reset();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Resets the index and clears the saved index data.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you reset the index, the parameters that the user provided when initializing the index are retained.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000002008390996"></a>

<a name="table3792193711620"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexVStar&amp; operator=(const AscendIndexVStar&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexVStar&amp;</code>: An <code>AscendIndexVStar</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
