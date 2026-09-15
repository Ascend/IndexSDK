# `AscendIndexGreat`<a name="en-us_TOPIC_0000002044829945"></a>

## Overview<a name="en-us_TOPIC_0000002008751966"></a>

This self-developed vector retrieval algorithm provides approximate retrieval for high-dimensional large base libraries on the Ascend side and the Kunpeng side. It uses a self-developed retrieval strategy to retrieve the top `K` most similar vectors from the base library.

The vectors stored in the base library and the query vectors passed to each API must be normalized `float` values.

This algorithm does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you must acquire a lock before use. Otherwise, the retrieval API may fail. It also does not support sharing one Device across threads.

This algorithm is mainly designed for approximate fuzzy search in large base library scenarios, and its accuracy is lower than brute-force retrieval. In small base library scenarios, you are advised to increase the hyperparameter values appropriately to reduce the loss in accuracy.

> [!NOTE]
>
>- When you create an `Index` instance, set `params.dim` according to the actual situation.
>- The `Index` has two algorithm modes: `KMode`, which uses only the Kunpeng-side algorithm, and `AKMode`, which uses the Ascend plus Kunpeng algorithm. In `AKMode`, you must generate the corresponding operators in advance.
>- Ensure that `subSpaceDim` and `nlist` match the corresponding parameters used for codebook training.

## `AscendIndexGreat`<a name="en-us_TOPIC_0000002044829953"></a>

<a name="table5404639201712"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat(const std::string&amp; mode, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::string&amp; mode</code>: Specifies the algorithm mode.<br><code>const std::vector&lt;int&gt;&amp; deviceList</code>: The specified NPU-side device IDs.<br><code>bool verbose</code>: Specifies whether to enable the <code>verbose</code> option. When enabled, some operations provide additional print prompts. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>mode</code> supports only <code>KMode</code> and <code>AKMode</code>. For <code>deviceList</code>, use the <code>npu-smi</code> command to query the corresponding NPU IDs. Only one device ID is supported. After you create an <code>Index</code> instance with this constructor, you must first call <code>LoadIndex</code> to load the pre-saved <code>Index</code> instance from disk, and then you can perform other operations.</td></tr>
</tbody></table>

<a name="table72261454131719"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>explicit AscendIndexGreat(const AscendIndexGreatInitParams&amp; kModeInitParams);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">Initialization parameters required by the <code>Index</code>, specifically <code>kModeInitParams</code>. For details, see <code>AscendIndexGreatInitParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">See the parameter descriptions and constraints in <code>AscendIndexGreatInitParams</code>.</td></tr>
</tbody></table>

<a name="table198261931819"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat(const AscendIndexVstarInitParams&amp; aModeInitParams, const AscendIndexGreatInitParams&amp; kModeInitParams);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">Initialization parameters required by the <code>Index</code>, specifically <code>aModeInitParams</code> and <code>kModeInitParams</code>. For details, see <code>AscendIndexVstarInitParams</code> and <code>AscendIndexGreatInitParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Refer to the parameter descriptions and constraints in <code>AscendIndexVstarInitParams</code> and <code>AscendIndexGreatInitParams</code>.<br>The <code>dim</code> values of <code>aModeInitParams</code> and <code>kModeInitParams</code> must be the same.</td></tr>
</tbody></table>

<a name="table32891532172215"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat(const AscendIndexGreat&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares this copy constructor as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexGreat&amp;</code>: A constant <code>AscendIndexGreat</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `~AscendIndexGreat`<a name="en-us_TOPIC_0000002013257524"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexGreat() = default;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">The destructor of <code>AscendIndexGreat</code>. It destroys the <code>AscendIndexGreat</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000002008751990"></a>

<a name="table39961720122213"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexGreat &amp;operator=(const AscendIndexGreat&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator as deleted. In other words, this is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexGreat&amp;</code>: A constant <code>AscendIndexGreat</code> object.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `Add`<a name="en-us_TOPIC_0000002044950953"></a>

<a name="table11133547191811"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseRawData);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexGreat</code> base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseRawData</code>: The feature vectors to add to the base library.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: The operation status. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of <code>baseRawData</code> must be <code>dim * nTotal</code>. <code>nTotal</code> is the number of vectors to add to the base library, and <code>dim</code> is the dimension of each vector. The valid range of the total number of base library vectors is <code>10000 ≤ nTotal ≤ 1e8</code>. This algorithm does not support adding data again after the base library has been added. The <code>Add</code> API cannot be used together with the <code>AddWithIds</code> API.</td></tr>
</tbody></table>

## `AddWithIds`<a name="en-us_TOPIC_0000002044829957"></a>

<a name="table2436200181918"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddWithIds (const std::vector&lt;float&gt;&amp; baseRawData, const std::vector&lt;int64_t&gt;&amp; ids);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the AscendIndexGreat base index. When features are added through <code>AddWithIds</code>, the default IDs for the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::vector&lt;float&gt;&amp; baseRawData</code>: Feature vectors to add to the base index.<br><code>const std::vector&lt;int64_t&gt;&amp; ids</code>: IDs of the feature vectors to add to the base index. IDs must be unique within the <code>Index</code> instance.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The length of the <code>baseRawData</code> array must be <code>dim * nTotal</code>. <code>nTotal</code> is the number of vectors to be added to the base index, and <code>dim</code> is the dimensionality of each vector. The total number of base vectors must satisfy <code>10000 ≤ nTotal ≤ 1e8</code>. The length of <code>ids</code> must be <code>nTotal</code>. Users must ensure the validity of <code>ids</code> according to their own business scenario. If duplicate IDs exist in the base index, the <code>label</code> in the search results cannot be mapped to a specific base vector. This algorithm does not support adding vectors after the base index has been built. The <code>AddWithIds</code> API cannot be used together with the <code>Add</code> API.</td></tr>
</tbody></table>

## `LoadIndex`<a name="en-us_TOPIC_0000002008751978"></a>

<a name="table17789162191912"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads the <code>Index</code> structure from disk, including compressed, dimension-reduced feature vectors and codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; indexPath</code>: Path to load the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The file corresponding to <code>indexPath</code> must be a persisted file generated by calling <code>WriteIndex</code>, and the running user must have read permission for it. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

<a name="table98570373191"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR LoadIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and the original data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; aModeIndexPath</code>: Path to load the AMode index.<br><code>const std::string&amp; kModeIndexPath</code>: Path to load the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The files corresponding to <code>aModeIndexPath</code> and <code>kModeIndexPath</code> must be the persisted files generated by calling <code>WriteIndex</code>, and the running user must have read permission for them. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

## `WriteIndex`<a name="en-us_TOPIC_0000002044950957"></a>

<a name="table84194504191"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; indexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>const std::string&amp; indexPath</code>: Path to write the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The user must ensure that the directory containing the <code>indexPath</code> file exists and that the running user has write permission for that directory. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

<a name="table14392122132014"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR WriteIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>const std::string&amp; aModeIndexPath</code>: Path to write the AMode index.<br><code>const std::string&amp; kModeIndexPath</code>: Path to write the KMode index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The user must ensure that the directories containing the <code>aModeIndexPath</code> and <code>kModeIndexPath</code> file paths exist and that the running user has write permission for those directories. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

## `AddCodeBooks`<a name="en-us_TOPIC_0000002008751982"></a>

<a name="table339181620207"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddCodeBooks(const std::string&amp; codeBooksPath);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Loads an already generated codebook into the <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const std::string&amp; codeBooksPath</code>: Path to the generated codebook.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">This API can only be used when initializing an index in <code>AKMode</code>.<br>The user must ensure that the directory containing the <code>codeBooksPath</code> file exists, and the file content must be a valid codebook. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</td></tr>
</tbody></table>

## `Search`<a name="en-us_TOPIC_0000002008910274"></a>

<a name="table537563852013"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(const AscendIndexSearchParams&amp; searchParams);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implements the AscendIndexGreat feature-vector search API. Based on the input feature vectors, it returns the distances and IDs of the most similar <code>topK</code> features.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">For the <code>searchParams</code> structure, see the <code>AscendIndexSearchParams</code> API.<br><code>size_t n</code>: Number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature-vector data.<br><code>int topK</code>: Number of most similar results to return.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>topK</code> ∈ (0, 4096]. <code>n</code> ∈ (0, 10000]. <code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>. <code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. <code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</td></tr>
</tbody></table>

## `SearchWithMask`<a name="en-us_TOPIC_0000002044950961"></a>

<a name="table186956182018"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; searchParams, const std::vector&lt;uint8_t&gt;&amp; mask);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Implements the AscendIndexGreat feature-vector search API. Based on the input feature vectors, it returns the distances and IDs of the most similar <code>topK</code> features. In addition, the user can input a <code>uint8</code> array to mask specific base-index IDs so that the feature vectors corresponding to those IDs are excluded from retrieval.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">For the <code>searchParams</code> structure, see the <code>AscendIndexSearchParams</code> API.<br><code>size_t n</code>: Number of query feature vectors.<br><code>std::vector&lt;float&gt;&amp; queryData</code>: Feature-vector data.<br><code>int topK</code>: Number of most similar results to return.<br><code>const std::vector&lt;uint8_t&gt;&amp; mask</code>: External filtering mask, in bits. 0 means the feature is filtered out; 1 means the feature is selected.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>std::vector&lt;float&gt;&amp; dists</code>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.<br><code>std::vector&lt;int64_t&gt;&amp; labels</code>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>topK</code> ∈ (0, 4096]. <code>n</code> ∈ (0, 10000]. <code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>. <code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. <code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>. The total amount of data pointed to by <code>mask</code> must be greater than or equal to <code>n * ceil(nTotal / 8)</code>.</td></tr>
</tbody></table>

## `GetNTotal`<a name="en-us_TOPIC_0000002044829965"></a>

<a name="table971712872115"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetNTotal (uint64_t&amp; nTotal) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the number of feature vectors that have been added to the AscendIndexGreat base index.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>uint64_t&amp; nTotal</code>: Number of feature vectors added to the base index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `GetDim`<a name="en-us_TOPIC_0000002008751986"></a>

<a name="table113422226216"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetDim(int&amp; dim) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the dimensionality of the feature vectors added to the AscendIndexGreat base index.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>int&amp; dim</code>: Dimensionality of the feature vectors added to the base index.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `Reset`<a name="en-us_TOPIC_0000002008910278"></a>

<a name="table1974793512118"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Reset();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Clears the data stored in this <code>Index</code>, including compressed, dimension-reduced feature vectors and codebook data, while retaining the parameters entered when the user initialized the index.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `SetHyperSearchParams`<a name="en-us_TOPIC_0000002044950965"></a>

<a name="table1011347192118"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams&amp; params);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Sets the hyperparameters used when searching this <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle"><code>const AscendIndexHyperParams&amp; params</code>: Search hyperparameters. For details, see <code>AscendIndexHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `GetHyperSearchParams`<a name="en-us_TOPIC_0000002400547905"></a>

<a name="table749915518225"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetHyperSearchParams(AscendIndexHyperParams&amp; params) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the search hyperparameters used when searching this <code>Index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle"><code>Input</code></td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>AscendIndexHyperParams&amp; params</code>: Search hyperparameters. For details, see <code>AscendIndexHyperParams</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
