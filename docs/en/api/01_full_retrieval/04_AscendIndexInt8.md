# `AscendIndexInt8`<a id="en-us_TOPIC_0000001506495841"></a>

## Overview<a id="en-us_TOPIC_0000001506495913"></a>

`AscendIndexInt8` is the base class of the indexes that use INT8 feature vectors in the feature retrieval component. It defines interfaces for other INT8 indexes in feature retrieval.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must lock before use, or the retrieval interface may raise exceptions. It also does not support sharing one device across different threads. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `add`<a name="en-us_TOPIC_0000001506334825"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const int8_t *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

<a name="table6211414109"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add(idx_t n, const char *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const char *x</code>: Feature vectors to add to the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

> [!NOTE]
>
>- The `add` interface cannot be used together with the `add_with_ids` interface.
>- After you use the `add` interface, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` interface.

## `add_with_ids`<a name="en-us_TOPIC_0000001506614905"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const int8_t *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library and specifies the feature IDs.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const int8_t *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library. The IDs must be unique within the <code>Index</code> instance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

<a name="table38814511704"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void add_with_ids(idx_t n, const char *x, const idx_t *ids);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds new feature vectors to the <code>AscendIndexInt8</code> base library and specifies the feature IDs.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors to add to the base library.<br><code>const char *x</code>: Feature vectors to add to the base library.<br><code>const idx_t *ids</code>: IDs corresponding to the feature vectors to add to the base library. The IDs must be unique within the <code>Index</code> instance.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</td></tr>
</tbody></table>

## `assign`<a name="en-us_TOPIC_0000001506495721"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Feature-vector retrieval interface of <code>AscendIndexInt8</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const int8_t *x</code>: Feature-vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the length of <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be greater than <code>0</code> and less than <code>1e9</code>. <code>k</code> must be greater than <code>0</code> and less than or equal to <code>4096</code>. <code>n * k</code> must be less than <code>1e10</code>.</td></tr>
</tbody></table>

## `AscendIndexInt8`<a name="en-us_TOPIC_0000001506614993"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config)`;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexInt8</code>. It creates an <code>AscendIndexInt8</code> with dimension <code>dims</code>. The dimension of the vector set managed by a single <code>Index</code> is unique. Device-side resources are set according to the values configured in <code>config</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexInt8</code>.<br><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndexInt8</code> when performing feature-vector similarity retrieval. Currently supported values are <code>faiss::MetricType::METRIC_L2</code> and <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.<br><code>AscendIndexInt8Config config</code>: Device-side resource configuration.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> must be an integer that is not smaller than 64 and not larger than 1024, and it must be divisible by 64.</td></tr>
</tbody></table>

<a name="table103312407520"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8(const AscendIndexInt8&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares this <code>Index</code> copy constructor as deleted. Therefore, the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8&amp;</code>: <code>AscendIndexInt8</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table1882220715614"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexInt8();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexInt8</code>. It destroys the <code>AscendIndexInt8</code> object and releases resources.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getDeviceList`<a name="en-us_TOPIC_0000001672982421"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::vector&lt;int&gt; getDeviceList() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Return the device-side Ascend AI Processor settings managed by <code>Index</code>. Subclasses inherit from it and implement it. This base class does not provide a corresponding implementation and returns only an empty <code>vector&lt;int&gt;</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The device-side Ascend AI Processor settings managed by <code>Index</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getDim`<a name="en-us_TOPIC_0000001690599922"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDim() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the dimension of the feature vector set managed by <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The dimension of the feature vector set managed by <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getNTotal`<a name="en-us_TOPIC_0000001738718517"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::idx_t getNTotal() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the number of feature vectors that <code>AscendIndexInt8</code> has added to the base vector set.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of feature vectors that <code>AscendIndexInt8</code> has added to the base vector set.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getMetricType`<a name="en-us_TOPIC_0000001738678653"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::MetricType getMetricType() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Get the distance metric type used by <code>AscendIndexInt8</code> when performing feature vector similarity retrieval.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The distance metric type used by <code>AscendIndexInt8</code> when performing feature vector similarity retrieval.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `isTrained`<a name="en-us_TOPIC_0000001690759666"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>bool isTrained() const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Determine whether <code>AscendIndexInt8</code> is trained.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The trained state of <code>AscendIndexInt8</code>. <code>true</code> means trained, and <code>false</code> means not trained.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `operator =`<a name="en-us_TOPIC_0000001506414841"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8&amp; operator=(const AscendIndexInt8&amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8&amp;</code>: A constant <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reclaimMemory`<a name="en-us_TOPIC_0000001506615133"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual size_t reclaimMemory();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `remove_ids`<a name="en-us_TOPIC_0000001456695088"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>size_t remove_ids(const faiss::IDSelector &amp;sel);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implement the interface for deleting the specified feature vectors from the base vector set in <code>AscendIndexInt8</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to delete. For details on usage and definition, see the corresponding Faiss documentation.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">The number of deleted feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `reserveMemory`<a name="en-us_TOPIC_0000001506615065"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void reserveMemory(size_t numVecs);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>size_t numVecs</code>: Number of base vectors for which to reserve memory.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `search`<a name="en-us_TOPIC_0000001506414889"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implement the feature vector search interface for <code>AscendIndexInt8</code>, and return the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const int8_t *x</code>: Feature vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid distances with <code>65504</code> or <code>-65504</code> depending on the metric.<br><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid labels with <code>-1</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of the query feature vector data <code>x</code> should be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, <code>n</code> is greater than <code>0</code> and less than <code>1e9</code>. Here, <code>k</code> is greater than <code>0</code> and less than or equal to <code>4096</code>.</td></tr>
</tbody></table>

<a name="table88671631181418"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Implement the feature vector search interface for <code>AscendIndexInt8</code>, and return the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of query feature vectors.<br><code>const char *x</code>: Feature vector data.<br><code>idx_t k</code>: Number of most similar results to return.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors.<br><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of the query feature vector data <code>x</code> should be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, <code>n</code> is greater than <code>0</code> and less than <code>1e9</code>. Here, <code>k</code> is greater than <code>0</code> and less than or equal to <code>4096</code>.</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000001456534956"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void train(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const int8_t *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `updateCentroids`<a name="en-us_TOPIC_0000001506414833"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void updateCentroids(idx_t n, const int8_t *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const int8_t *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table2023134918146"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void updateCentroids(idx_t n, const char *x);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">A virtual function defined in the base class. See the subclass for details.</td></tr>
<tr><td width="150" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const char *x</code>: Feature vector data.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
