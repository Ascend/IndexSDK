# `AscendIndexIVFSP`<a name="en-us_TOPIC_0000001635576081"></a>

## Function Description<a name="en-us_TOPIC_0000001635815481"></a>

The Ascend-native IVFSP retrieval algorithm uses an in-house matrix approximation strategy to compress feature vectors before storing them in the base library. It then uses an in-house inverted-list strategy to select the base-library entries most likely to contain the ground truth. Finally, it uses an in-house retrieval strategy on the filtered base library to obtain the top K vector results.

`AscendIndexIVFSP` supports only standard mode scenarios and <term>Atlas Inference Series products</term>.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `add`<a name="en-us_TOPIC_0000001585895568"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void add(idx_t n, const float *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Adds feature vectors to the base library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1794683517262"><a name="p1794683517262"></a><a name="p1794683517262"></a><strong id="b016584171514"><a name="b016584171514"></a><a name="b016584171514"></a><code>idx_t n</code></strong>: Number of feature vectors to add to the base library.</p>
<p id="p1594616353266"><a name="p1594616353266"></a><a name="p1594616353266"></a><strong id="b12572953101710"><a name="b12572953101710"></a><a name="b12572953101710"></a><code>const float *x</code></strong>: Feature vectors to add to the base library.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1572065617233"></a><a name="ul1572065617233"></a><ul id="ul1572065617233"><li>The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>The total number of base-library vectors, <code>n</code>, is usually greater than 0 and less than <code>1e9</code>.</li><li>The amount of data added at one time must be smaller than or equal to the base-library data size.</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
> - The `add` API cannot be used together with the `add_with_ids` API.
> - After you use the `add` API, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` API.
> - The `add` API is optimized for small-batch addition scenarios. In this scenario, accuracy may decrease depending on the dataset. You are advised to use small-batch addition when a base library already exists.

## `add_with_ids`<a name="en-us_TOPIC_0000001586055512"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Adds feature vectors to the base library and specifies the corresponding IDs.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p133861411291"><a name="p133861411291"></a><a name="p133861411291"></a><strong id="b1195072611184"><a name="b1195072611184"></a><a name="b1195072611184"></a><code>idx_t n</code></strong>: Number of feature vectors to add to the base library.</p>
<p id="p338634162915"><a name="p338634162915"></a><a name="p338634162915"></a><strong id="b551412342183"><a name="b551412342183"></a><a name="b551412342183"></a><code>const float *x</code></strong>: Feature vectors to add to the base library.</p>
<p id="p17386184182913"><a name="p17386184182913"></a><a name="p17386184182913"></a><strong id="b42592374181"><a name="b42592374181"></a><a name="b42592374181"></a><code>const idx_t *ids</code></strong>: IDs of the feature vectors to add to the base library.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p152321022122915"><a name="p152321022122915"></a><a name="p152321022122915"></a>The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. You need to ensure that <code>ids</code> is valid according to your service scenario. If duplicate IDs exist in the base library, the <code>label</code> in the retrieval results cannot be mapped to a specific base-library vector.</p>
<p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>The value range of <code>n</code> is <code>0 &lt; n &lt; 1e9</code>.</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
> The `add_with_ids` API is optimized for small-batch addition scenarios. In this scenario, accuracy may decrease depending on the dataset. You are advised to use small-batch addition when a base library already exists.

## `AscendIndexIVFSP`<a name="en-us_TOPIC_0000001585736168"></a>

> [!NOTE]
>
> - Before you pass parameter `config` to the function, set the values of `conf.handleBatch`, `conf.nprobe`, and `conf.searchListSize` according to the actual situation. For field descriptions, see <a href="./06_AscendIndexIVFSPConfig.md#en-us_TOPIC_0000001635696057">Common Parameters</a>.
> - The values of `conf.handleBatch` and `conf.searchListSize` must be consistent with the `nprobe handle batch` and `search list size` values used when generating the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> service operator model file.
> - `conf.filterable` (inherited from <a href="../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig">`AscendIndexConfig`</a>) is `false` by default. If you want to use the `search_with_filter()` API, set **`conf.filterable = true`**. Setting `conf.filterable` to `true` stores extra information on the NPU card and consumes more NPU-side memory.

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Constructor of <code>AscendIndexIVFSP</code>. It sets Device-side resources based on the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1561263817143"><a name="p1561263817143"></a><a name="p1561263817143"></a><strong id="b19612183881419"><a name="b19612183881419"></a><a name="b19612183881419"></a><code>int dims</code></strong>: Dimension of a set of feature vectors managed by <code>AscendIndexIVFSP</code>.</p>
<p id="p1661213891412"><a name="p1661213891412"></a><a name="p1661213891412"></a><strong id="b48633716167"><a name="b48633716167"></a><a name="b48633716167"></a><code>int nonzeroNum</code></strong>: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.</p>
<p id="p17612133810143"><a name="p17612133810143"></a><a name="p17612133810143"></a><strong id="b761216381145"><a name="b761216381145"></a><a name="b761216381145"></a><code>int nlist</code></strong>: Number of clustering centers. This corresponds to the value of the <code>&lt;centroid num&gt;</code> parameter in the <a href="../../05_user_guide.md#ivfsp">IVFSP service operator model file generation</a>.</p>
<p id="p166121738111411"><a name="p166121738111411"></a><a name="p166121738111411"></a><strong id="b249718368317"><a name="b249718368317"></a><a name="b249718368317"></a><code>const char *codeBookPath</code></strong>: Path of the codebook file used by IVFSP.</p>
<p id="p1061210384146"><a name="p1061210384146"></a><a name="p1061210384146"></a><strong id="b451311429018"><a name="b451311429018"></a><a name="b451311429018"></a><code>faiss::ScalarQuantizer::QuantizerType qType</code></strong>: Scalar quantization type. The current supported value is only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</p>
<p id="p2617038181412"><a name="p2617038181412"></a><a name="p2617038181412"></a><strong id="b1861763817147"><a name="b1861763817147"></a><a name="b1861763817147"></a><code>faiss::MetricType metric</code></strong>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. The current <code>faiss::MetricType metric</code> supports only <code>METRIC_L2</code>.</p>
<p id="p1161723812145"><a name="p1161723812145"></a><a name="p1161723812145"></a><strong id="b11550924103117"><a name="b11550924103117"></a><a name="b11550924103117"></a><code>AscendIndexIVFSPConfig</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul719511175195"></a><a name="ul719511175195"></a><ul id="ul719511175195"><li>The values of <code>&lt;dim&gt;</code>, <code>&lt;nonzero num&gt;</code>, and <code>&lt;centroid num&gt;</code> used when training and generating the codebook must correspond to the <code>dims</code>, <code>nonzeroNum</code>, and <code>nlist</code> parameters of this function.</li><li>The codebook loaded from <code>codeBookPath</code> must correspond to the <code>dims</code>, <code>nonzeroNum</code>, and <code>nlist</code> parameters of this function, and the user who runs the program must be the owner of the codebook file. The codebook file cannot be a symbolic link.</li><li>When <code>dims</code> ∈ {64, 128, 256}, <code>nlist</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dims</code> ∈ {512, 768}, <code>nlist</code> ∈ {256, 512, 1024, 2048}.</li><li><code>nonzeroNum</code> must be a multiple of 16 and less than or equal to <code>min(128, dims)</code>.</li><li><code>metric</code> ∈ {faiss::MetricType::METRIC_L2}.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table49022324218"></a>
<table><tbody><tr id="row199021732102118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p79020325216"><a name="p79020325216"></a><a name="p79020325216"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7378142174712"><a name="p7378142174712"></a><a name="p7378142174712"></a><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &amp;codeBookSharedIdx, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></p>
</td>
</tr>
<tr id="row190216323214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p13902153202111"><a name="p13902153202111"></a><a name="p13902153202111"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1390393214211"><a name="p1390393214211"></a><a name="p1390393214211"></a>Constructor of <code>AscendIndexIVFSP</code>. It sets Device-side resources based on the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row3903113252110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p89039324212"><a name="p89039324212"></a><a name="p89039324212"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p790310324213"><a name="p790310324213"></a><a name="p790310324213"></a><strong id="b209032032182118"><a name="b209032032182118"></a><a name="b209032032182118"></a><code>int dims</code></strong>: Dimension of a set of feature vectors managed by <code>AscendIndexIVFSP</code>.</p>
<p id="p1590310322217"><a name="p1590310322217"></a><a name="p1590310322217"></a><strong id="b99031324210"><a name="b99031324210"></a><a name="b99031324210"></a><code>int nonzeroNum</code></strong>: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.</p>
<p id="p490373216218"><a name="p490373216218"></a><a name="p490373216218"></a><strong id="b390383218212"><a name="b390383218212"></a><a name="b390383218212"></a><code>int nlist</code></strong>: Number of clustering centers. This corresponds to the value of the <code>&lt;centroid num&gt;</code> parameter in the <a href="../../05_user_guide.md#ivfsp">IVFSP service operator model file generation</a>.</p>
<p id="p390313219218"><a name="p390313219218"></a><a name="p390313219218"></a><strong id="b116451015104820"><a name="b116451015104820"></a><a name="b116451015104820"></a><code>const AscendIndexIVFSP &amp;codeBookSharedIdx</code></strong>: <code>AscendIndexIVFSP</code> object that shares the codebook.</p>
<p id="p1990343252111"><a name="p1990343252111"></a><a name="p1990343252111"></a><strong id="b49034325211"><a name="b49034325211"></a><a name="b49034325211"></a><code>faiss::ScalarQuantizer::QuantizerType qType</code></strong>: Scalar quantization type. The current supported value is only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</p>
<p id="p14903132162110"><a name="p14903132162110"></a><a name="p14903132162110"></a><strong id="b119031132182114"><a name="b119031132182114"></a><a name="b119031132182114"></a><code>faiss::MetricType metric</code></strong>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. The current <code>faiss::MetricType metric</code> supports only <code>METRIC_L2</code>.</p>
<p id="p20903173272114"><a name="p20903173272114"></a><a name="p20903173272114"></a><strong id="b1490310325216"><a name="b1490310325216"></a><a name="b1490310325216"></a><code>AscendIndexIVFSPConfig</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row890313323211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1090314321219"><a name="p1090314321219"></a><a name="p1090314321219"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6903123222115"><a name="p6903123222115"></a><a name="p6903123222115"></a>None</p>
</td>
</tr>
<tr id="row190393211210"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2903532142118"><a name="p2903532142118"></a><a name="p2903532142118"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p18903163218211"><a name="p18903163218211"></a><a name="p18903163218211"></a>None</p>
</td>
</tr>
<tr id="row4903123214219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p39031232162110"><a name="p39031232162110"></a><a name="p39031232162110"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul790383216215"></a><a name="ul790383216215"></a><ul id="ul790383216215"><li>The values of <code>&lt;dim&gt;</code>, <code>&lt;nonzero num&gt;</code>, and <code>&lt;centroid num&gt;</code> used when training and generating the codebook must correspond to the <code>dims</code>, <code>nonzeroNum</code>, and <code>nlist</code> parameters of this function.</li><li>The shared codebook configuration of <code>codeBookSharedIdx</code> must match the codebook configuration of the current <code>Index</code>, and the Device resources must also match.</li><li>When <code>dims</code> ∈ {64, 128, 256}, <code>nlist</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dims</code> ∈ {512, 768}, <code>nlist</code> ∈ {256, 512, 1024, 2048}.</li><li><code>nonzeroNum</code> must be a multiple of 16 and less than or equal to <code>min(128, dims)</code>.</li><li><code>metric</code> ∈ {faiss::MetricType::METRIC_L2}.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table8581162710235"></a>
<table><tbody><tr id="row258119270238"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p758152711231"><a name="p758152711231"></a><a name="p758152711231"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p158117272235"><a name="p158117272235"></a><a name="p158117272235"></a><code>AscendIndexIVFSP (const AscendIndexIVFSP&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row6581192742313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p358110271235"><a name="p358110271235"></a><a name="p358110271235"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2581327192318"><a name="p2581327192318"></a><a name="p2581327192318"></a>Declares the copy constructor of this index as deleted. Therefore, it is a non-copyable type.</p>
</td>
</tr>
<tr id="row858114273233"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7581162715238"><a name="p7581162715238"></a><a name="p7581162715238"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b650433582416"><a name="b650433582416"></a><a name="b650433582416"></a><code>const AscendIndexIVFSP&amp;</code></strong>: Constant <code>AscendIndexIVFSP</code>.</p>
</td>
</tr>
<tr id="row5581152722313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p458117278231"><a name="p458117278231"></a><a name="p458117278231"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1258111270239"><a name="p1258111270239"></a><a name="p1258111270239"></a>None</p>
</td>
</tr>
<tr id="row4581162702318"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p15581202722317"><a name="p15581202722317"></a><a name="p15581202722317"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p3581127182311"><a name="p3581127182311"></a><a name="p3581127182311"></a>None</p>
</td>
</tr>
<tr id="row125811227162312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p45811227132319"><a name="p45811227132319"></a><a name="p45811227132319"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table186918413239"></a>
<table><tbody><tr id="row1386916412234"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p78691441132310"><a name="p78691441132310"></a><a name="p78691441132310"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1386914415234"><a name="p1386914415234"></a><a name="p1386914415234"></a><code>virtual ~AscendIndexIVFSP();</code></p>
</td>
</tr>
<tr id="row686920419239"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1886910414237"><a name="p1886910414237"></a><a name="p1886910414237"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p178691241122313"><a name="p178691241122313"></a><a name="p178691241122313"></a>Destructor of <code>AscendIndexIVFSP</code>. It destroys the <code>AscendIndexIVFSP</code> object and releases resources.</p>
</td>
</tr>
<tr id="row28695418235"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128698415234"><a name="p128698415234"></a><a name="p128698415234"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p086914116238"><a name="p086914116238"></a><a name="p086914116238"></a>None</p>
</td>
</tr>
<tr id="row19869641142315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18691641192314"><a name="p18691641192314"></a><a name="p18691641192314"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2869204132310"><a name="p2869204132310"></a><a name="p2869204132310"></a>None</p>
</td>
</tr>
<tr id="row6869134122317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p16869114115232"><a name="p16869114115232"></a><a name="p16869114115232"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p9869164162316"><a name="p9869164162316"></a><a name="p9869164162316"></a>None</p>
</td>
</tr>
<tr id="row3869841102310"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19869441112317"><a name="p19869441112317"></a><a name="p19869441112317"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p186914112231"><a name="p186914112231"></a><a name="p186914112231"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table241282321712"></a>
<table><tbody><tr id="row1441202301711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1541222315179"><a name="p1541222315179"></a><a name="p1541222315179"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15204727163415"><a name="p15204727163415"></a><a name="p15204727163415"></a><code>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</code></p>
</td>
</tr>
<tr id="row84121238175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3412122315172"><a name="p3412122315172"></a><a name="p3412122315172"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p56033285334"><a name="p56033285334"></a><a name="p56033285334"></a>Constructor of <code>AscendIndexIVFSP</code>. It sets Device-side resources based on the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row164121237173"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p164121723181717"><a name="p164121723181717"></a><a name="p164121723181717"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul534674763310"></a><a name="ul534674763310"></a><ul id="ul534674763310"><li><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexIVFSP</code>.</li><li><code>int nonzeroNum</code>: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.</li><li><code>int nlist</code>: Number of clustering centers. This corresponds to the value of the <code>&lt;centroid num&gt;</code> parameter in the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> service operator model file generation section.</li><li><code>faiss::ScalarQuantizer::QuantizerType qType</code>: Scalar quantization type. The current supported value is only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</li><li><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. The current <code>faiss::MetricType metric</code> supports only <code>METRIC_L2</code>.</li><li><code>AscendIndexIVFSPConfig</code>: Device-side resource configuration.</li></ul>
</td>
</tr>
<tr id="row6413192313178"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p6413142341717"><a name="p6413142341717"></a><a name="p6413142341717"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p19203142716347"><a name="p19203142716347"></a><a name="p19203142716347"></a>None</p>
</td>
</tr>
<tr id="row341316237178"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p24131423201720"><a name="p24131423201720"></a><a name="p24131423201720"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p6203427123419"><a name="p6203427123419"></a><a name="p6203427123419"></a>None</p>
</td>
</tr>
<tr id="row7413102361711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p6413323131715"><a name="p6413323131715"></a><a name="p6413323131715"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul162613915340"></a><a name="ul162613915340"></a><ul id="ul162613915340"><li>When <code>dims</code> ∈ {64, 128, 256}, <code>nlist</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dims</code> ∈ {512, 768}, <code>nlist</code> ∈ {256, 512, 1024, 2048}.</li><li><code>nonzeroNum</code> must be a multiple of 16 and less than or equal to <code>min(128, dims)</code>.</li><li><code>metric</code> ∈ {faiss::MetricType::METRIC_L2}.</li></ul>
</td>
</tr>
</tbody>
</table>

## `loadAllData`<a id="en-us_TOPIC_0000001585736172"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void loadAllData(const char *dataPath);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Loads the <code>Index</code> structure from disk into the Device, including the compressed, reduced-dimensional feature vectors and the codebook data.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b102253712317"><a name="b102253712317"></a><a name="b102253712317"></a><code>const char *dataPath</code></strong>: Path to the data file.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>The file corresponding to <code>dataPath</code> should be the file written by <code>saveAllData</code>, and the process user must have read permission for it. The file must not be a symbolic link.</p>
<p id="p1430141710323"><a name="p1430141710323"></a><a name="p1430141710323"></a>This API does not support codebook sharing. If you need codebook sharing, you are advised to use the <code>loadAllData</code> overload that accepts <code>codeBookSharedIdx</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table115591219131513"></a>
<table><tbody><tr id="row1955918198153"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p255921981517"><a name="p255921981517"></a><a name="p255921981517"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p390319328341"><a name="p390319328341"></a><a name="p390319328341"></a><code>static std::shared_ptr&lt;AscendIndexIVFSP&gt; loadAllData(const AscendIndexIVFSPConfig &amp;config, const uint8_t *data, size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx = nullptr);</code></p>
</td>
</tr>
<tr id="row10559191931517"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1559111916158"><a name="p1559111916158"></a><a name="p1559111916158"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p144174125351"><a name="p144174125351"></a><a name="p144174125351"></a>Restores an <code>AscendIndexIVFSP</code> object from memory.</p>
</td>
</tr>
<tr id="row4559219161516"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5559519101517"><a name="p5559519101517"></a><a name="p5559519101517"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul3105123212352"></a><a name="ul3105123212352"></a><ul id="ul3105123212352"><li><strong id="b3975154112358"><a name="b3975154112358"></a><a name="b3975154112358"></a><code>const AscendIndexIVFSPConfig &amp;config</code></strong>: Device-side resource configuration. Currently, you only need to set <code>config.deviceList</code> and <code>config.resourceSize</code>. The other configuration parameters are restored from memory.</li><li><strong id="b8319545173518"><a name="b8319545173518"></a><a name="b8319545173518"></a><code>const uint8_t *data</code></strong>: Memory pointer obtained by <code>saveAllData</code>.</li><li><strong id="b1312214484354"><a name="b1312214484354"></a><a name="b1312214484354"></a><code>size_t dataLen</code></strong>: Actual length of the <code>data</code> pointer.</li><li><strong id="b125458521353"><a name="b125458521353"></a><a name="b125458521353"></a><code>const AscendIndexIVFSP *codeBookSharedIdx</code></strong>: Pointer to the <code>AscendIndexIVFSP</code> that shares the codebook. The default value is <code>nullptr</code>, which means that the codebook is not shared.</li></ul>
</td>
</tr>
<tr id="row18559201914151"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1255961941516"><a name="p1255961941516"></a><a name="p1255961941516"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1555981921519"><a name="p1555981921519"></a><a name="p1555981921519"></a>None</p>
</td>
</tr>
<tr id="row855915191150"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6560131941520"><a name="p6560131941520"></a><a name="p6560131941520"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1671312145366"><a name="p1671312145366"></a><a name="p1671312145366"></a>A smart pointer to the <code>AscendIndexIVFSP</code> object restored from memory.</p>
</td>
</tr>
<tr id="row956019190157"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5560111920158"><a name="p5560111920158"></a><a name="p5560111920158"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4752731193612"></a><a name="ul4752731193612"></a><ul id="ul4752731193612"><li><code>data</code> must be a non-null valid pointer.</li><li><code>dataLen</code> must be the actual length of the <code>data</code> pointer. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>The codebook configuration of the shared <code>codeBookSharedIdx</code> must match the codebook configuration of the current <code>Index</code>, and the Device resources must also match.</li></ul>
</td>
</tr>
</tbody>
</table>

## `operator=`<a name="en-us_TOPIC_0000001635975413"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>AscendIndexIVFSP&amp; operator=(const AscendIndexIVFSP&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Declares this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b88405472511"><a name="b88405472511"></a><a name="b88405472511"></a><code>const AscendIndexIVFSP&amp;</code></strong>: Constant <code>AscendIndexIVFSP</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `remove_ids`<a name="en-us_TOPIC_0000001635576085"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Implements the API for deleting the specified feature vectors from the base library in <code>AscendIndexIVFSP</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b15391516143014"><a name="b15391516143014"></a><a name="b15391516143014"></a><code>const faiss::IDSelector &amp;sel</code></strong>: Feature vectors to delete. For details about usage and definition, see the relevant Faiss documentation.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>Returns the number of deleted feature vectors.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `reset`<a name="en-us_TOPIC_0000001635815485"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void reset() override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Clears the base-library vectors of this <code>AscendIndexIVFSP</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `saveAllData`<a name="en-us_TOPIC_0000001635696053"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void saveAllData(const char *dataPath);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Writes the <code>Index</code> structure from the Device side to disk. The data written to disk includes the compressed, reduced-dimensional feature vectors and the codebook data.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b19557108143418"><a name="b19557108143418"></a><a name="b19557108143418"></a><code>const char *dataPath</code></strong>: Path to the output data file.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>Ensure that the directory containing the <code>dataPath</code> file exists, and that the process user has write permission for the directory. For security hardening, the directory hierarchy must not contain symbolic links.</p>
<p id="p10274445174214"><a name="p10274445174214"></a><a name="p10274445174214"></a>When the file corresponding to <code>dataPath</code> already exists, the file is overwritten. In this case, the process user should be the file owner.</p>
</td>
</tr>
</tbody>
</table>

<a name="table11876949141314"></a>
<table><tbody><tr id="row12876549141317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p20876649191320"><a name="p20876649191320"></a><a name="p20876649191320"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7338202573713"><a name="p7338202573713"></a><a name="p7338202573713"></a><code>void saveAllData(uint8_t *&amp;data, size_t &amp;dataLen) const;</code></p>
</td>
</tr>
<tr id="row1587654912137"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p148761549201313"><a name="p148761549201313"></a><a name="p148761549201313"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6203133116375"><a name="p6203133116375"></a><a name="p6203133116375"></a>Stores the <code>AscendIndexIVFSP</code> object in memory.</p>
</td>
</tr>
<tr id="row17876184916136"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p38761849171311"><a name="p38761849171311"></a><a name="p38761849171311"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1333871953710"><a name="p1333871953710"></a><a name="p1333871953710"></a>None</p>
</td>
</tr>
<tr id="row7876174971312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p138768495136"><a name="p138768495136"></a><a name="p138768495136"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1142124363714"><a name="p1142124363714"></a><a name="p1142124363714"></a><strong id="b7190348133717"><a name="b7190348133717"></a><a name="b7190348133717"></a><code>uint8_t *&amp;data</code></strong>: Memory pointer used to store <code>AscendIndexIVFSP</code> data.</p>
<p id="p10142204317379"><a name="p10142204317379"></a><a name="p10142204317379"></a><strong id="b133501052103714"><a name="b133501052103714"></a><a name="b133501052103714"></a><code>size_t &amp;dataLen</code></strong>: Actual length of the <code>data</code> pointer.</p>
</td>
</tr>
<tr id="row487615490131"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3876949141317"><a name="p3876949141317"></a><a name="p3876949141317"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p233621903719"><a name="p233621903719"></a><a name="p233621903719"></a>None</p>
</td>
</tr>
<tr id="row987624981313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5876149181310"><a name="p5876149181310"></a><a name="p5876149181310"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1418092193810"><a name="p1418092193810"></a><a name="p1418092193810"></a>The input <code>data</code> must be a null pointer. After the API returns, the user must call <code>delete</code> to free the memory after using <code>data</code>. Otherwise, a memory leak occurs.</p>
</td>
</tr>
</tbody>
</table>

## `search`<a name="en-us_TOPIC_0000001635815489"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1587414619374"><a name="p1587414619374"></a><a name="p1587414619374"></a><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Implements the feature vector query API for <code>AscendIndexIVFSP</code>, returning the IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1246083018397"><a name="p1246083018397"></a><a name="p1246083018397"></a><strong id="b1246015305391"><a name="b1246015305391"></a><a name="b1246015305391"></a><code>idx_t n</code></strong>: Number of query feature vectors.</p>
<p id="p17460230163917"><a name="p17460230163917"></a><a name="p17460230163917"></a><strong id="b12460133016398"><a name="b12460133016398"></a><a name="b12460133016398"></a><code>const float *x</code></strong>: Feature vector data.</p>
<p id="p546073017399"><a name="p546073017399"></a><a name="p546073017399"></a><strong id="b5460630173912"><a name="b5460630173912"></a><a name="b5460630173912"></a><code>idx_t k</code></strong>: Number of most similar results to return.</p>
<p id="p13712185717441"><a name="p13712185717441"></a><a name="p13712185717441"></a><strong id="b15637734194512"><a name="b15637734194512"></a><a name="b15637734194512"></a><code>const SearchParameters *params</code></strong>: Optional Faiss parameter. The default value is <code>nullptr</code>, and this parameter is currently unsupported.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p9711201512411"><a name="p9711201512411"></a><a name="p9711201512411"></a><strong id="b127111415174119"><a name="b127111415174119"></a><a name="b127111415174119"></a><code>float *distances</code></strong>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid distances with <code>65504</code> or <code>-65504</code>.</p>
<p id="p4711515104119"><a name="p4711515104119"></a><a name="p4711515104119"></a><strong id="b571161514419"><a name="b571161514419"></a><a name="b571161514419"></a><code>idx_t *labels</code></strong>: IDs of the top <code>k</code> nearest vectors to the query. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid labels with <code>-1</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>The length of the query feature vector data <code>x</code> should be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The value range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed <code>4096</code>.</p>
</td>
</tr>
</tbody>
</table>

## `search_with_filter`<a name="en-us_TOPIC_0000001585736176"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Feature vector query API for <code>AscendIndexIVFSP</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It also provides CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array with a length of <code>n * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four numbers of each filter, which are 128 bits, represent the corresponding CID. The last two numbers represent the left-closed timestamp range, that is, <code>[x, y)</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b12607182024312"><a name="b12607182024312"></a><a name="b12607182024312"></a><code>idx_t n</code></strong>: Number of query feature vectors.</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><strong id="b960782054317"><a name="b960782054317"></a><a name="b960782054317"></a><code>const float *x</code></strong>: Feature vector data.</p>
<p id="p176073203438"><a name="p176073203438"></a><a name="p176073203438"></a><strong id="b9607820114310"><a name="b9607820114310"></a><a name="b9607820114310"></a><code>idx_t k</code></strong>: Number of most similar results to return.</p>
<p id="p76071120164313"><a name="p76071120164313"></a><a name="p76071120164313"></a><strong id="b16607122014317"><a name="b16607122014317"></a><a name="b16607122014317"></a><code>const void *filters</code></strong>: Filter conditions.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14659337439"><a name="p14659337439"></a><a name="p14659337439"></a><strong id="b96520338430"><a name="b96520338430"></a><a name="b96520338430"></a><code>float *distances</code></strong>: Distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b76513333436"><a name="b76513333436"></a><a name="b76513333436"></a><code>idx_t *labels</code></strong>: IDs of the top <code>k</code> nearest vectors to the query.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>The value range of <code>n</code> is <code>0 &lt; n &lt; 1e9</code>.</li><li><code>k</code> is usually not allowed to exceed <code>4096</code>.</li><li><code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers, and their lengths should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>filters</code> must be a non-null pointer to a <code>uint32_t</code> array of length <code>n * 6</code>. Otherwise, out-of-bounds reads may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `setNumProbes`<a name="en-us_TOPIC_0000001635576089"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void setNumProbes(int nprobes);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Sets the total number of candidate buckets used during search.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b16217144619214"><a name="b16217144619214"></a><a name="b16217144619214"></a><code>int nprobes</code></strong>: <code>nprobe</code> count of <code>AscendIndexIVFSP</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p24932033718"><a name="p24932033718"></a><a name="p24932033718"></a><code>nprobes</code> must be a multiple of 16 and satisfy <code>0 &lt; nprobes ≤ nlist</code>.</p>
</td>
</tr>
</tbody>
</table>

## `setVerbose`<a name="en-us_TOPIC_0000001586055516"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>void setVerbose(bool verbose);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Sets whether to print the progress of adding feature vectors to the base library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b14187193523611"><a name="b14187193523611"></a><a name="b14187193523611"></a><code>bool verbose</code></strong>: Whether to print the progress of adding feature vectors to the base library.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `trainCodeBook`<a name="en-us_TOPIC_0000002148530670"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p108681253358"><a name="p108681253358"></a><a name="p108681253358"></a><code>void trainCodeBook(const AscendIndexCodeBookInitParams &amp;codeBookInitParams) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5404143115354"><a name="p5404143115354"></a><a name="p5404143115354"></a>IVFSP codebook training API. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable <code>export OMP_NUM_THREADS=4</code> to speed it up.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12524203743519"><a name="p12524203743519"></a><a name="p12524203743519"></a><code>const AscendIndexCodeBookInitParams &amp;codeBookInitParams</code>: Initialization parameters required for codebook training.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184441341163411"><a name="p184441341163411"></a><a name="p184441341163411"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1944394111342"><a name="p1944394111342"></a><a name="p1944394111342"></a>See <a href="../02_approximate_retrieval/13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams">AscendIndexCodeBookInitParams</a>.</p>
</td>
</tr>
</tbody>
</table>

## `addCodeBook`<a name="en-us_TOPIC_0000002148372594"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p091033816364"><a name="p091033816364"></a><a name="p091033816364"></a><code>void addCodeBook(const char *codeBookPath);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16438944153611"><a name="p16438944153611"></a><a name="p16438944153611"></a>Adds a trained codebook.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p7964249173612"><a name="p7964249173612"></a><a name="p7964249173612"></a><code>const char *codeBookPath</code>: Codebook path.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184441341163411"><a name="p184441341163411"></a><a name="p184441341163411"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p198901214595"><a name="p198901214595"></a><a name="p198901214595"></a>The file corresponding to <code>codeBookPath</code> should be the codebook file produced by <code>trainCodeBook</code>, and the process user must have read permission for it. The file must not be a symbolic link.</p>
</td>
</tr>
</tbody>
</table>

## `AscendIndexCodeBookInitParams`<a name="en-us_TOPIC_0000002183731529"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4159142563912"><a name="p4159142563912"></a><a name="p4159142563912"></a><code>AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum, std::string codeBookOutputDir, std::string learnDataPath, bool verbose);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p123622413395"><a name="p123622413395"></a><a name="p123622413395"></a>Initialization structure for IVFSP codebook training.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1044544115344"><a name="p1044544115344"></a><a name="p1044544115344"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Parameter Values</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p12329155424114"><a name="p12329155424114"></a><a name="p12329155424114"></a><strong id="b1039615564575"><a name="b1039615564575"></a><a name="b1039615564575"></a><code>int numIter</code></strong>: Number of training iterations. The default value is <code>1</code>.</p>
<p id="p632945434111"><a name="p632945434111"></a><a name="p632945434111"></a><strong id="b4257125835714"><a name="b4257125835714"></a><a name="b4257125835714"></a><code>int device</code></strong>: Logical device ID. The default value is <code>0</code>.</p>
<p id="p1032911541415"><a name="p1032911541415"></a><a name="p1032911541415"></a><strong id="b4898195914578"><a name="b4898195914578"></a><a name="b4898195914578"></a><code>float ratio</code></strong>: Sampling rate of the original samples used for training. The default value is <code>1.0</code>.</p>
<p id="p15329175434116"><a name="p15329175434116"></a><a name="p15329175434116"></a><strong id="b166116165813"><a name="b166116165813"></a><a name="b166116165813"></a><code>int batchSize</code></strong>: Train with batches of size <code>batchSize</code>. This value must match <code>&lt;batch_size&gt;</code> in the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> training operator model file generation section. The default value is <code>32768</code>.</p>
<p id="p9329185464116"><a name="p9329185464116"></a><a name="p9329185464116"></a><strong id="b11743147185810"><a name="b11743147185810"></a><a name="b11743147185810"></a><code>int codeNum</code></strong>: Operate on at most <code>codeNum</code> samples at a time when updating the codebook. This value must be a power of two and must match <code>&lt;codebook_batch_size&gt;</code> in the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> training operator model file generation section. The default value is <code>32768</code>.</p>
<p id="p1232955415418"><a name="p1232955415418"></a><a name="p1232955415418"></a><strong id="b596299165819"><a name="b596299165819"></a><a name="b596299165819"></a><code>std::string codeBookOutputDir</code></strong>: Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.</p>
<p id="p163291154204114"><a name="p163291154204114"></a><a name="p163291154204114"></a><strong id="b1851871235817"><a name="b1851871235817"></a><a name="b1851871235817"></a><code>std::string learnDataPath</code></strong>: Path to the original feature file used for training. The file supports the bin and npy formats. For bin files, the storage order is row-major and the data type is <code>float32</code>.</p>
<p id="p103292545418"><a name="p103292545418"></a><a name="p103292545418"></a><strong id="b1032851425818"><a name="b1032851425818"></a><a name="b1032851425818"></a><code>bool verbose</code></strong>: Whether to enable additional output. The default value is <code>true</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1076619519437"></a><a name="ul1076619519437"></a><ul id="ul1076619519437"><li><code>numIter</code> ∈ (0, 20].</li><li><code>ratio</code> ∈ (0, 1.0].</li><li><code>batchSize</code> ∈ (0, 32768].</li><li><code>codeNum</code> ∈ (0, 32768].</li><li>When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner.</li><li>Before you run codebook training, refer to the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> training operator model file generation instructions.</li></ul>
</td>
</tr>
</tbody>
</table>

## `trainCodeBookFromMem`<a name="en-us_TOPIC_0000002257319034"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4398205434419"><a name="p4398205434419"></a><a name="p4398205434419"></a><code>void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>IVFSP codebook training API. Training data is loaded from memory. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable <code>export OMP_NUM_THREADS=4</code> to speed it up.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a><code>const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams</code>: Initialization parameters required for codebook training.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11397185416444"><a name="p11397185416444"></a><a name="p11397185416444"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p8383354134415"><a name="p8383354134415"></a><a name="p8383354134415"></a>For details about <code>AscendIndexCodeBookInitFromMemParams</code>, see <a href="#ascendindexcodebookinitfrommemparams"><code>AscendIndexCodeBookInitFromMemParams</code></a>.</p>
</td>
</tr>
</tbody>
</table>

## `AscendIndexCodeBookInitFromMemParams`<a name="en-us_TOPIC_0000002291969193"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7524719186"><a name="p7524719186"></a><a name="p7524719186"></a><code>AscendIndexCodeBookInitFromMemParams (int numIter, int device, float ratio, int batchSize, int codeNum,bool verbose,std::string codeBookOutputDir,const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>Initialization structure for IVFSP codebook training. Training data is loaded from memory.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Parameter Values</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><strong id="b3354181515167"><a name="b3354181515167"></a><a name="b3354181515167"></a><code>int numIter</code></strong>: Number of training iterations. The default value is <code>1</code>.</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><strong id="b13953183012163"><a name="b13953183012163"></a><a name="b13953183012163"></a><code>int device</code></strong>: Logical device ID. The default value is <code>0</code>.</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><strong id="b229253513169"><a name="b229253513169"></a><a name="b229253513169"></a><code>float ratio</code></strong>: Sampling rate of the original samples used for training. The default value is <code>1.0</code>.</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><strong id="b1712474051618"><a name="b1712474051618"></a><a name="b1712474051618"></a><code>int batchSize</code></strong>: Train with batches of size <code>batchSize</code>. This value must match <code>&lt;batch_size&gt;</code> in the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> training operator model file generation section. The value must be greater than 0. The default value is <code>32768</code>.</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><strong id="b76015463165"><a name="b76015463165"></a><a name="b76015463165"></a><code>int codeNum</code></strong>: Operate on at most <code>codeNum</code> samples at a time when updating the codebook. This value must be a power of two and must match <code>&lt;codebook_batch_size&gt;</code> in the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> training operator model file generation section. The value must be greater than 0. The default value is <code>32768</code>.</p>
<p id="p16645243598"><a name="p16645243598"></a><a name="p16645243598"></a><strong id="b16826155511614"><a name="b16826155511614"></a><a name="b16826155511614"></a><code>std::string codeBookOutputDir</code></strong>: Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.</p>
<p id="p864518435917"><a name="p864518435917"></a><a name="p864518435917"></a><strong id="b172846021716"><a name="b172846021716"></a><a name="b172846021716"></a><code>bool verbose</code></strong>: Whether to enable additional output. The default value is <code>true</code>.</p>
<p id="p146455433910"><a name="p146455433910"></a><a name="p146455433910"></a><strong id="b049205121720"><a name="b049205121720"></a><a name="b049205121720"></a><code>const float *memLearnData</code></strong>: Pointer to in-memory data. The default value is a null pointer.</p>
<p id="p1864544319911"><a name="p1864544319911"></a><a name="p1864544319911"></a><strong id="b3740189191719"><a name="b3740189191719"></a><a name="b3740189191719"></a><code>size_t memLearnDataSize</code></strong>: Length of the in-memory data. The default value is <code>0</code>.</p>
<p id="p106451943193"><a name="p106451943193"></a><a name="p106451943193"></a><strong id="b28410155178"><a name="b28410155178"></a><a name="b28410155178"></a><code>bool isTrainAndAdd</code></strong>: Whether to add the codebook directly to the <code>Index</code> after training. The default value is <code>false</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul129009408302"></a><a name="ul129009408302"></a><ul id="ul129009408302"><li><code>numIter</code> ∈ (0, 20]</li><li><code>ratio</code> ∈ (0, 1.0]</li><li><code>memLearnDataSize % dim == 0</code></li><li><code>memLearnDataSize ≤ 25G</code></li></ul>
<a name="ul154204603015"></a><a name="ul154204603015"></a><ul id="ul154204603015"><li>When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner.</li><li>Before you run codebook training, refer to the <a href="../../05_user_guide.md#ivfsp">IVFSP</a> training operator model file generation instructions.</li></ul>
<a name="ul547975410309"></a><a name="ul547975410309"></a><ul id="ul547975410309"><li>When <code>isTrainAndAdd</code> is <code>true</code>, the trained codebook is added directly to the <code>Index</code> and is not written to disk.</li><li>When <code>isTrainAndAdd</code> is <code>false</code>, the codebook is saved to <code>codeBookOutputDir</code>, and you must call <code>addCodeBook</code> manually.</li><li><code>memLearnDataSize</code> must be the actual length of the <code>memLearnData</code> pointer. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>
