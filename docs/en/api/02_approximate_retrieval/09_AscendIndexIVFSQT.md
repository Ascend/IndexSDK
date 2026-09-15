# `AscendIndexIVFSQT`<a name="en-us_TOPIC_0000001456375224"></a>

## Function Description<a name="en-us_TOPIC_0000001506615005"></a>

The `AscendIndexIVFSQT` class contains the three-stage retrieval `IVFSQ` algorithm with dimensionality reduction. You need to pass two parameters to specify the dimensions before and after dimensionality reduction, and the original dimension must be divisible by the reduced dimension. It is suitable for scenarios with a base library on the order of 10 million.

You need to generate the operators required for three-stage retrieval according to the IVFSQT operator generation method.

This type provides fuzzy clustering. Before bucket assignment, use the `threshold` parameter to control the degree of fuzziness. Set the `threshold` value according to the base library capacity and the available memory size. A `threshold` that is too large can cause insufficient memory and lead to failure. For <term>Atlas 200/300/500 inference product</term> environments, you are advised to set it to `[1.0, 1.1]`. For <term>Atlas Inference Series products</term>, you are advised to set it to `[1.0, 1.5]`. For search, you are advised to use **`batch size = 65536`**.

The workflow is: 1. Construct the `Index` object. 2. Train the data. 3. Add the data. 4. Update the data. 5. Search the data. 6. Destroy the `Index` object. After `update`, adding data is no longer supported. If you need to search new data, destroy the original `Index` object and use the workflow again from the beginning.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

## `AscendIndexIVFSQT`<a name="en-us_TOPIC_0000001506495685"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a><code>AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor for <code>AscendIndexIVFSQT</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b46201291626"><a name="b46201291626"></a><a name="b46201291626"></a><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>: CPU-side <code>Index</code> resource.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b159906314214"><a name="b159906314214"></a><a name="b159906314214"></a><code>AscendIndexIVFSQTConfig config</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1113653515219"></a><a name="ul1113653515219"></a><ul id="ul1113653515219"><li><code>index</code> must be a valid CPU <code>Index</code> pointer.</li><li><code>index-&gt;d</code> ∈ {256}.</li><li><code>index-&gt;sq.d</code> ∈ {32, 64, 128}.</li><li>The dimension of <code>index</code> must be greater than the dimension of <code>index-&gt;sq</code>, and it must be divisible by the dimension of <code>index-&gt;sq</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table124585216195"></a>
<table><tbody><tr id="row164575271917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p134518527191"><a name="p134518527191"></a><a name="p134518527191"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p18649496410"><a name="p18649496410"></a><a name="p18649496410"></a><code>AscendIndexIVFSQT(int dimIn, int dimOut, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</code></p>
</td>
</tr>
<tr id="row1045152101914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1245185213193"><a name="p1245185213193"></a><a name="p1245185213193"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1845115251915"><a name="p1845115251915"></a><a name="p1845115251915"></a>Constructor for <code>AscendIndexIVFSQT</code>. It creates an <code>AscendIndexIVFSQT</code>, and the Device-side resources are set according to the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row16451352141910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p204585213198"><a name="p204585213198"></a><a name="p204585213198"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b162991621040"><a name="b162991621040"></a><a name="b162991621040"></a><code>int dimIn</code></strong>: Dimension of the original feature vectors managed by <code>AscendIndexIVFSQT</code>.</p>
<p id="p055565814416"><a name="p055565814416"></a><a name="p055565814416"></a><strong id="b18128104346"><a name="b18128104346"></a><a name="b18128104346"></a><code>int dimOut</code></strong>: Dimension of the reduced feature vectors managed by <code>AscendIndexIVFSQT</code>.</p>
<p id="p169755411358"><a name="p169755411358"></a><a name="p169755411358"></a><strong id="b11908056418"><a name="b11908056418"></a><a name="b11908056418"></a><code>int nlist</code></strong>: Number of clustering centers. This corresponds to the <code>coarse_centroid_num</code> parameter in the operator generation script.</p>
<p id="p895114473339"><a name="p895114473339"></a><a name="p895114473339"></a><strong id="b5178209448"><a name="b5178209448"></a><a name="b5178209448"></a><code>faiss::ScalarQuantizer::QuantizerType qtype</code></strong>: Quantizer type of <code>AscendIndexIVFSQT</code>.</p>
<p id="p174585217192"><a name="p174585217192"></a><a name="p174585217192"></a><strong id="b27578121644"><a name="b27578121644"></a><a name="b27578121644"></a><code>faiss::MetricType metric</code></strong>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval.</p>
<p id="p345135213199"><a name="p345135213199"></a><a name="p345135213199"></a><strong id="b155089152412"><a name="b155089152412"></a><a name="b155089152412"></a><code>AscendIndexIVFSQTConfig config</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row19459527195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p94525214193"><a name="p94525214193"></a><a name="p94525214193"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p204520521198"><a name="p204520521198"></a><a name="p204520521198"></a>None</p>
</td>
</tr>
<tr id="row10451352191917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p44514527191"><a name="p44514527191"></a><a name="p44514527191"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p17451152151917"><a name="p17451152151917"></a><a name="p17451152151917"></a>None</p>
</td>
</tr>
<tr id="row154545211199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15456526194"><a name="p15456526194"></a><a name="p15456526194"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3942173318420"></a><a name="ul3942173318420"></a><ul id="ul3942173318420"><li><code>dimIn</code> ∈ {256}.</li><li><code>dimOut</code> ∈ {32, 64, 128}.</li><li><code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}.</li><li><code>qtype = ScalarQuantizer::QuantizerType::QT_8bit</code>. Only the <code>ScalarQuantizer::QuantizerType::QT_8bit</code> quantizer type is currently supported.</li><li><code>metric = faiss::MetricType::METRIC_INNER_PRODUCT</code>. Only <code>faiss::MetricType::METRIC_INNER_PRODUCT</code> is currently supported.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table68594118203"></a>
<table><tbody><tr id="row12859818205"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1885911122013"><a name="p1885911122013"></a><a name="p1885911122013"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p885213589106"><a name="p885213589106"></a><a name="p885213589106"></a><code>AscendIndexIVFSQT(const AscendIndexIVFSQT&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row158592122017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p68591616209"><a name="p68591616209"></a><a name="p68591616209"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p78592113207"><a name="p78592113207"></a><a name="p78592113207"></a>Declares the copy constructor of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row18859201122014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18859141122017"><a name="p18859141122017"></a><a name="p18859141122017"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b3710328459"><a name="b3710328459"></a><a name="b3710328459"></a><code>const AscendIndexIVFSQT&amp;</code></strong>: <code>AscendIndexIVFSQT</code> object.</p>
</td>
</tr>
<tr id="row28605142020"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p286001112012"><a name="p286001112012"></a><a name="p286001112012"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16860017208"><a name="p16860017208"></a><a name="p16860017208"></a>None</p>
</td>
</tr>
<tr id="row1186001132017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1586020118202"><a name="p1586020118202"></a><a name="p1586020118202"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p886015192016"><a name="p886015192016"></a><a name="p886015192016"></a>None</p>
</td>
</tr>
<tr id="row38604142015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1586012112011"><a name="p1586012112011"></a><a name="p1586012112011"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~AscendIndexIVFSQT`<a name="en-us_TOPIC_0000001456854984"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndexIVFSQT();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Destructor for <code>AscendIndexIVFSQT</code>. It destroys the <code>AscendIndexIVFSQT</code> object and releases resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `copyFrom`<a name="en-us_TOPIC_0000001456695060"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1215384082314"><a name="p1215384082314"></a><a name="p1215384082314"></a><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a><code>AscendIndexIVFSQT</code> copies an existing <code>index</code> to Ascend and retains the original Device-side resource configuration of <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b9345217864"><a name="b9345217864"></a><a name="b9345217864"></a><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer.</p>
<a name="ul1113653515219"></a><a name="ul1113653515219"></a><ul id="ul1113653515219"><li><code>index-&gt;d</code> ∈ {256}.</li><li><code>index-&gt;sq.d</code> ∈ {32, 64, 128}.</li><li>The dimension of <code>index</code> must be greater than the dimension of <code>index-&gt;sq</code>, and it must be divisible by the dimension of <code>index-&gt;sq</code>.</li><li>Do not call this API on an updated object.</li></ul>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000001506495825"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyTo(faiss::IndexIVFScalarQuantizer *index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1720318284418"><a name="p1720318284418"></a><a name="p1720318284418"></a>Copies the retrieval resources of <code>AscendIndexIVFSQT</code> to the CPU side.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b158351007610"><a name="b158351007610"></a><a name="b158351007610"></a><code>faiss::IndexIVFScalarQuantizer *index</code></strong>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The user is responsible for freeing the memory occupied by the <code>Index</code>.</p>
</td>
</tr>
</tbody>
</table>

## `fineTune`<a name="en-us_TOPIC_0000001456694860"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p193348812010"><a name="p193348812010"></a><a name="p193348812010"></a><code>void fineTune(size_t n, const float *x);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Fine-tunes and optimizes the centroids to avoid uneven bucket assignment.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p492321510014"><a name="p492321510014"></a><a name="p492321510014"></a><strong id="b101231056175619"><a name="b101231056175619"></a><a name="b101231056175619"></a><code>size_t n</code></strong>: Number of feature vectors.</p>
<p id="p12314314436"><a name="p12314314436"></a><a name="p12314314436"></a><strong id="b3709358115620"><a name="b3709358115620"></a><a name="b3709358115620"></a><code>const float *x</code></strong>: Feature vector data.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p1529214384393"><a name="p1529214384393"></a><a name="p1529214384393"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `getFuzzyK`<a name="en-us_TOPIC_0000001456855008"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a><code>int getFuzzyK() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>Gets the maximum value used when a vector is assigned to buckets.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b8278171308"><a name="b8278171308"></a><a name="b8278171308"></a><code>int</code></strong>: Maximum value used when a vector is assigned to buckets.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getListCodesAndIds`<a name="en-us_TOPIC_0000001687739112"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Returns the feature vectors and corresponding IDs for a specific <code>nlistId</code> in the current <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b234205219283"><a name="b234205219283"></a><a name="b234205219283"></a><code>int listId</code></strong>: Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p812472610226"><a name="p812472610226"></a><a name="p812472610226"></a><strong id="b8752144372820"><a name="b8752144372820"></a><a name="b8752144372820"></a><code>std::vector&lt;uint8_t&gt;&amp; codes</code></strong>: Feature vectors at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</p>
<p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b198817462287"><a name="b198817462287"></a><a name="b198817462287"></a><code>std::vector&lt;ascend_idx_t&gt;&amp; ids</code></strong>: Feature vector IDs at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p13621611141120"><a name="p13621611141120"></a><a name="p13621611141120"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `getListLength`<a name="en-us_TOPIC_0000001735977797"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>uint32_t getListLength(int listId) const override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Returns the length for a specific <code>nlistId</code> in the current <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b121461446192713"><a name="b121461446192713"></a><a name="b121461446192713"></a><code>int listId</code></strong>: Specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Length at the specific <code>nlistId</code> in the <code>nlist</code> of <code>AscendIndexIVFSQT</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p13621611141120"><a name="p13621611141120"></a><a name="p13621611141120"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `getLowerBound`<a name="en-us_TOPIC_0000001506614885"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p139751803263"><a name="p139751803263"></a><a name="p139751803263"></a><code>int getLowerBound() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>Returns the threshold for second-level clustering.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p192391322172611"><a name="p192391322172611"></a><a name="p192391322172611"></a>Threshold for second-level clustering.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getMergeThres`<a name="en-us_TOPIC_0000001506615073"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p898555319557"><a name="p898555319557"></a><a name="p898555319557"></a><code>int getMergeThres() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p476310541227"><a name="p476310541227"></a><a name="p476310541227"></a>Gets the threshold for merging sub-buckets.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>Threshold for merging sub-buckets.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getQMax`<a name="en-us_TOPIC_0000001456535208"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a><code>float getQMax() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>Returns the maximum feature vector value.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>Maximum feature vector value.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getQMin`<a name="en-us_TOPIC_0000001506615029"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p114441322513"><a name="p114441322513"></a><a name="p114441322513"></a><code>float getQMin() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p810264716532"><a name="p810264716532"></a><a name="p810264716532"></a>Returns the minimum feature vector value.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>Minimum feature vector value.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getThreshold`<a name="en-us_TOPIC_0000001506334633"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a><code>float getThreshold() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>Gets the threshold used to determine whether a vector is assigned to multiple buckets.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b4330591711"><a name="b4330591711"></a><a name="b4330591711"></a><code>float</code></strong>: Threshold used to determine whether a vector is assigned to multiple buckets.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `operator=`<a name="en-us_TOPIC_0000001506615085"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a><code>AscendIndexIVFSQT&amp; operator=(const AscendIndexIVFSQT&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the assignment operator of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b0567942255"><a name="b0567942255"></a><a name="b0567942255"></a><code>const AscendIndexIVFSQT&amp;</code></strong>: <code>AscendIndexIVFSQT</code> object.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `remove_ids`<a name="en-us_TOPIC_0000001506615053"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>Deletes base library features by ID.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b8255123114917"><a name="b8255123114917"></a><a name="b8255123114917"></a><code>const faiss::IDSelector &amp;sel</code></strong>: The feature vectors to delete. For details about usage and definition, see the relevant Faiss documentation.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>Returns the number of deleted feature vectors.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>This API is not supported in the current version.</p>
</td>
</tr>
</tbody>
</table>

## `reset`<a name="en-us_TOPIC_0000001506334789"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a><code>void reset() override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p10290194315362"><a name="p10290194315362"></a><a name="p10290194315362"></a>Resets the index and clears the feature data.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1414182616338"><a name="p1414182616338"></a><a name="p1414182616338"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>Do not continue using this object after calling this API.</p>
</td>
</tr>
</tbody>
</table>

## `setAddTotal`<a name="en-us_TOPIC_0000001456375316"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p11341163171010"><a name="p11341163171010"></a><a name="p11341163171010"></a><code>void setAddTotal(size_t addTotal);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p19691924201019"><a name="p19691924201019"></a><a name="p19691924201019"></a>Sets the total number of base library vectors to add. The default value is <code>100000000</code>. You must set <code>PreciseMemControl</code> to <code>true</code> first.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b999816351837"><a name="b999816351837"></a><a name="b999816351837"></a><code>size_t addTotal</code></strong>: The total number of base library vectors to add.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p8927121410219"><a name="p8927121410219"></a><a name="p8927121410219"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `setFuzzyK`<a name="en-us_TOPIC_0000001456534940"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a><code>void setFuzzyK(int value);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>Sets the maximum value for each vector when it is assigned to a bucket.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a><strong id="b144117156596"><a name="b144117156596"></a><a name="b144117156596"></a><code>int value</code></strong>: The maximum value for each vector when it is assigned to a bucket. You are advised to keep it at the default value <code>3</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p64091154105412"><a name="p64091154105412"></a><a name="p64091154105412"></a>The valid range of <code>value</code> is <code>(0, 10]</code>.</p>
</td>
</tr>
</tbody>
</table>

## `setLowerBound`<a name="en-us_TOPIC_0000001506334777"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p620411196166"><a name="p620411196166"></a><a name="p620411196166"></a><code>void setLowerBound(int lowerBound);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p6359132829"><a name="p6359132829"></a><a name="p6359132829"></a>Sets the threshold for second-level clustering. The default value is <code>32</code>.</p>
<p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>If the number of elements in a first-level clustering bucket is greater than <code>lowerBound</code>, second-level clustering is performed. Otherwise, the original state is retained.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b173513119118"><a name="b173513119118"></a><a name="b173513119118"></a><code>int lowerBound</code></strong>: The threshold for second-level clustering.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p7518112174013"><a name="p7518112174013"></a><a name="p7518112174013"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `setMemoryLimit`<a name="en-us_TOPIC_0000001506614917"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p388113312502"><a name="p388113312502"></a><a name="p388113312502"></a><code>void setMemoryLimit(float memoryLimit);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>Sets the Host memory limit. The default value is <code>32</code>, in <code>GB</code>. You must set <code>PreciseMemControl</code> to <code>true</code> first.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b177901237145615"><a name="b177901237145615"></a><a name="b177901237145615"></a><code>float memoryLimit</code></strong>: The memory limit.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p8927121410219"><a name="p8927121410219"></a><a name="p8927121410219"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `setMergeThres`<a name="en-us_TOPIC_0000001456694900"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a><code>void setMergeThres(int mergeThres);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p476310541227"><a name="p476310541227"></a><a name="p476310541227"></a>Sets the threshold for merging sub-buckets. The default value is <code>5</code>.</p>
<p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>If the number of elements in a sub-bucket after second-level clustering is smaller than <code>mergeThres</code>, merge the elements of that sub-bucket into other sub-buckets.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b117388471122"><a name="b117388471122"></a><a name="b117388471122"></a><code>int mergeThres</code></strong>: The threshold for merging sub-buckets.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p276517505390"><a name="p276517505390"></a><a name="p276517505390"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `setNumProbes`<a name="en-us_TOPIC_0000001736410013"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>void setNumProbes(int nprobes) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Sets the <code>nprobe</code> value of the current <code>AscendIndexIVFSQT</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b16217144619214"><a name="b16217144619214"></a><a name="b16217144619214"></a><code>int nprobes</code></strong>: The <code>nprobe</code> value of <code>AscendIndexIVFSQT</code>. You are advised to keep it at the default value <code>64</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li><code>nprobes</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}.</li><li><code>l2Probe</code> ≥ <code>nprobes</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, <code>l2Probe</code> ≤ <code>nprobes * 64</code>.</li><li><code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}.</li><li>For details about <code>l2Probe</code> and <code>l3SegmentNum</code>, see <a href="#setsearchparams"><code>setSearchParams</code></a>.</li><li><code>setNumProbes</code> is expected to be deprecated in September 2025. Use <a href="#setsearchparams"><code>setSearchParams</code></a> instead.</li></ul>
</td>
</tr>
</tbody>
</table>

## `setPreciseMemControl`<a name="en-us_TOPIC_0000001506334681"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p534012467165"><a name="p534012467165"></a><a name="p534012467165"></a><code>void setPreciseMemControl(bool preciseMemControl);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p19691924201019"><a name="p19691924201019"></a><a name="p19691924201019"></a>Specifies whether to precisely limit the memory size on the Host side.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b9961232154"><a name="b9961232154"></a><a name="b9961232154"></a><code>bool preciseMemControl</code></strong>: The default value is <code>false</code>, which disables precise memory limiting on the Host side. <code>true</code> enables it.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>This API is not supported in the current version. Do not call it.</p>
</td>
</tr>
</tbody>
</table>

## `setSearchParams`<a name="en-us_TOPIC_0000002052679693"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1115121782220"><a name="p1115121782220"></a><a name="p1115121782220"></a><code>void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p10890925192214"><a name="p10890925192214"></a><a name="p10890925192214"></a>Sets the parameters that affect retrieval accuracy and performance.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p14571833142216"><a name="p14571833142216"></a><a name="p14571833142216"></a><code>int nprobe</code>: The <code>nprobe</code> value of <code>AscendIndexIVFSQT</code>. You are advised to keep it at the default value <code>64</code>.</p>
<p id="p9571533132212"><a name="p9571533132212"></a><a name="p9571533132212"></a><code>int l2Probe</code>: The number of sub-buckets selected during second-stage retrieval. The default value is <code>48</code>.</p>
<p id="p8571033152213"><a name="p8571033152213"></a><a name="p8571033152213"></a><code>int l3SegmentNum</code>: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is <code>96</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li><code>nprobe</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}.</li><li><code>l2Probe</code> ≥ <code>nprobe</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, <code>l2Probe</code> ≤ <code>nprobe * 64</code>.</li><li><code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}.</li></ul>
</td>
</tr>
</tbody>
</table>

## `setSortMode`<a name="en-us_TOPIC_0000002165943965"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a><code>void setSortMode(int mode);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Sets the <code>topk</code> sorting mode. Mode <code>0</code> is approximate sorting. Mode <code>1</code> is exact sorting.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.3.1 "><p id="p6307181718287"><a name="p6307181718287"></a><a name="p6307181718287"></a><code>int mode</code>: The <code>topk</code> sorting mode.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.6.1 "><a name="ul998918501528"></a><a name="ul998918501528"></a><ul id="ul998918501528"><li>You must call this API before the <code>Search</code> API.</li><li><code>mode</code> supports only <code>0</code> or <code>1</code>. The default is <code>0</code>.<a name="ul73211618141111"></a><a name="ul73211618141111"></a><ul id="ul73211618141111"><li>Mode <code>0</code>: Approximate sorting truncates part of the <code>topk</code> results to improve performance.</li><li>Mode <code>1</code>: Exact sorting improves retrieval accuracy at the cost of some performance.</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `setThreshold`<a name="en-us_TOPIC_0000001456854808"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a><code>void setThreshold(float value);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>Sets the threshold for determining whether a vector is assigned to multiple buckets. The default value is <code>1.0</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a><strong id="b07338551508"><a name="b07338551508"></a><a name="b07338551508"></a><code>float value</code></strong>: The threshold for determining whether a vector is assigned to multiple buckets. You are advised to set it in the range <code>[1.0, 1.5]</code>. Because the Device side has a memory limit, once memory usage reaches the limit, the OOM mechanism is triggered and kills the process. You can check the Device-side memory limit data first (<code>/sys/fs/cgroup/memory/usermemory/memory.limit_in_bytes</code>) to estimate the size of the base library to add. If memory is tight, you are advised to keep the parameter in the range <code>[1.0, 1.1]</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>The valid range of <code>value</code> is <code>[0, fuzzyK - 1]</code>. For the valid range of <code>fuzzyK</code>, see the <a href="#getfuzzyk"><code>getFuzzyK</code></a> API.</p>
</td>
</tr>
</tbody>
</table>

## `setUseCpuUpdate`<a name="en-us_TOPIC_0000002167379329"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p2026619114105"><a name="p2026619114105"></a><a name="p2026619114105"></a><code>setUseCpuUpdate(int numThreads);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p7266411101019"><a name="p7266411101019"></a><a name="p7266411101019"></a>Specifies whether to use the CPU for <a href="#update"><code>update</code></a>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p172659115101"><a name="p172659115101"></a><a name="p172659115101"></a><strong id="b11603133521113"><a name="b11603133521113"></a><a name="b11603133521113"></a><code>int numThreads</code></strong>: The number of CPU cores used for <code>update</code>. The default value is the current number of CPU cores.</p>
<a name="ul76628243368"></a><a name="ul76628243368"></a><ul id="ul76628243368"><li>If the current CPU has more than 96 cores:<a name="ul11814123683920"></a><a name="ul11814123683920"></a><ul id="ul11814123683920"><li>If the current core count is smaller than the input <code>numThreads</code>, <strong id="b2028810524444"><a name="b2028810524444"></a><a name="b2028810524444"></a><code>numThreads</code></strong> = 96.</li><li>If <code>96 &lt; numThreads ≤</code> the current core count, <strong id="b324754114419"><a name="b324754114419"></a><a name="b324754114419"></a><code>numThreads</code></strong> = 96.</li><li>If <code>numThreads ≤ 96</code>, <code>numThreads</code> is the input value.</li></ul>
</li><li>If the current CPU has 96 or fewer cores:<a name="ul106753468457"></a><a name="ul106753468457"></a><ul id="ul106753468457"><li>If the current core count is smaller than the input <code>numThreads</code> and <code>numThreads ≤ 96</code>, <strong id="b20758111294711"><a name="b20758111294711"></a><a name="b20758111294711"></a><code>numThreads</code></strong> is the current core count.</li><li>If <code>0 &lt; numThreads ≤</code> the current core count, <strong id="b1695581814610"><a name="b1695581814610"></a><a name="b1695581814610"></a><code>numThreads</code></strong> is the input value.</li></ul>
</li></ul>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p152641411141014"><a name="p152641411141014"></a><a name="p152641411141014"></a>None.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1426321151015"><a name="p1426321151015"></a><a name="p1426321151015"></a>None.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul20955612171320"></a><a name="ul20955612171320"></a><ul id="ul20955612171320"><li>The value of <code>numThreads</code> must be greater than <code>0</code>.</li><li>Configure it before you use <a href="#update"><code>update</code></a>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `train`<a name="en-us_TOPIC_0000001456375352"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a><code>void train(idx_t n, const float *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Trains <code>AscendIndexIVFSQT</code>. This class inherits the relevant APIs in <code>AscendIndexIVFSQ</code> and provides concrete implementations.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b141925213710"><a name="b141925213710"></a><a name="b141925213710"></a><code>idx_t n</code></strong>: Number of feature vectors in the training set.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b1196267978"><a name="b1196267978"></a><a name="b1196267978"></a><code>const float *x</code></strong>: Feature vector data.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul15165212077"></a><a name="ul15165212077"></a><ul id="ul15165212077"><li>Training uses k-means clustering. A training set that is too small may affect query accuracy.</li><li>The valid range of <code>n</code> here is <code>nlist ≤ n ≤ 7,000,000</code>.</li><li>The pointer <code>x</code> must be a non-null pointer, and its length must be <code>dimIn * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `update`<a name="en-us_TOPIC_0000001506414869"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a><code>void update(bool cleanData = true);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>This is the second stage of three-stage retrieval. After all base library data has been added and before <code>search</code> is called, this API trains sub-bucket centers and assigns vectors to buckets according to those centers.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p6307181718287"><a name="p6307181718287"></a><a name="p6307181718287"></a><code>cleanData</code>: Specifies whether to clear intermediate data. The default value is <code>true</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p117332418161"><a name="p117332418161"></a><a name="p117332418161"></a>You only need to call this API once in a full retrieval workflow.</p>
</td>
</tr>
</tbody>
</table>

## `updateTParams`<a name="en-us_TOPIC_0000001456854936"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1418915311587"><a name="p1418915311587"></a><a name="p1418915311587"></a><code>void updateTParams(int l2Probe, int l3SegmentNum);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Passes in the parameters required for three-stage retrieval during testing.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b95897151286"><a name="b95897151286"></a><a name="b95897151286"></a><code>int l2Probe</code></strong>: The number of sub-buckets selected during second-stage retrieval. The default value is <code>48</code>.</p>
<p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b439191185"><a name="b439191185"></a><a name="b439191185"></a><code>int l3SegmentNum</code></strong>: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is <code>96</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li><code>nprobe</code> ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}.</li><li><code>l2Probe</code> ≥ <code>nprobe</code>, <code>l2Probe</code> ≤ <code>l3SegmentNum</code>, <code>l2Probe</code> ≤ <code>nprobe * 64</code>.</li><li><code>l3SegmentNum</code> ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}.</li><li>For details about the <code>nprobe</code> setting, see <a href="#setsearchparams"><code>setSearchParams</code></a>.</li><li><code>updateTParams</code> is expected to be deprecated in September 2026. Use <a href="#setsearchparams"><code>setSearchParams</code></a> instead.</li></ul>
</td>
</tr>
</tbody>
</table>
