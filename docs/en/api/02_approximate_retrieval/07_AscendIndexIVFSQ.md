# `AscendIndexIVFSQ`<a name="en-us_TOPIC_0000001506334625"></a>

## Function Description<a name="en-us_TOPIC_0000001456694964"></a>

`AscendIndexIVFSQ` uses IVF for acceleration and is a two-stage approximate retrieval algorithm.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

## `AscendIndexIVFSQ`<a name="en-us_TOPIC_0000001506414893"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a><code>AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor for <code>AscendIndexIVFSQ</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1580317419509"><a name="b1580317419509"></a><a name="b1580317419509"></a><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>: CPU-side <code>Index</code> resource.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b89210445502"><a name="b89210445502"></a><a name="b89210445502"></a><code>AscendIndexIVFSQConfig config</code></strong>: Device-side resource configuration.</p>
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
</td>
</tr>
</tbody>
</table>

<a name="table1823217151014"></a>
<table><tbody><tr id="row178231617161011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1882331711020"><a name="p1882331711020"></a><a name="p1882331711020"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p18649496410"><a name="p18649496410"></a><a name="p18649496410"></a><code>AscendIndexIVFSQ(int dims, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></p>
</td>
</tr>
<tr id="row8823317171017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7823617131019"><a name="p7823617131019"></a><a name="p7823617131019"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18823117111010"><a name="p18823117111010"></a><a name="p18823117111010"></a>Constructor for <code>AscendIndexIVFSQ</code>. It creates an <code>AscendIndexIVFSQ</code>, and the Device-side resources are set according to the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row1582381741012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128231617131015"><a name="p128231617131015"></a><a name="p128231617131015"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b342242917528"><a name="b342242917528"></a><a name="b342242917528"></a><code>int dims</code></strong>: Dimension of the feature vectors managed by <code>AscendIndexIVFSQ</code>.</p>
<p id="p169755411358"><a name="p169755411358"></a><a name="p169755411358"></a><strong id="b050293135214"><a name="b050293135214"></a><a name="b050293135214"></a><code>int nlist</code></strong>: Number of clustering centers. This corresponds to the <code>coarse_centroid_num</code> parameter in the operator generation script.</p>
<p id="p895114473339"><a name="p895114473339"></a><a name="p895114473339"></a><strong id="b42321338105211"><a name="b42321338105211"></a><a name="b42321338105211"></a><code>faiss::ScalarQuantizer::QuantizerType qtype</code></strong>: Quantizer type of <code>AscendIndexIVFSQ</code>.</p>
<p id="p7823317181017"><a name="p7823317181017"></a><a name="p7823317181017"></a><strong id="b11282104020522"><a name="b11282104020522"></a><a name="b11282104020522"></a><code>faiss::MetricType metric</code></strong>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval.</p>
<p id="p5823115014619"><a name="p5823115014619"></a><a name="p5823115014619"></a><strong id="b15262643135212"><a name="b15262643135212"></a><a name="b15262643135212"></a><code>bool encodeResidual</code></strong>: Whether to encode residuals.</p>
<p id="p168231017101016"><a name="p168231017101016"></a><a name="p168231017101016"></a><strong id="b1821144512529"><a name="b1821144512529"></a><a name="b1821144512529"></a><code>AscendIndexIVFSQConfig config</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row168231917191016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p6824121714106"><a name="p6824121714106"></a><a name="p6824121714106"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p138241317201019"><a name="p138241317201019"></a><a name="p138241317201019"></a>None</p>
</td>
</tr>
<tr id="row10824101711014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p68241317131018"><a name="p68241317131018"></a><a name="p68241317131018"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p682420176103"><a name="p682420176103"></a><a name="p682420176103"></a>None</p>
</td>
</tr>
<tr id="row5824161731013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p982431701017"><a name="p982431701017"></a><a name="p982431701017"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3234195217524"></a><a name="ul3234195217524"></a><ul id="ul3234195217524"><li><code>dims</code> ∈ {64, 128, 256, 384, 512}.</li><li><code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}.</li><li><code>qtype = ScalarQuantizer::QuantizerType::QT_8bit</code>. Only <code>ScalarQuantizer::QuantizerType::QT_8bit</code> is currently supported.</li><li><code>metric</code> ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}.<div class="note" id="note123188311292"><a name="note123188311292"></a><a name="note123188311292"></a><span class="notetitle"> Note: </span><div class="notebody"><p id="p2318163115919"><a name="p2318163115919"></a><a name="p2318163115919"></a>Currently, when <code>metric = faiss::MetricType::METRIC_INNER_PRODUCT</code>, <code>encodeResidual</code> only supports <code>false</code>. That is, the IVFSQ method with residual encoding is not currently supported. When <code>encodeResidual</code> is <code>true</code>, the code can run successfully, but there is an accuracy issue.</p>
</div></div>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table134501935171012"></a>
<table><tbody><tr id="row11451103521010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p44511935121011"><a name="p44511935121011"></a><a name="p44511935121011"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1445153561020"><a name="p1445153561020"></a><a name="p1445153561020"></a><code>AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);</code></p>
</td>
</tr>
<tr id="row1945123511015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p04512353102"><a name="p04512353102"></a><a name="p04512353102"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8451173581017"><a name="p8451173581017"></a><a name="p8451173581017"></a>Constructor for <code>AscendIndexIVFSQ</code>. It creates an <code>AscendIndexIVFSQ</code>, and the Device-side resources are set according to the values configured in <code>config</code>. This API does not perform initialization. The subclass performs the initialization-related work. This API will be deprecated later, so do not use it.</p>
</td>
</tr>
<tr id="row1645163571015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p74511835171017"><a name="p74511835171017"></a><a name="p74511835171017"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p64511935141018"><a name="p64511935141018"></a><a name="p64511935141018"></a><strong id="b54513358104"><a name="b54513358104"></a><a name="b54513358104"></a><code>int dims</code></strong>: Dimension of the feature vectors managed by <code>AscendIndexIVFSQ</code>.</p>
<p id="p13451183517103"><a name="p13451183517103"></a><a name="p13451183517103"></a><strong id="b1645153519106"><a name="b1645153519106"></a><a name="b1645153519106"></a><code>int nlist</code></strong>: Number of clustering centers. This corresponds to the <code>coarse_centroid_num</code> parameter in the operator generation script.</p>
<p id="p10451153591010"><a name="p10451153591010"></a><a name="p10451153591010"></a><strong id="b14451535111016"><a name="b14451535111016"></a><a name="b14451535111016"></a><code>faiss::MetricType metric</code></strong>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval.</p>
<p id="p54513357103"><a name="p54513357103"></a><a name="p54513357103"></a><strong id="b16451435121012"><a name="b16451435121012"></a><a name="b16451435121012"></a><code>AscendIndexIVFSQConfig config</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row8451113510107"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p5451123541012"><a name="p5451123541012"></a><a name="p5451123541012"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p12451435121015"><a name="p12451435121015"></a><a name="p12451435121015"></a>None</p>
</td>
</tr>
<tr id="row194511735181010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p17451163513101"><a name="p17451163513101"></a><a name="p17451163513101"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p5451203551020"><a name="p5451203551020"></a><a name="p5451203551020"></a>None</p>
</td>
</tr>
<tr id="row1945183511016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1345123518101"><a name="p1345123518101"></a><a name="p1345123518101"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul15452103551014"></a><a name="ul15452103551014"></a><ul id="ul15452103551014"><li><code>dims</code> ∈ {64, 128, 256, 384, 512}.</li><li><code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}.</li><li><code>metric</code> ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table663150151113"></a>
<table><tbody><tr id="row176440181111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p064509114"><a name="p064509114"></a><a name="p064509114"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p885213589106"><a name="p885213589106"></a><a name="p885213589106"></a><code>AscendIndexIVFSQ(const AscendIndexIVFSQ&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row186417021110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p664405110"><a name="p664405110"></a><a name="p664405110"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p76470121111"><a name="p76470121111"></a><a name="p76470121111"></a>Declares the copy constructor of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row964505113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2642019118"><a name="p2642019118"></a><a name="p2642019118"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b1399954415513"><a name="b1399954415513"></a><a name="b1399954415513"></a><code>const AscendIndexIVFSQ&amp;</code></strong>: Constant <code>AscendIndexIVFSQ</code>.</p>
</td>
</tr>
<tr id="row8641601111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p13648019116"><a name="p13648019116"></a><a name="p13648019116"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p06420013110"><a name="p06420013110"></a><a name="p06420013110"></a>None</p>
</td>
</tr>
<tr id="row1641608114"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p96418010111"><a name="p96418010111"></a><a name="p96418010111"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1264107115"><a name="p1264107115"></a><a name="p1264107115"></a>None</p>
</td>
</tr>
<tr id="row176420181110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p126412017119"><a name="p126412017119"></a><a name="p126412017119"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16647010117"><a name="p16647010117"></a><a name="p16647010117"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~AscendIndexIVFSQ`<a name="en-us_TOPIC_0000001456534936"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndexIVFSQ();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Destructor for <code>AscendIndexIVFSQ</code>. It destroys the <code>AscendIndexIVFSQ</code> object and releases resources.</p>
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

## `copyFrom`<a name="en-us_TOPIC_0000001456375244"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1215384082314"><a name="p1215384082314"></a><a name="p1215384082314"></a><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a><code>AscendIndexIVFSQ</code> copies an existing <code>index</code> to Ascend and retains the original Device-side resource configuration of <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b2023965517563"><a name="b2023965517563"></a><a name="b2023965517563"></a><code>const faiss::IndexIVFScalarQuantizer *index</code></strong>: CPU-side <code>Index</code> resource.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1346620304216"><a name="p1346620304216"></a><a name="p1346620304216"></a><code>index</code> must be a valid CPU <code>Index</code> pointer.</p>
<p id="p74662030922"><a name="p74662030922"></a><a name="p74662030922"></a>The value range of <code>index->d</code> is {64, 128, 256, 384, 512}.</p>
<p id="p154661430523"><a name="p154661430523"></a><a name="p154661430523"></a>The value range of <code>index->nlist</code> is {1024, 2048, 4096, 8192, 16384, 32768}.</p>
<p id="p646610301721"><a name="p646610301721"></a><a name="p646610301721"></a>The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. <code>metric_type</code> must be in {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}.</p>
<p id="p64663301228"><a name="p64663301228"></a><a name="p64663301228"></a><code>sq.qtype</code> supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</p>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000001506334649"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyTo(faiss::IndexIVFScalarQuantizer *index) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1720318284418"><a name="p1720318284418"></a><a name="p1720318284418"></a>Copies the retrieval resources of <code>AscendIndexIVFSQ</code> to the CPU side.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b750912215531"><a name="b750912215531"></a><a name="b750912215531"></a><code>faiss::IndexIVFScalarQuantizer *index</code></strong>: CPU-side <code>Index</code> resource.</p>
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

## `operator=`<a name="en-us_TOPIC_0000001456854860"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a><code>AscendIndexIVFSQ&amp; operator=(const AscendIndexIVFSQ&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the assignment operator of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b115570395614"><a name="b115570395614"></a><a name="b115570395614"></a><code>const AscendIndexIVFSQ&amp;</code></strong>: Constant <code>AscendIndexIVFSQ</code>.</p>
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

## `train`<a name="en-us_TOPIC_0000001456854976"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a><code>void train(idx_t n, const float *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Trains <code>AscendIndexIVFSQ</code>. This class inherits the relevant APIs in <code>AscendIndex</code> and provides a concrete implementation.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b351832435710"><a name="b351832435710"></a><a name="b351832435710"></a><code>idx_t n</code></strong>: Number of feature vectors in the training set.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b17199113075712"><a name="b17199113075712"></a><a name="b17199113075712"></a><code>const float *x</code></strong>: Feature vector data.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>Training uses k-means clustering. A small training set may affect query accuracy.</li><li>The value range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>.</li><li>The pointer <code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>
