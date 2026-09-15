# `AscendIndexSQ`<a name="en-us_TOPIC_0000001506614969"></a>

## Function Description<a name="en-us_TOPIC_0000001456695120"></a>

`AscendIndexSQ` performs Scalar Quantization on the input vectors.

The vectors stored in the base vector set and the query vectors of each API must be normalized float values.

It supports multithreaded concurrent calls. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

## `AscendIndexSQ`<a name="en-us_TOPIC_0000001506614933"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a><code>AscendIndexSQ(const faiss::IndexScalarQuantizer* index, AscendIndexSQConfig config = AscendIndexSQConfig());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>The constructor of <code>AscendIndexSQ</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b6104639815"><a name="b6104639815"></a><a name="b6104639815"></a><code>const faiss::IndexScalarQuantizer* index</code></strong>: CPU-side index resource.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1066616367111"><a name="b1066616367111"></a><a name="b1066616367111"></a><code>AscendIndexSQConfig config</code></strong>: Device-side resource configuration.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. <code>d</code> (the dimension parameter of the <code>Index</code>) must be in <code>{64, 128, 256, 384, 512, 768}</code>. The total number of base library vectors must be in the range <code>0 ≤ n &lt; 1e9</code>. <code>metric_type</code> must be in <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. <code>sq.qtype</code> supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table207325212487"></a>
<table><tbody><tr id="row57316521481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1731752204815"><a name="p1731752204815"></a><a name="p1731752204815"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a><code>AscendIndexSQ(const faiss::IndexIDMap* index, AscendIndexSQConfig config = AscendIndexSQConfig());</code></p>
</td>
</tr>
<tr id="row1573165204811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p147395220488"><a name="p147395220488"></a><a name="p147395220488"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10738528483"><a name="p10738528483"></a><a name="p10738528483"></a>The constructor of <code>AscendIndexSQ</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row3731652104814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p197495254810"><a name="p197495254810"></a><a name="p197495254810"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b11364627323"><a name="b11364627323"></a><a name="b11364627323"></a><code>const faiss::IndexIDMap* index</code></strong>: CPU-side <code>Index</code> resource.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b75252919212"><a name="b75252919212"></a><a name="b75252919212"></a><code>AscendIndexSQConfig config</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row37465224818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p147415211489"><a name="p147415211489"></a><a name="p147415211489"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8741452194814"><a name="p8741452194814"></a><a name="p8741452194814"></a>None</p>
</td>
</tr>
<tr id="row167475217487"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p474185234815"><a name="p474185234815"></a><a name="p474185234815"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p197455214820"><a name="p197455214820"></a><a name="p197455214820"></a>None</p>
</td>
</tr>
<tr id="row97455219484"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p974125219485"><a name="p974125219485"></a><a name="p974125219485"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p18741526488"><a name="p18741526488"></a><a name="p18741526488"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. <code>d</code> (the dimension parameter of the member index) must be in <code>{64, 128, 256, 384, 512, 768}</code>. The total number of base library vectors must be in the range <code>0 ≤ n &lt; 1e9</code>. <code>metric_type</code> must be in <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. <code>sq.qtype</code> supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table1132217014918"></a>
<table><tbody><tr id="row1132250114917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1432220012499"><a name="p1432220012499"></a><a name="p1432220012499"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11207257504"><a name="p11207257504"></a><a name="p11207257504"></a><code>AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexSQConfig config = AscendIndexSQConfig());</code></p>
</td>
</tr>
<tr id="row1232215064915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p9322140114910"><a name="p9322140114910"></a><a name="p9322140114910"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p432219014915"><a name="p432219014915"></a><a name="p432219014915"></a>The constructor of <code>AscendIndexSQ</code>. It creates an <code>AscendIndex</code> with dimension <code>dims</code>. The dimension of a vector set managed by one <code>Index</code> is unique. It then sets Device-side resources according to the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row23229044916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p732210174911"><a name="p732210174911"></a><a name="p732210174911"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16322607495"><a name="p16322607495"></a><a name="p16322607495"></a><strong id="b63116565211"><a name="b63116565211"></a><a name="b63116565211"></a><code>int dims</code></strong>: The dimension of a set of feature vectors managed by <code>AscendIndexSQ</code>. Valid values: <code>64, 128, 256, 384, 512, 768</code>.</p>
<p id="p995710373711"><a name="p995710373711"></a><a name="p995710373711"></a><strong id="b155310333495"><a name="b155310333495"></a><a name="b155310333495"></a><code>faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit</code></strong>: Currently, only <code>ScalarQuantizer::QuantizerType::QT_8bit</code> is supported.</p>
<p id="p163221204497"><a name="p163221204497"></a><a name="p163221204497"></a><strong id="b20474132614312"><a name="b20474132614312"></a><a name="b20474132614312"></a><code>faiss::MetricType metric</code></strong>: The distance metric type used by <code>AscendIndex</code> when performing feature vector similarity retrieval. Valid values: <code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.</p>
<p id="p1132217074917"><a name="p1132217074917"></a><a name="p1132217074917"></a><strong id="b1687210287316"><a name="b1687210287316"></a><a name="b1687210287316"></a><code>AscendIndexSQConfig config</code></strong>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row163222012498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1322110154919"><a name="p1322110154919"></a><a name="p1322110154919"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p73221102490"><a name="p73221102490"></a><a name="p73221102490"></a>None</p>
</td>
</tr>
<tr id="row6322190184913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1032260194919"><a name="p1032260194919"></a><a name="p1032260194919"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8322204495"><a name="p8322204495"></a><a name="p8322204495"></a>None</p>
</td>
</tr>
<tr id="row10322120124920"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p173221017492"><a name="p173221017492"></a><a name="p173221017492"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul193685551031"></a><a name="ul193685551031"></a><ul id="ul193685551031"><li><code>dims</code> must be in <code>{64, 128, 256, 384, 512, 768}</code>.</li><li><code>metric</code> must be in <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table16655810104919"></a>
<table><tbody><tr id="row19655810194912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16655710184912"><a name="p16655710184912"></a><a name="p16655710184912"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p8445440165114"><a name="p8445440165114"></a><a name="p8445440165114"></a><code>AscendIndexSQ(const AscendIndexSQ&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row665561014492"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p10655161013492"><a name="p10655161013492"></a><a name="p10655161013492"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p365541024916"><a name="p365541024916"></a><a name="p365541024916"></a>Declares the copy constructor as deleted. In other words, this is a non-copyable type.</p>
</td>
</tr>
<tr id="row4655110114913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1365501024920"><a name="p1365501024920"></a><a name="p1365501024920"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b16480691410"><a name="b16480691410"></a><a name="b16480691410"></a><code>const AscendIndexSQ&amp;</code></strong>: An <code>AscendIndexSQ</code> object.</p>
</td>
</tr>
<tr id="row13655121044912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p15655111064920"><a name="p15655111064920"></a><a name="p15655111064920"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p146551210104911"><a name="p146551210104911"></a><a name="p146551210104911"></a>None</p>
</td>
</tr>
<tr id="row116554109498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1265512105496"><a name="p1265512105496"></a><a name="p1265512105496"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p46569101499"><a name="p46569101499"></a><a name="p46569101499"></a>None</p>
</td>
</tr>
<tr id="row16568103491"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p8656810104914"><a name="p8656810104914"></a><a name="p8656810104914"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p19656210134916"><a name="p19656210134916"></a><a name="p19656210134916"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table17704194534915"></a>
<table><tbody><tr id="row147041745174918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p170414524913"><a name="p170414524913"></a><a name="p170414524913"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndexSQ();</code></p>
</td>
</tr>
<tr id="row370416455499"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p77042459498"><a name="p77042459498"></a><a name="p77042459498"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p97045456493"><a name="p97045456493"></a><a name="p97045456493"></a>The destructor of <code>AscendIndexSQ</code>. It destroys the <code>AscendIndexSQ</code> object and releases resources.</p>
</td>
</tr>
<tr id="row470419456497"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p77041745124910"><a name="p77041745124910"></a><a name="p77041745124910"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
</td>
</tr>
<tr id="row57042456497"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p470444514493"><a name="p470444514493"></a><a name="p470444514493"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p57041745194915"><a name="p57041745194915"></a><a name="p57041745194915"></a>None</p>
</td>
</tr>
<tr id="row4704845104910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p19704134513498"><a name="p19704134513498"></a><a name="p19704134513498"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p770454516493"><a name="p770454516493"></a><a name="p770454516493"></a>None</p>
</td>
</tr>
<tr id="row1870454504914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p77046459495"><a name="p77046459495"></a><a name="p77046459495"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p5704184513492"><a name="p5704184513492"></a><a name="p5704184513492"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `copyFrom`<a name="en-us_TOPIC_0000001506615037"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyFrom(const faiss::IndexScalarQuantizer* index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies an existing <code>index</code> to Ascend based on <code>AscendIndexSQ</code>, clears the current base library of <code>AscendIndexSQ</code>, and keeps the existing Device-side resource configuration of <code>AscendIndexSQ</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b113197161055"><a name="b113197161055"></a><a name="b113197161055"></a><code>const faiss::IndexScalarQuantizer* index</code></strong>: CPU-side <code>Index</code> resource.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. <code>d</code> (the dimension parameter of the <code>Index</code>) must be in <code>{64, 128, 256, 384, 512, 768}</code>. The total number of base library vectors must be in the range <code>0 ≤ n &lt; 1e9</code>. <code>metric_type</code> must be in <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. <code>sq.qtype</code> supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table853716365015"></a>
<table><tbody><tr id="row1253763155012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p65375319502"><a name="p65375319502"></a><a name="p65375319502"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15371437503"><a name="p15371437503"></a><a name="p15371437503"></a><code>void copyFrom(const faiss::IndexIDMap* index);</code></p>
</td>
</tr>
<tr id="row95371733508"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p11537735509"><a name="p11537735509"></a><a name="p11537735509"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p153715311508"><a name="p153715311508"></a><a name="p153715311508"></a>Copies an existing <code>index</code> to Ascend based on <code>AscendIndexSQ</code>, clears the current base library of <code>AscendIndexSQ</code>, and keeps the existing Device-side resource configuration of <code>AscendIndexSQ</code>.</p>
</td>
</tr>
<tr id="row155371130507"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1153720316503"><a name="p1153720316503"></a><a name="p1153720316503"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1553715335013"><a name="p1553715335013"></a><a name="p1553715335013"></a><strong id="b1898128253"><a name="b1898128253"></a><a name="b1898128253"></a><code>const faiss::IndexIDMap *index</code></strong>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row1253716318502"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p115377325010"><a name="p115377325010"></a><a name="p115377325010"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p153712355015"><a name="p153712355015"></a><a name="p153712355015"></a>None</p>
</td>
</tr>
<tr id="row9537203125019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p553711365019"><a name="p553711365019"></a><a name="p553711365019"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p553743135020"><a name="p553743135020"></a><a name="p553743135020"></a>None</p>
</td>
</tr>
<tr id="row55373320504"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1537143115010"><a name="p1537143115010"></a><a name="p1537143115010"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p12537193155012"><a name="p12537193155012"></a><a name="p12537193155012"></a><code>index</code> must be a valid <code>IndexIDMap</code> pointer. <code>d</code> (the dimension parameter of the member index) must be in <code>{64, 128, 256, 384, 512, 768}</code>. The total number of base library vectors must be in the range <code>0 ≤ n &lt; 1e9</code>. <code>metric_type</code> must be in <code>{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</code>. <code>sq.qtype</code> supports only <code>ScalarQuantizer::QuantizerType::QT_8bit</code>.</p>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000001456695084"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1019716501395"><a name="p1019716501395"></a><a name="p1019716501395"></a><code>void copyTo(faiss::IndexScalarQuantizer* index) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies the retrieval resources of <code>AscendIndexSQ</code> to the CPU side.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b217675214518"><a name="b217675214518"></a><a name="b217675214518"></a><code>faiss::IndexScalarQuantizer* index</code></strong>: CPU-side <code>Index</code> resource.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The user must free the memory occupied by the <code>Index</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table817201512500"></a>
<table><tbody><tr id="row1517171595016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9171015145014"><a name="p9171015145014"></a><a name="p9171015145014"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyTo(faiss::IndexIDMap* index) const;</code></p>
</td>
</tr>
<tr id="row5171115145019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3177155503"><a name="p3177155503"></a><a name="p3177155503"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p51761519504"><a name="p51761519504"></a><a name="p51761519504"></a>Copies the retrieval resources of <code>AscendIndexSQ</code> to the CPU side.</p>
</td>
</tr>
<tr id="row101711535017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p717151512506"><a name="p717151512506"></a><a name="p717151512506"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12170155508"><a name="p12170155508"></a><a name="p12170155508"></a><strong id="b18757155564"><a name="b18757155564"></a><a name="b18757155564"></a><code>faiss::IndexIDMap *index</code></strong>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row61781514507"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p417181516501"><a name="p417181516501"></a><a name="p417181516501"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1617181515015"><a name="p1617181515015"></a><a name="p1617181515015"></a>None</p>
</td>
</tr>
<tr id="row917171512503"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p11172156506"><a name="p11172156506"></a><a name="p11172156506"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p317215145014"><a name="p317215145014"></a><a name="p317215145014"></a>None</p>
</td>
</tr>
<tr id="row6179153503"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p20171415125010"><a name="p20171415125010"></a><a name="p20171415125010"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p191791513507"><a name="p191791513507"></a><a name="p191791513507"></a><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The user must free the memory occupied by the <code>Index</code>.</p>
</td>
</tr>
</tbody>
</table>

## `getBase`<a name="en-us_TOPIC_0000001456694928"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void getBase(int deviceId, char* xb) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the feature vectors managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b0928171820610"><a name="b0928171820610"></a><a name="b0928171820610"></a><code>int deviceId</code></strong>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b1959712112614"><a name="b1959712112614"></a><a name="b1959712112614"></a><code>char* xb</code></strong>: The base library feature vectors stored by <code>AscendIndexSQ</code> on <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul126112716618"></a><a name="ul126112716618"></a><ul id="ul126112716618"><li><code>deviceId</code> must be a valid device ID.</li><li><code>xb</code> must be a non-null pointer, and its length must be <code>dims * BaseSize * sizeof(uint8_t)</code> bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>BaseSize</code> is the return value of <code>getBaseSize</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `getBaseSize`<a name="en-us_TOPIC_0000001456854788"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>size_t getBaseSize(int deviceId) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the number of feature vectors managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b10691175516203"><a name="b10691175516203"></a><a name="b10691175516203"></a><code>int deviceId</code></strong>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>The number of feature vectors on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>deviceId</code> must be a valid device ID.</p>
</td>
</tr>
</tbody>
</table>

## `getIdxMap`<a name="en-us_TOPIC_0000001456375152"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt;&amp; idxMap) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the feature vector IDs managed by this <code>AscendIndexSQ</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b156818551961"><a name="b156818551961"></a><a name="b156818551961"></a><code>int deviceId</code></strong>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b69151357768"><a name="b69151357768"></a><a name="b69151357768"></a><code>std::vector&lt;idx_t&gt; &amp;idxMap</code></strong>: The base library feature vector IDs stored by <code>AscendIndexSQ</code> on <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>deviceId</code> must be a valid device ID.</p>
</td>
</tr>
</tbody>
</table>

## `operator =`<a name="en-us_TOPIC_0000001456375300"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7779180105218"><a name="p7779180105218"></a><a name="p7779180105218"></a><code>AscendIndexSQ&amp; operator=(const AscendIndexSQ&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the assignment operator as deleted. In other words, this is a non-copyable type.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b16267637049"><a name="b16267637049"></a><a name="b16267637049"></a><code>const AscendIndexSQ&amp;</code></strong>: An <code>AscendIndexSQ</code> object.</p>
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

## `search_with_filter`<a name="en-us_TOPIC_0000001810589742"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p676092910161"><a name="p676092910161"></a><a name="p676092910161"></a><code>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10290157145418"><a name="p10290157145418"></a><a name="p10290157145418"></a>The feature vector query API of <code>AscendIndexSQ</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It also provides CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array of length <code>n * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four values of each filter, that is, 128 bits, represent the corresponding CID. The last two values represent the left-closed timestamp interval, that is, <code>[x, y)</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b1976572871110"><a name="b1976572871110"></a><a name="b1976572871110"></a><code>idx_t n</code></strong>: The number of query feature vectors. Valid range: <code>0 &lt; n &lt; 1e9</code>.</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b12370123112117"><a name="b12370123112117"></a><a name="b12370123112117"></a><code>const float *x</code></strong>: Feature vector data.</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b3869633101113"><a name="b3869633101113"></a><a name="b3869633101113"></a><code>idx_t k</code></strong>: The number of most similar results to return. Valid range: typically no more than <code>4096</code>.</p>
<p id="p13978130112613"><a name="p13978130112613"></a><a name="p13978130112613"></a><strong id="b157981335181116"><a name="b157981335181116"></a><a name="b157981335181116"></a><code>const void *filters</code></strong>: Filter conditions.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b7967538161117"><a name="b7967538161117"></a><a name="b7967538161117"></a><code>float *distances</code></strong>: The distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b34371240191113"><a name="b34371240191113"></a><a name="b34371240191113"></a><code>idx_t *labels</code></strong>: The IDs of the top <code>k</code> nearest vectors for the query.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul6584134771119"></a><a name="ul6584134771119"></a><ul id="ul6584134771119"><li><code>n</code> must be in the range <code>(0, 1e9)</code>.</li><li><code>k</code> is usually not allowed to exceed <code>4096</code>.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>filters</code> must be a non-null pointer to a <code>uint32_t</code> array of length <code>n * 6</code>. Otherwise, out-of-bounds read errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `search_with_masks`<a name="en-us_TOPIC_0000001456694932"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11977033135012"><a name="p11977033135012"></a><a name="p11977033135012"></a><code>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7584153011582"><a name="p7584153011582"></a><a name="p7584153011582"></a>The feature vector query API of <code>AscendIndexSQ</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. <code>mask</code> is a bit string of <code>0</code>s and <code>1</code>s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. <code>1</code> means participate, and <code>0</code> means do not participate.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b8509184711813"><a name="b8509184711813"></a><a name="b8509184711813"></a><code>idx_t n</code></strong>: The number of query feature vectors. Valid range: <code>0 &lt; n &lt; 1e9</code>.</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b2086114494814"><a name="b2086114494814"></a><a name="b2086114494814"></a><code>const float *x</code></strong>: Feature vector data.</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b2484155112813"><a name="b2484155112813"></a><a name="b2484155112813"></a><code>idx_t k</code></strong>: The number of most similar results to return. Valid range: typically no more than <code>4096</code>.</p>
<p id="p841235065815"><a name="p841235065815"></a><a name="p841235065815"></a><strong id="b77116531187"><a name="b77116531187"></a><a name="b77116531187"></a><code>const void *mask</code></strong>: Feature library mask.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b676467798"><a name="b676467798"></a><a name="b676467798"></a><code>float *distances</code></strong>: The distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b5824159193"><a name="b5824159193"></a><a name="b5824159193"></a><code>idx_t *labels</code></strong>: The IDs of the top <code>k</code> nearest vectors for the query.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul184581435495"></a><a name="ul184581435495"></a><ul id="ul184581435495"><li><code>n</code> must be in the range <code>(0, 1e9)</code>.</li><li><code>k</code> is usually not allowed to exceed <code>4096</code>.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>mask</code> must be a non-null pointer, and its length must be <code>n * ceil(ntotal / 8)</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>ntotal</code> is the number of base library features.</li><li><code>mask</code> is set according to the order of the base library. If you call <code>remove_ids</code> to delete feature vectors before calling this API, the order of the base library features changes. First call <code>getIdxMap</code> to obtain the IDs of the base library features, and then set <code>mask</code>.</li><li>To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect.</li></ul>
</td>
</tr>
</tbody>
</table>

## `train`<a name="en-us_TOPIC_0000001506414905"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void train(idx_t n, const float *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Trains the quantizer on <code>AscendIndexSQ</code>. This API inherits the interface from <code>AscendFaiss</code> and provides the concrete implementation. <strong>Note that you must train the <code>Index</code> before you call <code>add</code>.</strong></p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b986713077"><a name="b986713077"></a><a name="b986713077"></a><code>idx_t n</code></strong>: The number of feature vectors in the training set. Valid range: <code>0 &lt; n &lt; 1e9</code>.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b11961715876"><a name="b11961715876"></a><a name="b11961715876"></a><code>const float *x</code></strong>: Feature vector data.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1796193986"></a><a name="ul1796193986"></a><ul id="ul1796193986"><li><code>n</code> must be in the range <code>(0, 1e9)</code>.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>Training collects data distribution statistics. A small training set may affect query accuracy.</li></ul>
</td>
</tr>
</tbody>
</table>
