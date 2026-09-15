# `AscendIndexFlat`<a id="en-us_TOPIC_0000001506334757"></a>

## Function Description<a name="en-us_TOPIC_0000001506334829"></a>

<code>AscendIndexFlat</code> is the most basic feature retrieval algorithm. It stores FP16 floating-point feature vectors and performs brute-force search.

It supports concurrent multithreaded calls. You need to set the <code>MX_INDEX_MULTITHREAD</code> environment variable to <code>1</code>, that is, <code>export MX_INDEX_MULTITHREAD=1</code>. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval uses OMP internally for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> <code>AscendIndexFlat</code> supports online operator conversion for L2 and IP distances. If the environment variable <code>MX_INDEX_USE_ONLINEOP</code> is set to <code>1</code> (set it with <code>export MX_INDEX_USE_ONLINEOP=1</code>), the operator is converted and called online. To use online operators, the application must explicitly call <code>(void)aclFinalize()</code> at the end. You also need to include the header file <code>#include "acl/acl.h"</code>.

## `AscendIndexFlat`<a name="en-us_TOPIC_0000001456375308"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a><code>AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndexFlat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>const faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6239525111613"><a name="p6239525111613"></a><a name="p6239525111613"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</p>
</td>
</tr>
</tbody>
</table>

<a name="table1735274911381"></a>
<table><tbody><tr id="row163522495386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p113526492389"><a name="p113526492389"></a><a name="p113526492389"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p935216491386"><a name="p935216491386"></a><a name="p935216491386"></a><code>AscendIndexFlat(const faiss::IndexIDMap *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></p>
</td>
</tr>
<tr id="row14352124915385"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p2352749113819"><a name="p2352749113819"></a><a name="p2352749113819"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13528493381"><a name="p13528493381"></a><a name="p13528493381"></a>Constructor of <code>AscendIndexFlat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row13352184923815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p535219495381"><a name="p535219495381"></a><a name="p535219495381"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p11352164923819"><a name="p11352164923819"></a><a name="p11352164923819"></a><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</p>
<p id="p14352104915380"><a name="p14352104915380"></a><a name="p14352104915380"></a><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row5352154943813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p12352114933817"><a name="p12352114933817"></a><a name="p12352114933817"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16352194953814"><a name="p16352194953814"></a><a name="p16352194953814"></a>None</p>
</td>
</tr>
<tr id="row1735214943812"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1535284912383"><a name="p1535284912383"></a><a name="p1535284912383"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p535264943818"><a name="p535264943818"></a><a name="p535264943818"></a>None</p>
</td>
</tr>
<tr id="row235216491381"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1035274933810"><a name="p1035274933810"></a><a name="p1035274933810"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</p>
</td>
</tr>
</tbody>
</table>

<a name="table142416323911"></a>
<table><tbody><tr id="row1257343916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p19251332393"><a name="p19251332393"></a><a name="p19251332393"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11207257504"><a name="p11207257504"></a><a name="p11207257504"></a><code>AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></p>
</td>
</tr>
<tr id="row2258310398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p9252383918"><a name="p9252383918"></a><a name="p9252383918"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p102553123917"><a name="p102553123917"></a><a name="p102553123917"></a>Constructor of <code>AscendIndexFlat</code>. It creates an <code>AscendIndexFlat</code> with dimension <code>dims</code>. The dimension of the vector set managed by a single <code>Index</code> is unique. It configures device-side resources according to the values in <code>config</code>.</p>
</td>
</tr>
<tr id="row1525633399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p16259393917"><a name="p16259393917"></a><a name="p16259393917"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10256318392"><a name="p10256318392"></a><a name="p10256318392"></a><code>int dims</code>: Dimension of the feature vector set managed by <code>AscendIndex</code>.</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><code>faiss::MetricType metric</code>: Distance metric type used by <code>AscendIndexFlat</code> when performing feature vector similarity retrieval.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row102514316397"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7254343920"><a name="p7254343920"></a><a name="p7254343920"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p22553153917"><a name="p22553153917"></a><a name="p22553153917"></a>None</p>
</td>
</tr>
<tr id="row22516313918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p62511393920"><a name="p62511393920"></a><a name="p62511393920"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p92519310395"><a name="p92519310395"></a><a name="p92519310395"></a>None</p>
</td>
</tr>
<tr id="row6251237398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p18251437396"><a name="p18251437396"></a><a name="p18251437396"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1988754314420"></a><a name="ul1988754314420"></a><ul id="ul1988754314420"><li><code>dims</code> ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}.</li><li><code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table5169814143913"></a>
<table><tbody><tr id="row1116961423914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p4169121473916"><a name="p4169121473916"></a><a name="p4169121473916"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7112274471"><a name="p7112274471"></a><a name="p7112274471"></a><code>AscendIndexFlat(const AscendIndexFlat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row1416991413916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p121699143394"><a name="p121699143394"></a><a name="p121699143394"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p191699142396"><a name="p191699142396"></a><a name="p191699142396"></a>Declares this <code>Index</code> copy constructor as deleted, meaning that the type is non-copyable.</p>
</td>
</tr>
<tr id="row61691614163913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5169814123916"><a name="p5169814123916"></a><a name="p5169814123916"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndexFlat&amp;</code>: Constant <code>AscendIndexFlat</code>.</p>
</td>
</tr>
<tr id="row151691414153917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1016941403913"><a name="p1016941403913"></a><a name="p1016941403913"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p21691714113915"><a name="p21691714113915"></a><a name="p21691714113915"></a>None</p>
</td>
</tr>
<tr id="row181697141391"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3169161413395"><a name="p3169161413395"></a><a name="p3169161413395"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p01691714163917"><a name="p01691714163917"></a><a name="p01691714163917"></a>None</p>
</td>
</tr>
<tr id="row1416991443915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p01691014143911"><a name="p01691014143911"></a><a name="p01691014143911"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p161695143397"><a name="p161695143397"></a><a name="p161695143397"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table04891725153918"></a>
<table><tbody><tr id="row194894256391"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p13489525203913"><a name="p13489525203913"></a><a name="p13489525203913"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndexFlat();</code></p>
</td>
</tr>
<tr id="row1248962513399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p94891225163914"><a name="p94891225163914"></a><a name="p94891225163914"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1148902511397"><a name="p1148902511397"></a><a name="p1148902511397"></a>Destructor of <code>AscendIndexFlat</code>. It destroys the <code>AscendIndexFlat</code> object and releases resources.</p>
</td>
</tr>
<tr id="row15489182583911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p10489142503915"><a name="p10489142503915"></a><a name="p10489142503915"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
</td>
</tr>
<tr id="row6489525163919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p10489152514391"><a name="p10489152514391"></a><a name="p10489152514391"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p13489425133914"><a name="p13489425133914"></a><a name="p13489425133914"></a>None</p>
</td>
</tr>
<tr id="row1248992503912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p548912515395"><a name="p548912515395"></a><a name="p548912515395"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16489182512392"><a name="p16489182512392"></a><a name="p16489182512392"></a>None</p>
</td>
</tr>
<tr id="row16489725193918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p134891125183917"><a name="p134891125183917"></a><a name="p134891125183917"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15489102517393"><a name="p15489102517393"></a><a name="p15489102517393"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `copyFrom`<a name="en-us_TOPIC_0000001456535180"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyFrom(const faiss::IndexFlat *index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies an existing <code>Index</code> to Ascend based on <code>AscendIndexFlat</code>, clears the current base vector set in <code>AscendIndexFlat</code>, and keeps the original device-side resource configuration of <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>const faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</p>
</td>
</tr>
</tbody>
</table>

<a name="table525914213409"></a>
<table><tbody><tr id="row16259174214406"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1025994215407"><a name="p1025994215407"></a><a name="p1025994215407"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6259442124016"><a name="p6259442124016"></a><a name="p6259442124016"></a><code>void copyFrom(const faiss::IndexIDMap *index);</code></p>
</td>
</tr>
<tr id="row1925914423401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1025984212403"><a name="p1025984212403"></a><a name="p1025984212403"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p92592422406"><a name="p92592422406"></a><a name="p92592422406"></a>Copies an existing <code>index</code> to Ascend based on <code>AscendIndexFlat</code>, clears the current base vector set in <code>AscendIndexFlat</code>, and keeps the original device-side resource configuration of <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row5259842124019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7259174224012"><a name="p7259174224012"></a><a name="p7259174224012"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p925984284010"><a name="p925984284010"></a><a name="p925984284010"></a><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row14259042124016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1125954204015"><a name="p1125954204015"></a><a name="p1125954204015"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14260174234011"><a name="p14260174234011"></a><a name="p14260174234011"></a>None</p>
</td>
</tr>
<tr id="row20260174294018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12260842114015"><a name="p12260842114015"></a><a name="p12260842114015"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p15260242114012"><a name="p15260242114012"></a><a name="p15260242114012"></a>None</p>
</td>
</tr>
<tr id="row1626015428401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p152607429407"><a name="p152607429407"></a><a name="p152607429407"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1726012424402"><a name="p1726012424402"></a><a name="p1726012424402"></a><code>index</code> must be a valid <code>IndexIDMap</code> pointer. Otherwise, the program may crash or the function may become unavailable. The dimension <code>d</code> parameter of this <code>Index</code> must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</p>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000001456535148"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyTo(faiss::IndexFlat *index) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies the retrieval resources of <code>AscendIndexFlat</code> to the CPU side.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</p>
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

<a name="table154531752144016"></a>
<table><tbody><tr id="row12453652124015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1245375234010"><a name="p1245375234010"></a><a name="p1245375234010"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p194531452144019"><a name="p194531452144019"></a><a name="p194531452144019"></a><code>void copyTo(faiss::IndexIDMap *index) const;</code></p>
</td>
</tr>
<tr id="row74535524403"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7453752174010"><a name="p7453752174010"></a><a name="p7453752174010"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p045325210409"><a name="p045325210409"></a><a name="p045325210409"></a>Copies the retrieval resources of <code>AscendIndexFlat</code> to the CPU side.</p>
</td>
</tr>
<tr id="row11453135211406"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p94531252184014"><a name="p94531252184014"></a><a name="p94531252184014"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16453175224020"><a name="p16453175224020"></a><a name="p16453175224020"></a><code>faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row345495215407"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p8454135218406"><a name="p8454135218406"></a><a name="p8454135218406"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p74541752174010"><a name="p74541752174010"></a><a name="p74541752174010"></a>None</p>
</td>
</tr>
<tr id="row19454852184017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3454752144017"><a name="p3454752144017"></a><a name="p3454752144017"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p145465264013"><a name="p145465264013"></a><a name="p145465264013"></a>None</p>
</td>
</tr>
<tr id="row845415211403"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p14541452144011"><a name="p14541452144011"></a><a name="p14541452144011"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p24541452194012"><a name="p24541452194012"></a><a name="p24541452194012"></a><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The user must free the memory occupied by the <code>Index</code>.</p>
</td>
</tr>
</tbody>
</table>

## `getBase`<a name="en-us_TOPIC_0000001456375236"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void getBase(int deviceId, char* xb) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the feature vectors managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int deviceId</code>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><code>char* xb</code>: The base library feature vectors stored by <code>AscendIndexFlat</code> on <code>deviceId</code>.</p>
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
<p id="p835381215917"><a name="p835381215917"></a><a name="p835381215917"></a><code>xb</code> must be a non-null pointer, and its length must be <code>dims * BaseSize * sizeof(float32)</code> bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>BaseSize</code> is the return value of <code>getBaseSize</code>.</p>
</td>
</tr>
</tbody>
</table>

## `getBaseSize`<a name="en-us_TOPIC_0000001456854956"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>size_t getBaseSize(int deviceId) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the number of feature vectors managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int deviceId</code>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Number of feature vectors on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>deviceId</code> must be a valid device ID.</p>
</td>
</tr>
</tbody>
</table>

## `getIdxMap`<a name="en-us_TOPIC_0000001506334785"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the feature vector IDs managed by this <code>AscendIndexFlat</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int deviceId</code>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><code>std::vector&lt;idx_t&gt; &amp;idxMap</code>: The base library feature vector IDs stored by <code>AscendIndexFlat</code> on <code>deviceId</code>.</p>
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

## `operator =`<a name="en-us_TOPIC_0000001506495701"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexFlat&amp; operator=(const AscendIndexFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares this <code>Index</code> assignment operator as deleted, meaning that the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndexFlat&amp;</code>: Constant <code>AscendIndexFlat</code>.</p>
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

## `search_with_masks`<a name="en-us_TOPIC_0000001810529650"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11977033135012"><a name="p11977033135012"></a><a name="p11977033135012"></a><code>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7584153011582"><a name="p7584153011582"></a><a name="p7584153011582"></a>The feature vector query API of <code>AscendIndexFlat</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. <code>mask</code> is a bit string of <code>0</code>s and <code>1</code>s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. <code>1</code> means participate, and <code>0</code> means do not participate.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><code>idx_t n</code>: The number of query feature vectors.</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><code>const float *x</code>: Feature vector data.</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><code>idx_t k</code>: The number of most similar results to return.</p>
<p id="p841235065815"><a name="p841235065815"></a><a name="p841235065815"></a><code>const void *mask</code>: Feature library mask.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul184581435495"></a><a name="ul184581435495"></a><ul id="ul184581435495"><li><code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>.</li><li><code>k</code> is usually not allowed to exceed <code>4096</code>.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>mask</code> must be a non-null pointer, and its length must be <code>n * ceil(ntotal / 8)</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>ntotal</code> is the number of base library features.</li><li><code>mask</code> is set according to the order of the base library. If you call <code>remove_ids</code> to delete feature vectors before calling this API, the order of the base library features changes. First call <code>getIdxMap</code> to obtain the IDs of the base library features, and then set <code>mask</code>.</li><li>To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table0628133121511"></a>
<table><tbody><tr id="row5682739155"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.1.1"><p id="p156823331511"><a name="p156823331511"></a><a name="p156823331511"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.1.1 "><p id="p1868293131514"><a name="p1868293131514"></a><a name="p1868293131514"></a><code>void search_with_masks(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></p>
</td>
</tr>
<tr id="row1368233181518"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.2.1"><p id="p36821633157"><a name="p36821633157"></a><a name="p36821633157"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p8682432157"><a name="p8682432157"></a><a name="p8682432157"></a>The feature vector query API of <code>AscendIndexFlat</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors. <code>mask</code> is a bit string of <code>0</code>s and <code>1</code>s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. <code>1</code> means participate, and <code>0</code> means do not participate.</p>
</td>
</tr>
<tr id="row196837312153"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.3.1"><p id="p10683233157"><a name="p10683233157"></a><a name="p10683233157"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.3.1 "><p id="p1768318381520"><a name="p1768318381520"></a><a name="p1768318381520"></a><code>idx_t n</code>: The number of query feature vectors.</p>
<p id="p2683123121519"><a name="p2683123121519"></a><a name="p2683123121519"></a><code>const uint16_t *x</code>: Feature vector data.</p>
<p id="p146831137157"><a name="p146831137157"></a><a name="p146831137157"></a><code>idx_t k</code>: The number of most similar results to return.</p>
<p id="p768320371520"><a name="p768320371520"></a><a name="p768320371520"></a><code>const void *mask</code>: Feature library mask.</p>
</td>
</tr>
<tr id="row1868310301516"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.4.1"><p id="p1668319311511"><a name="p1668319311511"></a><a name="p1668319311511"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.4.1 "><p id="p1168315312152"><a name="p1168315312152"></a><a name="p1168315312152"></a><code>float *distances</code>: The distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p10683133158"><a name="p10683133158"></a><a name="p10683133158"></a><code>idx_t *labels</code>: The IDs of the top <code>k</code> nearest vectors for the query.</p>
</td>
</tr>
<tr id="row1668310317152"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.5.1"><p id="p19683203141517"><a name="p19683203141517"></a><a name="p19683203141517"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.5.1 "><p id="p468393131518"><a name="p468393131518"></a><a name="p468393131518"></a>None</p>
</td>
</tr>
<tr id="row1768312318157"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.6.1"><p id="p1368343181514"><a name="p1368343181514"></a><a name="p1368343181514"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><a name="ul16154204141611"></a><a name="ul16154204141611"></a><ul id="ul16154204141611"><li><code>n</code> must satisfy <code>0 &lt; n &lt; 1e9</code>.</li><li><code>k</code> is usually not allowed to exceed <code>4096</code>.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers, and each must have a length of <code>k * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>mask</code> must be a non-null pointer, and its length must be <code>n * ceil(ntotal / 8)</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>ntotal</code> is the number of base library features.</li><li><code>mask</code> is set according to the order of the base library. If you call <code>remove_ids</code> to delete feature vectors before calling this API, the order of the base library features changes. First call <code>getIdxMap</code> to obtain the IDs of the base library features, and then set <code>mask</code>.</li><li>To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect.</li></ul>
</td>
</tr>
</tbody>
</table>
