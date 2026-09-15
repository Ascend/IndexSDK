# `AscendIndexInt8Flat`<a name="en-us_TOPIC_0000001506334741"></a>

## Function Description<a name="en-us_TOPIC_0000001506615033"></a>

<code>AscendIndexInt8Flat</code> stores <code>INT8</code> feature vectors and performs brute-force search.

It supports concurrent multithreaded calls. You need to set the <code>MX_INDEX_MULTITHREAD</code> environment variable to <code>1</code>, that is, <code>export MX_INDEX_MULTITHREAD=1</code>. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval uses OMP internally for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

## `AscendIndexInt8Flat`<a name="en-us_TOPIC_0000001456375168"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p43030218474"><a name="p43030218474"></a><a name="p43030218474"></a><code>AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndexInt8Flat</code>. It creates an <code>AscendIndexInt8</code> with dimension <code>dims</code>. The dimension of the vector set managed by a single <code>Index</code> is unique. It configures device-side resources according to the values in <code>config</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int dims</code>: Dimension of the feature vector set managed by <code>AscendIndexInt8</code>.</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><code>faiss::MetricType metric</code>: Distance metric type used by <code>AscendIndex</code> when performing feature vector similarity retrieval.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><code>AscendIndexInt8FlatConfig config</code>: Device-side resource configuration.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul246474615523"></a><a name="ul246474615523"></a><ul id="ul246474615523"><li><code>dims</code> ∈ {64, 128, 256, 384, 512, 768, 1024}.</li><li><code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table08035919302"></a>
<table><tbody><tr id="row280317933013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p080310943012"><a name="p080310943012"></a><a name="p080310943012"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a><code>AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></p>
</td>
</tr>
<tr id="row1880379113018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p128039983019"><a name="p128039983019"></a><a name="p128039983019"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p28031693305"><a name="p28031693305"></a><a name="p28031693305"></a>Constructor of <code>AscendIndexInt8Flat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row168031396307"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p13803993309"><a name="p13803993309"></a><a name="p13803993309"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p198038933015"><a name="p198038933015"></a><a name="p198038933015"></a><code>const faiss::IndexScalarQuantizer *index</code>: CPU-side <code>Index</code> resource.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>AscendIndexInt8FlatConfig config</code>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row1580359153014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p88038983010"><a name="p88038983010"></a><a name="p88038983010"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p138034993013"><a name="p138034993013"></a><a name="p138034993013"></a>None</p>
</td>
</tr>
<tr id="row13803119153019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p16803159103018"><a name="p16803159103018"></a><a name="p16803159103018"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p108037918306"><a name="p108037918306"></a><a name="p108037918306"></a>None</p>
</td>
</tr>
<tr id="row78038912309"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p88031919307"><a name="p88031919307"></a><a name="p88031919307"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. It must be a pointer of the <code>faiss::IndexScalarQuantizer</code> type generated by the <code>copyTo</code> interface of <code>AscendIndexInt8Flat</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table11312020103012"></a>
<table><tbody><tr id="row18131520123011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p51314208305"><a name="p51314208305"></a><a name="p51314208305"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p134321816154815"><a name="p134321816154815"></a><a name="p134321816154815"></a><code>AscendIndexInt8Flat(const faiss::IndexIDMap *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</code></p>
</td>
</tr>
<tr id="row14131152033015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p171311220173012"><a name="p171311220173012"></a><a name="p171311220173012"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16131202010306"><a name="p16131202010306"></a><a name="p16131202010306"></a>Constructor of <code>AscendIndexInt8Flat</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row213118206301"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p613111201309"><a name="p613111201309"></a><a name="p613111201309"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1131192014300"><a name="p1131192014300"></a><a name="p1131192014300"></a><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</p>
<p id="p01311220103014"><a name="p01311220103014"></a><a name="p01311220103014"></a><code>AscendIndexInt8FlatConfig config</code>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="row1113242019308"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14132132015303"><a name="p14132132015303"></a><a name="p14132132015303"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8132220193011"><a name="p8132220193011"></a><a name="p8132220193011"></a>None</p>
</td>
</tr>
<tr id="row8132132093017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p91321620163010"><a name="p91321620163010"></a><a name="p91321620163010"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1132720163012"><a name="p1132720163012"></a><a name="p1132720163012"></a>None</p>
</td>
</tr>
<tr id="row12132820203018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2132142003020"><a name="p2132142003020"></a><a name="p2132142003020"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p313222019308"><a name="p313222019308"></a><a name="p313222019308"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. It must be a pointer of the <code>faiss::IndexIDMap</code> type generated by the <code>copyTo</code> interface of <code>AscendIndexInt8Flat</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table186285584308"></a>
<table><tbody><tr id="row11628358133010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p14628358193015"><a name="p14628358193015"></a><a name="p14628358193015"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a><code>AscendIndexInt8Flat(const AscendIndexInt8Flat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row1362814589304"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p5628195813011"><a name="p5628195813011"></a><a name="p5628195813011"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p96281058103020"><a name="p96281058103020"></a><a name="p96281058103020"></a>Declares this <code>Index</code> copy constructor as deleted, meaning that the type is non-copyable.</p>
</td>
</tr>
<tr id="row56281058123019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1962825833011"><a name="p1962825833011"></a><a name="p1962825833011"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndexInt8Flat&amp;</code>: Constant <code>AscendIndexInt8Flat</code>.</p>
</td>
</tr>
<tr id="row16281558103012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p36281758163011"><a name="p36281758163011"></a><a name="p36281758163011"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p86282058183018"><a name="p86282058183018"></a><a name="p86282058183018"></a>None</p>
</td>
</tr>
<tr id="row6628175820307"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p662825863011"><a name="p662825863011"></a><a name="p662825863011"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p6628658183012"><a name="p6628658183012"></a><a name="p6628658183012"></a>None</p>
</td>
</tr>
<tr id="row12628958103010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p6628195853016"><a name="p6628195853016"></a><a name="p6628195853016"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p962855810301"><a name="p962855810301"></a><a name="p962855810301"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table206471151315"></a>
<table><tbody><tr id="row564841517316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p464801510315"><a name="p464801510315"></a><a name="p464801510315"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndexInt8Flat();</code></p>
</td>
</tr>
<tr id="row156481015103116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1364813150319"><a name="p1364813150319"></a><a name="p1364813150319"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1064810155319"><a name="p1064810155319"></a><a name="p1064810155319"></a>Destructor of <code>AscendIndexInt8Flat</code>. It destroys the <code>AscendIndexInt8Flat</code> object and releases resources.</p>
</td>
</tr>
<tr id="row1564851515314"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1648181512311"><a name="p1648181512311"></a><a name="p1648181512311"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
</td>
</tr>
<tr id="row156481215103116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1264891513116"><a name="p1264891513116"></a><a name="p1264891513116"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p7648121583112"><a name="p7648121583112"></a><a name="p7648121583112"></a>None</p>
</td>
</tr>
<tr id="row564812154316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p7648111593118"><a name="p7648111593118"></a><a name="p7648111593118"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p064841512313"><a name="p064841512313"></a><a name="p064841512313"></a>None</p>
</td>
</tr>
<tr id="row3648915103115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p86481115153115"><a name="p86481115153115"></a><a name="p86481115153115"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1364861518319"><a name="p1364861518319"></a><a name="p1364861518319"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `copyFrom`<a name="en-us_TOPIC_0000001456375340"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyFrom(const faiss::IndexIDMap* index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies an existing <code>index</code> to Ascend based on <code>AscendIndexInt8Flat</code>, and keeps the original device-side resource configuration of <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>const faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The dimension <code>d</code> parameter of the member index of this <code>Index</code> must be in the range {64, 128, 256, 384, 512, 768, 1024}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</p>
</td>
</tr>
</tbody>
</table>

<a name="table862731073217"></a>
<table><tbody><tr id="row562716101326"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p6627201073216"><a name="p6627201073216"></a><a name="p6627201073216"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6908132519515"><a name="p6908132519515"></a><a name="p6908132519515"></a><code>void copyFrom(const faiss::IndexScalarQuantizer* index);</code></p>
</td>
</tr>
<tr id="row1562701093213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1962719100320"><a name="p1962719100320"></a><a name="p1962719100320"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16271610153210"><a name="p16271610153210"></a><a name="p16271610153210"></a>Copies an existing <code>index</code> to Ascend based on <code>AscendIndexInt8Flat</code>, and keeps the original device-side resource configuration of <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row6627181014329"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p20627610203213"><a name="p20627610203213"></a><a name="p20627610203213"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p126277101327"><a name="p126277101327"></a><a name="p126277101327"></a><code>const faiss::IndexScalarQuantizer* index</code>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row362771017325"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p262713107326"><a name="p262713107326"></a><a name="p262713107326"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1762761014326"><a name="p1762761014326"></a><a name="p1762761014326"></a>None</p>
</td>
</tr>
<tr id="row18627710123214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p15627310183210"><a name="p15627310183210"></a><a name="p15627310183210"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1362715109324"><a name="p1362715109324"></a><a name="p1362715109324"></a>None</p>
</td>
</tr>
<tr id="row18627510133211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p26271100323"><a name="p26271100323"></a><a name="p26271100323"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1562781003214"><a name="p1562781003214"></a><a name="p1562781003214"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The dimension <code>d</code> parameter of the <code>Index</code> must be in the range {64, 128, 256, 384, 512, 768, 1024}. The total number of base vectors must satisfy <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be one of {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</p>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000001506334805"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyTo(faiss::IndexScalarQuantizer* index) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies the retrieval resources of <code>AscendIndexInt8Flat</code> to the CPU side.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>faiss::IndexScalarQuantizer* index</code>: CPU-side <code>Index</code> resource.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The resources occupied by <code>Index</code> are freed by the user.</p>
</td>
</tr>
</tbody>
</table>

<a name="table1981952413329"></a>
<table><tbody><tr id="row6819122423218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9819112423211"><a name="p9819112423211"></a><a name="p9819112423211"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1981912463211"><a name="p1981912463211"></a><a name="p1981912463211"></a><code>void copyTo(faiss::IndexIDMap* index) const;</code></p>
</td>
</tr>
<tr id="row128191424163217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p16819224173215"><a name="p16819224173215"></a><a name="p16819224173215"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p28192024163218"><a name="p28192024163218"></a><a name="p28192024163218"></a>Copies the retrieval resources of <code>AscendIndexInt8Flat</code> to the CPU side.</p>
</td>
</tr>
<tr id="row1281910243329"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p188196241328"><a name="p188196241328"></a><a name="p188196241328"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p281916241321"><a name="p281916241321"></a><a name="p281916241321"></a><code>faiss::IndexIDMap *index</code>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="row2819182413219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p281972453215"><a name="p281972453215"></a><a name="p281972453215"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6819192473220"><a name="p6819192473220"></a><a name="p6819192473220"></a>None</p>
</td>
</tr>
<tr id="row14819152483212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p4819024113219"><a name="p4819024113219"></a><a name="p4819024113219"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p178191624133216"><a name="p178191624133216"></a><a name="p178191624133216"></a>None</p>
</td>
</tr>
<tr id="row381919240326"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p78191224173213"><a name="p78191224173213"></a><a name="p78191224173213"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4819132473214"><a name="p4819132473214"></a><a name="p4819132473214"></a><code>index</code> must be a valid <code>IndexIDMap</code> pointer. The resources occupied by <code>Index</code> are freed by the user.</p>
</td>
</tr>
</tbody>
</table>

## `getBase`<a name="en-us_TOPIC_0000001506334753"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void getBase(int deviceId, std::vector&lt;int8_t&gt; &amp;xb) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the feature vectors managed by this <code>AscendIndexInt8Flat</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int deviceId</code>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><code>std::vector&lt;int8_t&gt; &amp;xb</code>: Base feature vectors stored by <code>AscendIndexInt8Flat</code> on <code>deviceId</code>.</p>
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

## `getBaseSize`<a name="en-us_TOPIC_0000001506414709"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>size_t getBaseSize(int deviceId) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the number of feature vectors managed by this <code>AscendIndexInt8Flat</code> on the specified <code>deviceId</code>.</p>
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

## `getIdxMap`<a name="en-us_TOPIC_0000001506495853"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Gets the feature vector IDs managed by this <code>AscendIndexInt8Flat</code> on the specified <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int deviceId</code>: Device-side device ID.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><code>std::vector&lt;idx_t&gt; &amp;idxMap</code>: Base feature vector IDs stored by <code>AscendIndexInt8Flat</code> on <code>deviceId</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>deviceId</code> must be a valid device ID.</p>
</td>
</tr>
</tbody>
</table>

## `operator =`<a name="en-us_TOPIC_0000001506414909"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8Flat&amp; operator=(const AscendIndexInt8Flat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares this <code>Index</code> assignment operator as deleted, meaning that the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndexInt8Flat&amp;</code>: Constant <code>AscendIndexInt8Flat</code>.</p>
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

## `reset`<a name="en-us_TOPIC_0000001506495889"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void reset();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Clears the base vectors in this <code>AscendIndexInt8Flat</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a>None</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1911619471633"><a name="p1911619471633"></a><a name="p1911619471633"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `search_with_masks`<a name="en-us_TOPIC_0000001456694912"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void search_with_masks(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Implements the feature vector search interface for <code>AscendIndexInt8</code>, and returns the distances and IDs of the <code>k</code> most similar features based on the input feature vectors and the <code>mask</code>. The mask is a <code>0</code>/<code>1</code> bit string. Each bit indicates whether the corresponding feature in the base vector set participates in distance computation. <code>1</code> means participate, and <code>0</code> means not participate.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><code>idx_t n</code>: Number of query feature vectors.</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><code>const int8_t* x</code>: Feature vector data.</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p1838795616530"><a name="p1838795616530"></a><a name="p1838795616530"></a><code>const void* mask</code>: Base vector set filter mask.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul197738222579"></a><a name="ul197738222579"></a><ul id="ul197738222579"><li><code>n</code> must be greater than <code>0</code> and less than <code>1e9</code>.</li><li><code>k</code> is usually not allowed to exceed <code>4096</code>.</li><li><code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers, and their lengths should be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>mask</code> must be a non-null pointer, and the length of the passed mask must be <code>⌈ntotal / 8⌉ * n</code> (<code>ntotal</code> is the number of vectors in the base vector set).</li><li>The mask is set in the order of the base vector set. If <code>remove_ids</code> is called before this interface, the order of base vectors changes. Therefore, call <code>getIdxMap</code> first to obtain the IDs of the base vectors, and then set the mask.</li><li>This interface requires the base vector set to be stored on a single device. Otherwise, the filtering result may be incorrect.</li></ul>
</td>
</tr>
</tbody>
</table>

## `setPageSize`<a name="en-us_TOPIC_0000002007453769"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p20912871594"><a name="p20912871594"></a><a name="p20912871594"></a><code>void setPageSize(uint16_t pageBlockNum);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p393922219912"><a name="p393922219912"></a><a name="p393922219912"></a>Sets the number of base-vector blocks that this <code>AscendIndexInt8Flat</code> computes consecutively in one <code>search</code> call.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1836319401096"><a name="p1836319401096"></a><a name="p1836319401096"></a><code>uint16_t pageBlockNum</code>: Number of base-vector blocks to compute consecutively in one call. If you do not set this parameter, the default is to compute 16 blocks consecutively at a time. The size of one block is determined by <code>blockSize</code> in <code>AscendIndexInt8FlatConfig</code>. The larger the value, the more memory <code>search</code> uses.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul250991973317"></a><a name="ul250991973317"></a><ul id="ul250991973317"><li><code>pageBlockNum</code> must be greater than <code>0</code> and less than or equal to <code>144</code>.</li><li>This interface is mainly used for large base vector set scenarios and for performance tuning of the <code>search</code> interface. The larger the value, the more preallocated memory configured by <code>resourceSize</code> in <code>AscendIndexInt8FlatConfig</code> it consumes. You are advised to request enough preallocated memory first and then use this interface to tune parameters.</li></ul>
</td>
</tr>
</tbody>
</table>
