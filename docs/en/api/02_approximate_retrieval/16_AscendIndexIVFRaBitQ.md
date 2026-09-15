# `AscendIndexIVFRaBitQ`<a name="en-us_TOPIC_0000002513157720"></a>

## Function Description<a name="en-us_TOPIC_0000002544797635"></a>

`AscendIndexIVFRaBitQ` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports L2 distance computation.

## `AscendIndexIVFRaBitQ`<a name="en-us_TOPIC_0000002513317654"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1857144175420"><a name="p1857144175420"></a><a name="p1857144175420"></a><code>AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFRaBitQConfig config)</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor for <code>AscendIndexIVFRaBitQ</code>, which creates a retrieval <code>Index</code> on Ascend.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1580317419509"><a name="b1580317419509"></a><a name="b1580317419509"></a><code>int dims</code></strong>: Dimension of the base library retrieval vectors.</p>
<p id="p9902601825"><a name="p9902601825"></a><a name="p9902601825"></a><strong id="b19494101811220"><a name="b19494101811220"></a><a name="b19494101811220"></a><code>faiss::MetricType metric</code></strong>: Distance type. Supports <code>faiss::METRIC_L2</code> and <code>faiss::METRIC_IP</code>.</p>
<p id="p15757141212318"><a name="p15757141212318"></a><a name="p15757141212318"></a><strong id="b1966283819616"><a name="b1966283819616"></a><a name="b1966283819616"></a><code>int nlist</code></strong>: Number of IVF buckets.</p>
<p id="p54901733102914"><a name="p54901733102914"></a><a name="p54901733102914"></a><strong id="b1191265511617"><a name="b1191265511617"></a><a name="b1191265511617"></a><code>AscendIndexIVFRaBitQConfig config</code></strong>: Device-side resource configuration.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul79274019592"></a><a name="ul79274019592"></a><ul id="ul79274019592"><li><code>dims</code> currently supports only <code>128</code>.</li><li><code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table663150151113"></a>
<table><tbody><tr id="row176440181111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p064509114"><a name="p064509114"></a><a name="p064509114"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14739204217152"><a name="p14739204217152"></a><a name="p14739204217152"></a><code>AscendIndexIVFRaBitQ& operator=(const AscendIndexIVFRaBitQ&) = delete;</code></p>
</td>
</tr>
<tr id="row186417021110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p664405110"><a name="p664405110"></a><a name="p664405110"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p76470121111"><a name="p76470121111"></a><a name="p76470121111"></a>Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</p>
</td>
</tr>
<tr id="row964505113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2642019118"><a name="p2642019118"></a><a name="p2642019118"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b3738139972"><a name="b3738139972"></a><a name="b3738139972"></a><code>const AscendIndexIVFRaBitQ&</code></strong>: Constant <code>AscendIndexIVFRaBitQ</code>.</p>
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

## `~AscendIndexIVFRaBitQ`<a name="en-us_TOPIC_0000002544837623"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19588544116"><a name="p19588544116"></a><a name="p19588544116"></a><code>~AscendIndexIVFRaBitQ()</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Destructor of <code>AscendIndexIVFRaBitQ</code>, which destroys the <code>AscendIndexIVFRaBitQ</code> object and releases resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
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

## `operator=`<a name="en-us_TOPIC_0000002513157724"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a><code>AscendIndexIVFRaBitQ& operator=(const AscendIndexIVFRaBitQ&) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b142971612074"><a name="b142971612074"></a><a name="b142971612074"></a><code>const AscendIndexIVFRaBitQ&</code></strong>: Constant <code>AscendIndexIVFRaBitQ</code>.</p>
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

## `train`<a name="en-us_TOPIC_0000002544797639"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a><code>void train(idx_t n, const float *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Trains <code>AscendIndexIVFRaBitQ</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>Training uses k-means clustering. A relatively small training set may affect retrieval accuracy.</li><li>The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>.</li><li>The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>Setting <code>useKmeansPP</code> to <code>true</code> enables NPU clustering; otherwise CPU clustering is used. For precision issues, see <a href="../../07_faq.md#floating-point-computation-precision-issues">Floating-Point Computation Precision Issues</a>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `remove_ids`<a name="en-us_TOPIC_0000002513157728"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p159810144013"><a name="p159810144013"></a><a name="p159810144013"></a><code>void remove_ids(size_t n, const idx_t* ids);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Removes the trained vectors in <code>AscendIndexIVFRaBitQ</code> corresponding to the provided index IDs, by calling the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b351832435710"><a name="b351832435710"></a><a name="b351832435710"></a><code>size_t n</code></strong>: Number of feature vectors to delete.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b17199113075712"><a name="b17199113075712"></a><a name="b17199113075712"></a><code>const idx_t *ids</code></strong>: IDs of the feature vectors to delete.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>.</li><li>The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `copyFrom`<a name="en-us_TOPIC_0000002557609263"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1644393118420"><a name="p1644393118420"></a><a name="p1644393118420"></a><code>void copyFrom(const faiss::IndexIVFRaBitQ *index)</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Provides a CPU-side <code>IndexIVFRaBitQ</code> index, loads data from the trained index to the Device side for subsequent retrieval, and calls the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b17199113075712"><a name="b17199113075712"></a><a name="b17199113075712"></a><code>const faiss::IndexIVFRaBitQ *index</code></strong>: Trained CPU-side <code>IndexIVFRaBitQ</code> index.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>The pointer <code>index</code> must be non-null, and it must point to a trained <code>IndexIVFRaBitQ</code> index.</li><li>Before calling this API to read data, configure <code>AscendIndexIVFRaBitQConfig</code> and create an <code>AscendIndexIVFRaBitQ</code> object according to the normal procedure.</li></ul>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000002557689209"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1644393118420"><a name="p1644393118420"></a><a name="p1644393118420"></a><code>void copyTo(faiss::IndexIVFRaBitQ *index) const</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Provides a CPU-side <code>IndexIVFRaBitQ</code> index, downloads the trained data from the Device side into the CPU index for persistence, and calls the relevant APIs in <code>AscendIndexIVFRaBitQImpl</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b17199113075712"><a name="b17199113075712"></a><a name="b17199113075712"></a><code>faiss::IndexIVFRaBitQ *index</code></strong>: CPU-side <code>IndexIVFRaBitQ</code> index.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>The pointer <code>index</code> must be non-null.</li><li>Before calling this API to persist data, create an <code>AscendIndexIVFRaBitQ</code> object and train it into the index according to the normal procedure.</li></ul>
</td>
</tr>
</tbody>
</table>

## `update`<a name="en-us_TOPIC_0000002566242121"></a>

<a name="table962730101715"></a>
<table><tbody><tr id="row12622305178"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p8621330141716"><a name="p8621330141716"></a><a name="p8621330141716"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p8222411311"><a name="p8222411311"></a><a name="p8222411311"></a><code>std::vector&lt;idx_t&gt; update(idx_t n, float* x, idx_t* ids)</code></p>
</td>
</tr>
<tr id="row14621830121715"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p186218309171"><a name="p186218309171"></a><a name="p186218309171"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p662143019176"><a name="p662143019176"></a><a name="p662143019176"></a>Batch-updates the vectors in the <code>AscendIndexIVFRaBitQ</code> base library corresponding to <code>ids</code> to <code>x</code>. IDs that do not exist in the base library are not updated, and the list of missing IDs is returned.</p>
</td>
</tr>
<tr id="row8629301176"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p1062113010175"><a name="p1062113010175"></a><a name="p1062113010175"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1962530201714"><a name="p1962530201714"></a><a name="p1962530201714"></a><strong id="b351832435710"><a name="b351832435710"></a><a name="b351832435710"></a><code>idx_t n</code></strong>: Number of feature vectors to update.</p>
<p id="p42931353121010"><a name="p42931353121010"></a><a name="p42931353121010"></a><strong id="b87001153171012"><a name="b87001153171012"></a><a name="b87001153171012"></a><code>float *x</code></strong>: List of feature vectors to update.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b156263017171"><a name="b156263017171"></a><a name="b156263017171"></a><code>idx_t *ids</code></strong>: List of feature vector IDs to update.</p>
</td>
</tr>
<tr id="row18621130141716"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p1762530161715"><a name="p1762530161715"></a><a name="p1762530161715"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p1462183015175"><a name="p1462183015175"></a><a name="p1462183015175"></a>None</p>
</td>
</tr>
<tr id="row1262133017174"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p06283010176"><a name="p06283010176"></a><a name="p06283010176"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p462730171716"><a name="p462730171716"></a><a name="p462730171716"></a><strong id="b354121016121"><a name="b354121016121"></a><a name="b354121016121"></a><code>std::vector&lt;idx_t&gt; noExistIds</code></strong>: Returns the list of vector IDs that do not exist.</p>
</td>
</tr>
<tr id="row062530101714"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p1562173041715"><a name="p1562173041715"></a><a name="p1562173041715"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul1662173016177"></a><a name="ul1662173016177"></a><ul id="ul1662173016177"><li>The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>.</li><li>The pointer <code>x</code> must be non-null, and its length must be <code>n * dim</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The pointer <code>ids</code> must be non-null, and its length must be <code>n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>
