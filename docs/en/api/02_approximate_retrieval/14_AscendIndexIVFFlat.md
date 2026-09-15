# `AscendIndexIVFFlat`<a name="en-us_TOPIC_0000002478095516"></a>

## Function Description<a name="en-us_TOPIC_0000002510095475"></a>

`AscendIndexIVFFlat` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports only IP distance.

## `AscendIndexIVFFlat`<a name="en-us_TOPIC_0000002509975505"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1857144175420"><a name="p1857144175420"></a><a name="p1857144175420"></a><code>AscendIndexIVFFlat(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFFlatConfig config)</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor for <code>AscendIndexIVFFlat</code>, which creates a retrieval <code>Index</code> on Ascend.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1580317419509"><a name="b1580317419509"></a><a name="b1580317419509"></a><code>int dims</code></strong>: Dimension of the base library retrieval vectors.</p>
<p id="p9902601825"><a name="p9902601825"></a><a name="p9902601825"></a><strong id="b19494101811220"><a name="b19494101811220"></a><a name="b19494101811220"></a><code>faiss::MetricType metric</code></strong>: Distance type. Currently only <code>faiss::METRIC_INNER_PRODUCT</code> is supported.</p>
<p id="p15757141212318"><a name="p15757141212318"></a><a name="p15757141212318"></a><strong id="b1966283819616"><a name="b1966283819616"></a><a name="b1966283819616"></a><code>int nlist</code></strong>: Number of IVF buckets.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1191265511617"><a name="b1191265511617"></a><a name="b1191265511617"></a><code>AscendIndexIVFFlatConfig config</code></strong>: Device-side resource configuration.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14739204217152"><a name="p14739204217152"></a><a name="p14739204217152"></a><code>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row186417021110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p664405110"><a name="p664405110"></a><a name="p664405110"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p76470121111"><a name="p76470121111"></a><a name="p76470121111"></a>Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</p>
</td>
</tr>
<tr id="row964505113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2642019118"><a name="p2642019118"></a><a name="p2642019118"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b3738139972"><a name="b3738139972"></a><a name="b3738139972"></a><code>const AscendIndexIVFFlat&amp;</code></strong>: Constant <code>AscendIndexIVFFlat</code>.</p>
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

## `~AscendIndexIVFFlat`<a name="en-us_TOPIC_0000002477935546"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19588544116"><a name="p19588544116"></a><a name="p19588544116"></a><code>~AscendIndexIVFFlat()</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Destructor of <code>AscendIndexIVFFlat</code>, which destroys the <code>AscendIndexIVFFlat</code> object and releases resources.</p>
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

## `operator=`<a name="en-us_TOPIC_0000002484264062"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a><code>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the copy assignment operator of this <code>Index</code> as deleted, making the type non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b142971612074"><a name="b142971612074"></a><a name="b142971612074"></a><code>const AscendIndexIVFFlat&amp;</code></strong>: Constant <code>AscendIndexIVFFlat</code>.</p>
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

## `train`<a name="en-us_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a><code>void train(idx_t n, const float *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Trains <code>AscendIndexIVFFlat</code>, inheriting the relevant APIs from <code>AscendIndex</code> and providing a concrete implementation.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>Training uses k-means clustering. A relatively small training set may affect retrieval accuracy.</li><li>The valid range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>.</li><li>The pointer <code>x</code> must be non-null, and its length must be <code>dims * n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>Currently only CPU clustering is supported. The <code>useKmeansPP</code> parameter cannot be set to <code>true</code>.</li></ul>
</td>
</tr>
</tbody>
</table>
