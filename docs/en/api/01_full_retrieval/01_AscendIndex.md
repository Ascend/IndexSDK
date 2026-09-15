# `AscendIndex`<a id="en-us_TOPIC_0000001456375304"></a>

## Overview<a name="en-us_TOPIC_0000001506414937"></a>

<code>AscendIndex</code> is the base class of the <code>Index</code> implementations for most retrieval methods in the feature retrieval component. It sits on top of Faiss and defines interfaces for the other indexes in feature retrieval.

## `add`<a id="en-us_TOPIC_0000001506614985"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p5684112753414"><a name="p5684112753414"></a><a name="p5684112753414"></a><code>void add(idx_t n, const float *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Implements <code>AscendIndex</code> index creation and adds new feature vectors to the base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>const float *x</code>: Feature vectors to add to the base library.</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p14372206704"><a name="p14372206704"></a><a name="p14372206704"></a>The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</p>
<p id="p967614571013"><a name="p967614571013"></a><a name="p967614571013"></a><code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>.</p>
<div class="note" id="note153615513612"><a name="note153615513612"></a><a name="note153615513612"></a><span class="notetitle">
Note: </span><div class="notebody"><a name="ul103685518369"></a><a name="ul103685518369"></a><ul id="ul103685518369"><li>The <code>add</code> interface cannot be used together with the <code>add_with_ids</code> interface.</li><li>After you use the <code>add</code> interface, the <code>labels</code> in the search results may repeat. If your service has requirements for labels, you are advised to use the <code>add_with_ids</code> interface.</li></ul>
</div></div>
</td>
</tr>
</tbody>
</table>

<a name="table17254342193617"></a>
<table><tbody><tr id="row1254164217362"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p225474203614"><a name="p225474203614"></a><a name="p225474203614"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p23521745171019"><a name="p23521745171019"></a><a name="p23521745171019"></a><code>void add(idx_t n, const uint16_t *x);</code></p>
</td>
</tr>
<tr id="row18254442183618"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p22541442163617"><a name="p22541442163617"></a><a name="p22541442163617"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p4352184531013"><a name="p4352184531013"></a><a name="p4352184531013"></a>Implements <code>AscendIndex</code> index creation and adds new feature vectors to the base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</p>
</td>
</tr>
<tr id="row7254184215362"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p1025415425363"><a name="p1025415425363"></a><a name="p1025415425363"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p9352184551012"><a name="p9352184551012"></a><a name="p9352184551012"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p1935220453102"><a name="p1935220453102"></a><a name="p1935220453102"></a><code>const uint16_t *x</code>: Feature vectors to add to the base library.</p>
</td>
</tr>
<tr id="row5254194273613"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p19254542133611"><a name="p19254542133611"></a><a name="p19254542133611"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p14254164213614"><a name="p14254164213614"></a><a name="p14254164213614"></a>None</p>
</td>
</tr>
<tr id="row182547427362"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p102541942173615"><a name="p102541942173615"></a><a name="p102541942173615"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p152541742103611"><a name="p152541742103611"></a><a name="p152541742103611"></a>None</p>
</td>
</tr>
<tr id="row425404212368"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p19254204218367"><a name="p19254204218367"></a><a name="p19254204218367"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1457312121112"><a name="p1457312121112"></a><a name="p1457312121112"></a>The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</p>
<p id="p19688575103"><a name="p19688575103"></a><a name="p19688575103"></a><code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>.</p>
</td>
</tr>
</tbody>
</table>

## `add_with_ids`<a id="en-us_TOPIC_0000001456694864"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Implements <code>AscendIndex</code> index creation and adds new feature vectors to the base library, with an ID for each base-library feature.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>const float *x</code>: Feature vectors to add to the base library.</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library.</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul733045716013"></a><a name="ul733045716013"></a><ul id="ul733045716013"><li>The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>.</li><li>When the <code>filterable</code> filter switch is set to <code>true</code>, ensure that the timestamps in <code>ids</code> are positive.<p id="p94232041214"><a name="p94232041214"></a><a name="p94232041214"></a><code>ids</code> (of type <code>uint64_t</code>) contains <code>timestamp</code> (of type <code>int32_t</code>) and <code>cid</code> (camera ID, of type <code>uint8_t</code>), as shown below:</p>
<pre class="screen" id="screen2086011148112"><a name="screen2086011148112"></a><a name="screen2086011148112"></a>-----| cid | timestamp | -----
 14  |  8  |    32     |  10</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table562574920111"></a>
<table><tbody><tr id="row176667494111"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.1.1"><p id="p466617492115"><a name="p466617492115"></a><a name="p466617492115"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.1.1 "><p id="p866615493118"><a name="p866615493118"></a><a name="p866615493118"></a><code>void add_with_ids(idx_t n, const uint16_t *x, const idx_t *ids);</code></p>
</td>
</tr>
<tr id="row7666184961113"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.2.1"><p id="p66661749191116"><a name="p66661749191116"></a><a name="p66661749191116"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p46661497116"><a name="p46661497116"></a><a name="p46661497116"></a>Implements <code>AscendIndex</code> index creation and adds new feature vectors to the base library, with an ID for each base-library feature.</p>
</td>
</tr>
<tr id="row17666649161114"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.3.1"><p id="p11666124917117"><a name="p11666124917117"></a><a name="p11666124917117"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.3.1 "><p id="p15666164911115"><a name="p15666164911115"></a><a name="p15666164911115"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p566644951113"><a name="p566644951113"></a><a name="p566644951113"></a><code>const uint16_t *x</code>: Feature vectors to add to the base library.</p>
<p id="p18666249141110"><a name="p18666249141110"></a><a name="p18666249141110"></a><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library.</p>
</td>
</tr>
<tr id="row14666144911116"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.4.1"><p id="p1166674921110"><a name="p1166674921110"></a><a name="p1166674921110"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.4.1 "><p id="p196661949181117"><a name="p196661949181117"></a><a name="p196661949181117"></a>None</p>
</td>
</tr>
<tr id="row4666449191111"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.5.1"><p id="p7666749101110"><a name="p7666749101110"></a><a name="p7666749101110"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.5.1 "><p id="p1766784912116"><a name="p1766784912116"></a><a name="p1766784912116"></a>None</p>
</td>
</tr>
<tr id="row86671349131119"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.6.1"><p id="p1266714991113"><a name="p1266714991113"></a><a name="p1266714991113"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><a name="ul1215264517123"></a><a name="ul1215264517123"></a><ul id="ul1215264517123"><li>The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>.</li><li>When the <code>filterable</code> filter switch is set to <code>true</code>, ensure that the timestamps in <code>ids</code> are positive. <code>ids</code> (of type <code>uint64_t</code>) contains <code>timestamp</code> (of type <code>int32_t</code>) and <code>cid</code> (camera ID, of type <code>uint8_t</code>), as shown below:<a name="screen11981113915128"></a><a name="screen11981113915128"></a><pre class="screen" codetype="ColdFusion" id="screen11981113915128">-----| cid | timestamp | -----
 14  |  8  |    32     |  10</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndex`<a name="en-us_TOPIC_0000001456695048"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>AscendIndex(int dims, faiss::MetricType metric, AscendIndexConfig config)</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndex</code>. It creates an <code>AscendIndex</code> with dimension <code>dims</code>. A single index manages vectors with one fixed dimension. Device-side resources are set according to the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndex</code>.</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndex</code> when performing feature-vector similarity retrieval. Currently supported values are <code>faiss::MetricType::METRIC_L2</code> and <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>AscendIndexConfig config</code>: Device-side resource configuration.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>dims</code> must be an integer in the range <code>(0, 4096]</code> and must be divisible by <code>16</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table161511529133912"></a>
<table><tbody><tr id="row1615110293394"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2151429113910"><a name="p2151429113910"></a><a name="p2151429113910"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15151152943916"><a name="p15151152943916"></a><a name="p15151152943916"></a><code>AscendIndex(const AscendIndex&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row51517295398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p21514294391"><a name="p21514294391"></a><a name="p21514294391"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015122918399"><a name="p2015122918399"></a><a name="p2015122918399"></a>Declares the copy constructor of <code>AscendIndex</code> as deleted. Therefore, <code>AscendIndex</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row815120292398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7151122933917"><a name="p7151122933917"></a><a name="p7151122933917"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndex&amp;</code>: Constant <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row18151172918399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p615182993916"><a name="p615182993916"></a><a name="p615182993916"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8151329143914"><a name="p8151329143914"></a><a name="p8151329143914"></a>None</p>
</td>
</tr>
<tr id="row171511295399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p17151192917392"><a name="p17151192917392"></a><a name="p17151192917392"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16151122914394"><a name="p16151122914394"></a><a name="p16151122914394"></a>None</p>
</td>
</tr>
<tr id="row12151829153910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p615192973914"><a name="p615192973914"></a><a name="p615192973914"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15151929163918"><a name="p15151929163918"></a><a name="p15151929163918"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table62621513124018"></a>
<table><tbody><tr id="row726218134408"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1726212134400"><a name="p1726212134400"></a><a name="p1726212134400"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndex();</code></p>
</td>
</tr>
<tr id="row1926221314401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1926218134408"><a name="p1926218134408"></a><a name="p1926218134408"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82621213184020"><a name="p82621213184020"></a><a name="p82621213184020"></a>Destructor of <code>AscendIndex</code>. It destroys the <code>AscendIndex</code> object and releases resources.</p>
</td>
</tr>
<tr id="row15262213104015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p826221314402"><a name="p826221314402"></a><a name="p826221314402"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
</td>
</tr>
<tr id="row1726271324017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p5262213154014"><a name="p5262213154014"></a><a name="p5262213154014"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16262131311400"><a name="p16262131311400"></a><a name="p16262131311400"></a>None</p>
</td>
</tr>
<tr id="row0262121324020"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p8262191319409"><a name="p8262191319409"></a><a name="p8262191319409"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p726201319407"><a name="p726201319407"></a><a name="p726201319407"></a>None</p>
</td>
</tr>
<tr id="row526241324016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p526201310404"><a name="p526201310404"></a><a name="p526201310404"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15262111319403"><a name="p15262111319403"></a><a name="p15262111319403"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getDeviceList`<a name="en-us_TOPIC_0000001506495857"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a><code>std::vector&lt;int&gt; getDeviceList();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>Returns the device-side Ascend AI Processor configuration managed in the index. Derived classes provide the implementation. This class does not provide one and returns only an empty <code>vector&lt;int&gt;</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Device-side Ascend AI Processor configuration managed in the index.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `operator=`<a name="en-us_TOPIC_0000001506334661"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11253135664517"><a name="p11253135664517"></a><a name="p11253135664517"></a><code>AscendIndex&amp; operator=(const AscendIndex&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the copy assignment operator of <code>AscendIndex</code> as deleted. Therefore, <code>AscendIndex</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndex&amp;</code>: Constant <code>AscendIndex</code>.</p>
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

## `reclaimMemory`<a name="en-us_TOPIC_0000001456695092"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a><code>virtual size_t reclaimMemory();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>Reduces the memory occupied by the base library without changing the number of vectors in it. The implementation is inherited and provided by derived classes. This class does not provide an implementation.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Size of the reclaimed memory, in bytes.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `remove_ids`<a name="en-us_TOPIC_0000001456535000"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Removes the specified feature vectors from the base library in <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to remove. For details about usage and definition, see the corresponding Faiss documentation.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Number of removed feature vectors.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `reserveMemory`<a name="en-us_TOPIC_0000001456375348"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a><code>virtual void reserveMemory(size_t numVecs);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p169016017519"><a name="p169016017519"></a><a name="p169016017519"></a>Abstract interface for reserving memory for the base library before it is built. The implementation is inherited and provided by derived classes. This class does not provide an implementation.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a><code>size_t numVecs</code>: Number of vectors in the base library for which to reserve memory.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>None</p>
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

## `reset`<a name="en-us_TOPIC_0000001506414901"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void reset() override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Clears the base-library vectors of this <code>AscendIndex</code>.</p>
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

## `search`<a name="en-us_TOPIC_0000001506334641"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.1.1 "><p id="p8820054142218"><a name="p8820054142218"></a><a name="p8820054142218"></a><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Feature-vector retrieval interface. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><code>idx_t n</code>: Number of query feature vectors.</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><code>const float *x</code>: Feature-vector data.</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p179121029182319"><a name="p179121029182319"></a><a name="p179121029182319"></a><code>const SearchParameters *params</code>: Optional Faiss parameter. The default value is <code>nullptr</code>, and this parameter is currently not supported.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid distances with 65504 or -65504, depending on the metric.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid labels with -1.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.6.1 "><p id="p13601965223"><a name="p13601965223"></a><a name="p13601965223"></a>The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096.</p>
</td>
</tr>
<tr id="row68701915173311"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.7.1"><p id="p11871615173310"><a name="p11871615173310"></a><a name="p11871615173310"></a>Note</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.7.1 "><p id="p68716152338"><a name="p68716152338"></a><a name="p68716152338"></a>In scenarios that use the brute-force algorithm for small-base libraries, if performance drops when the base library and batch size are large, increase the <code>resources</code> parameter in <code>AscendIndexConfig</code>. The default value for the brute-force algorithm is <code>128</code> MB.</p>
</td>
</tr>
</tbody>
</table>

<a name="table03178548130"></a>
<table><tbody><tr id="row133713545133"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.1.1"><p id="p0371145411316"><a name="p0371145411316"></a><a name="p0371145411316"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.1.1 "><p id="p1737125421319"><a name="p1737125421319"></a><a name="p1737125421319"></a><code>void search(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels) const;</code></p>
</td>
</tr>
<tr id="row93719547138"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.2.1"><p id="p3371165419130"><a name="p3371165419130"></a><a name="p3371165419130"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p537135414133"><a name="p537135414133"></a><a name="p537135414133"></a>Feature vector retrieval interface. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
</td>
</tr>
<tr id="row537295491313"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.3.1"><p id="p1137213548130"><a name="p1137213548130"></a><a name="p1137213548130"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.3.1 "><p id="p1437219546136"><a name="p1437219546136"></a><a name="p1437219546136"></a><code>idx_t n</code>: Number of query feature vectors.</p>
<p id="p0372135401314"><a name="p0372135401314"></a><a name="p0372135401314"></a><code>const uint16_t *x</code>: Feature-vector data.</p>
<p id="p13372205419135"><a name="p13372205419135"></a><a name="p13372205419135"></a><code>idx_t k</code>: Number of most similar results to return.</p>
</td>
</tr>
<tr id="row13721254131312"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.4.1"><p id="p11372254101315"><a name="p11372254101315"></a><a name="p11372254101315"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.4.1 "><p id="p10372165412137"><a name="p10372165412137"></a><a name="p10372165412137"></a><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid distances with 65504 or -65504, depending on the metric.</p>
<p id="p53727546138"><a name="p53727546138"></a><a name="p53727546138"></a><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query. When the number of valid retrieval results is fewer than <code>k</code>, fill the remaining invalid labels with -1.</p>
</td>
</tr>
<tr id="row43722544139"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.5.1"><p id="p937225491317"><a name="p937225491317"></a><a name="p937225491317"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.5.1 "><p id="p1337275461310"><a name="p1337275461310"></a><a name="p1337275461310"></a>None</p>
</td>
</tr>
<tr id="row15372954111319"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.6.1"><p id="p437220547131"><a name="p437220547131"></a><a name="p437220547131"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><p id="p337255411313"><a name="p337255411313"></a><a name="p337255411313"></a>The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>n</code> must be in the range <code>0 &lt; n &lt; 1e9</code>. <code>k</code> is usually not allowed to exceed 4096.</p>
</td>
</tr>
<tr id="row19372135418134"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.7.1"><p id="p7372155411136"><a name="p7372155411136"></a><a name="p7372155411136"></a>Note</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.7.1 "><p id="p15372105431319"><a name="p15372105431319"></a><a name="p15372105431319"></a>In scenarios that use the small-base-library brute-force algorithm, if performance drops when the base library and batch size are large, increase the <code>resources</code> parameter in <code>AscendIndexConfig</code>. The default value of the brute-force algorithm is 128 MB.</p>
</td>
</tr>
</tbody>
</table>
