# `AscendIndexFlatL2`<a name="en-us_TOPIC_0000001456375424"></a>

## Function Description<a name="en-us_TOPIC_0000001877955534"></a>

<code>AscendIndexFlatL2</code> is a brute-force feature retrieval algorithm that stores FP16 floating-point values and uses the L2 distance.

It supports multithreaded concurrent calls. You must set the <code>MX_INDEX_MULTITHREAD</code> environment variable to <code>1</code>, that is, run <code>export MX_INDEX_MULTITHREAD=1</code>. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> The <code>AscendIndexFlatL2</code> algorithm supports online operator conversion. If the environment variable <code>MX_INDEX_USE_ONLINEOP</code> is set to <code>1</code> (<code>export MX_INDEX_USE_ONLINEOP=1</code>), it converts the operators online and calls them. To use online operators, the user must explicitly call <code>(void)aclFinalize()</code> at the end of the application. The header file <code>#include "acl/acl.h"</code> is required.

## `AscendIndexFlatL2`<a name="en-us_TOPIC_0000001506495761"></a>

<a name="en-us_topic_0000001294312541_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001294312541_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001294312541_p12559123810"><a name="en-us_topic_0000001294312541_p12559123810"></a><a name="en-us_topic_0000001294312541_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001294312541_p2425655144613"><a name="en-us_topic_0000001294312541_p2425655144613"></a><a name="en-us_topic_0000001294312541_p2425655144613"></a><code>AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001294312541_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001294312541_p1212599383"><a name="en-us_topic_0000001294312541_p1212599383"></a><a name="en-us_topic_0000001294312541_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001294312541_p131714208358"><a name="en-us_topic_0000001294312541_p131714208358"></a><a name="en-us_topic_0000001294312541_p131714208358"></a>The constructor of <code>AscendIndexFlatL2</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312541_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001294312541_p112195910383"><a name="en-us_topic_0000001294312541_p112195910383"></a><a name="en-us_topic_0000001294312541_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001294312541_p874812810555"><a name="en-us_topic_0000001294312541_p874812810555"></a><a name="en-us_topic_0000001294312541_p874812810555"></a><code>faiss::IndexFlatL2 *index</code>: CPU-side <code>Index</code> resource.</p>
<p id="en-us_topic_0000001294312541_p661314244382"><a name="en-us_topic_0000001294312541_p661314244382"></a><a name="en-us_topic_0000001294312541_p661314244382"></a><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312541_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001294312541_p17235973820"><a name="en-us_topic_0000001294312541_p17235973820"></a><a name="en-us_topic_0000001294312541_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001294312541_p973225082318"><a name="en-us_topic_0000001294312541_p973225082318"></a><a name="en-us_topic_0000001294312541_p973225082318"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312541_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001294312541_p182459113812"><a name="en-us_topic_0000001294312541_p182459113812"></a><a name="en-us_topic_0000001294312541_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001294312541_p132314362521"><a name="en-us_topic_0000001294312541_p132314362521"></a><a name="en-us_topic_0000001294312541_p132314362521"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312541_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001294312541_p423590386"><a name="en-us_topic_0000001294312541_p423590386"></a><a name="en-us_topic_0000001294312541_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001294312541_p182559163813"><a name="en-us_topic_0000001294312541_p182559163813"></a><a name="en-us_topic_0000001294312541_p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The value range of the total number of base library vectors is <code>0 ≤ n &lt; 1e9</code>. The <code>metric_type</code> parameter must be <code>faiss::MetricType::METRIC_L2</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="en-us_topic_0000001294591937_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001294591937_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001294591937_p12559123810"><a name="en-us_topic_0000001294591937_p12559123810"></a><a name="en-us_topic_0000001294591937_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001294591937_p144102184422"><a name="en-us_topic_0000001294591937_p144102184422"></a><a name="en-us_topic_0000001294591937_p144102184422"></a><code>AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001294591937_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001294591937_p1212599383"><a name="en-us_topic_0000001294591937_p1212599383"></a><a name="en-us_topic_0000001294591937_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001294591937_p94401440428"><a name="en-us_topic_0000001294591937_p94401440428"></a><a name="en-us_topic_0000001294591937_p94401440428"></a>The constructor of <code>AscendIndexFlatL2</code>. It creates an <code>AscendIndexFlatL2</code> with dimension <code>dims</code>. The dimension of a vector set managed by one <code>Index</code> is unique. It then sets Device-side resources according to the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="en-us_topic_0000001294591937_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001294591937_p112195910383"><a name="en-us_topic_0000001294591937_p112195910383"></a><a name="en-us_topic_0000001294591937_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001294591937_p874812810555"><a name="en-us_topic_0000001294591937_p874812810555"></a><a name="en-us_topic_0000001294591937_p874812810555"></a><code>int dims</code>: The dimension of a set of feature vectors managed by <code>AscendIndexFlatL2</code>.</p>
<p id="en-us_topic_0000001294591937_p1220621175115"><a name="en-us_topic_0000001294591937_p1220621175115"></a><a name="en-us_topic_0000001294591937_p1220621175115"></a><code>AscendIndexFlatConfig config</code>: Device-side resource configuration.</p>
</td>
</tr>
<tr id="en-us_topic_0000001294591937_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001294591937_p17235973820"><a name="en-us_topic_0000001294591937_p17235973820"></a><a name="en-us_topic_0000001294591937_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001294591937_p973225082318"><a name="en-us_topic_0000001294591937_p973225082318"></a><a name="en-us_topic_0000001294591937_p973225082318"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294591937_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001294591937_p182459113812"><a name="en-us_topic_0000001294591937_p182459113812"></a><a name="en-us_topic_0000001294591937_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001294591937_p132314362521"><a name="en-us_topic_0000001294591937_p132314362521"></a><a name="en-us_topic_0000001294591937_p132314362521"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294591937_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001294591937_p423590386"><a name="en-us_topic_0000001294591937_p423590386"></a><a name="en-us_topic_0000001294591937_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001294591937_p1229447954"><a name="en-us_topic_0000001294591937_p1229447954"></a><a name="en-us_topic_0000001294591937_p1229447954"></a><code>dims</code> ∈ {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 4096, 3584}.</p>
</td>
</tr>
</tbody>
</table>

<a name="en-us_topic_0000001247793230_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001247793230_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001247793230_p12559123810"><a name="en-us_topic_0000001247793230_p12559123810"></a><a name="en-us_topic_0000001247793230_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001247793230_p7112274471"><a name="en-us_topic_0000001247793230_p7112274471"></a><a name="en-us_topic_0000001247793230_p7112274471"></a><code>AscendIndexFlatL2(const AscendIndexFlatL2&amp;) = delete;</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001247793230_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001247793230_p1212599383"><a name="en-us_topic_0000001247793230_p1212599383"></a><a name="en-us_topic_0000001247793230_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001247793230_p131714208358"><a name="en-us_topic_0000001247793230_p131714208358"></a><a name="en-us_topic_0000001247793230_p131714208358"></a>Declares the copy constructor as deleted. In other words, this is a non-copyable type.</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793230_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001247793230_p112195910383"><a name="en-us_topic_0000001247793230_p112195910383"></a><a name="en-us_topic_0000001247793230_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001247793230_p867213174418"><a name="en-us_topic_0000001247793230_p867213174418"></a><a name="en-us_topic_0000001247793230_p867213174418"></a><code>const AscendIndexFlatL2&amp;</code>: Constant <code>AscendIndexFlatL2</code>.</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793230_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001247793230_p17235973820"><a name="en-us_topic_0000001247793230_p17235973820"></a><a name="en-us_topic_0000001247793230_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001247793230_p973225082318"><a name="en-us_topic_0000001247793230_p973225082318"></a><a name="en-us_topic_0000001247793230_p973225082318"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793230_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001247793230_p182459113812"><a name="en-us_topic_0000001247793230_p182459113812"></a><a name="en-us_topic_0000001247793230_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001247793230_p132314362521"><a name="en-us_topic_0000001247793230_p132314362521"></a><a name="en-us_topic_0000001247793230_p132314362521"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793230_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001247793230_p423590386"><a name="en-us_topic_0000001247793230_p423590386"></a><a name="en-us_topic_0000001247793230_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001247793230_p182559163813"><a name="en-us_topic_0000001247793230_p182559163813"></a><a name="en-us_topic_0000001247793230_p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="en-us_topic_0000001294312453_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001294312453_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001294312453_p12559123810"><a name="en-us_topic_0000001294312453_p12559123810"></a><a name="en-us_topic_0000001294312453_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001294312453_p132681218211"><a name="en-us_topic_0000001294312453_p132681218211"></a><a name="en-us_topic_0000001294312453_p132681218211"></a><code>virtual ~AscendIndexFlatL2()</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001294312453_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001294312453_p1212599383"><a name="en-us_topic_0000001294312453_p1212599383"></a><a name="en-us_topic_0000001294312453_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001294312453_p131714208358"><a name="en-us_topic_0000001294312453_p131714208358"></a><a name="en-us_topic_0000001294312453_p131714208358"></a>The destructor of <code>AscendIndexFlatL2</code>. It destroys the <code>AscendIndexFlatL2</code> object and releases resources.</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312453_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001294312453_p112195910383"><a name="en-us_topic_0000001294312453_p112195910383"></a><a name="en-us_topic_0000001294312453_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001294312453_p8451184515218"><a name="en-us_topic_0000001294312453_p8451184515218"></a><a name="en-us_topic_0000001294312453_p8451184515218"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312453_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001294312453_p17235973820"><a name="en-us_topic_0000001294312453_p17235973820"></a><a name="en-us_topic_0000001294312453_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001294312453_p973225082318"><a name="en-us_topic_0000001294312453_p973225082318"></a><a name="en-us_topic_0000001294312453_p973225082318"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312453_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001294312453_p182459113812"><a name="en-us_topic_0000001294312453_p182459113812"></a><a name="en-us_topic_0000001294312453_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001294312453_p132314362521"><a name="en-us_topic_0000001294312453_p132314362521"></a><a name="en-us_topic_0000001294312453_p132314362521"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294312453_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001294312453_p423590386"><a name="en-us_topic_0000001294312453_p423590386"></a><a name="en-us_topic_0000001294312453_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001294312453_p182559163813"><a name="en-us_topic_0000001294312453_p182559163813"></a><a name="en-us_topic_0000001294312453_p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `copyFrom`<a name="en-us_TOPIC_0000001456375400"></a>

<a name="en-us_topic_0000001248112146_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001248112146_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001248112146_p12559123810"><a name="en-us_topic_0000001248112146_p12559123810"></a><a name="en-us_topic_0000001248112146_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001248112146_p1531315343445"><a name="en-us_topic_0000001248112146_p1531315343445"></a><a name="en-us_topic_0000001248112146_p1531315343445"></a><code>void copyFrom(faiss::IndexFlat *index);</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001248112146_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001248112146_p1212599383"><a name="en-us_topic_0000001248112146_p1212599383"></a><a name="en-us_topic_0000001248112146_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001248112146_p131714208358"><a name="en-us_topic_0000001248112146_p131714208358"></a><a name="en-us_topic_0000001248112146_p131714208358"></a>Copies an existing <code>index</code> to Ascend based on <code>AscendIndexFlat</code>, clears the current base library of <code>AscendIndexFlatL2</code>, and keeps the existing Device-side resource configuration of <code>AscendIndex</code>.</p>
</td>
</tr>
<tr id="en-us_topic_0000001248112146_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001248112146_p112195910383"><a name="en-us_topic_0000001248112146_p112195910383"></a><a name="en-us_topic_0000001248112146_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001248112146_p874812810555"><a name="en-us_topic_0000001248112146_p874812810555"></a><a name="en-us_topic_0000001248112146_p874812810555"></a><code>const faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="en-us_topic_0000001248112146_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001248112146_p17235973820"><a name="en-us_topic_0000001248112146_p17235973820"></a><a name="en-us_topic_0000001248112146_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001248112146_p973225082318"><a name="en-us_topic_0000001248112146_p973225082318"></a><a name="en-us_topic_0000001248112146_p973225082318"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001248112146_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001248112146_p182459113812"><a name="en-us_topic_0000001248112146_p182459113812"></a><a name="en-us_topic_0000001248112146_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001248112146_p132314362521"><a name="en-us_topic_0000001248112146_p132314362521"></a><a name="en-us_topic_0000001248112146_p132314362521"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001248112146_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001248112146_p423590386"><a name="en-us_topic_0000001248112146_p423590386"></a><a name="en-us_topic_0000001248112146_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001248112146_p182559163813"><a name="en-us_topic_0000001248112146_p182559163813"></a><a name="en-us_topic_0000001248112146_p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The value range of the <code>d</code> dimension parameter of the <code>Index</code> is {64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3584}. The value range of the total number of base library vectors is <code>0 <= n < 1e9</code>. The <code>metric_type</code> parameter must be <code>faiss::MetricType::METRIC_L2</code>.</p>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000001456535052"></a>

<a name="en-us_topic_0000001247793178_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001247793178_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001247793178_p12559123810"><a name="en-us_topic_0000001247793178_p12559123810"></a><a name="en-us_topic_0000001247793178_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001247793178_p10713954155218"><a name="en-us_topic_0000001247793178_p10713954155218"></a><a name="en-us_topic_0000001247793178_p10713954155218"></a><code>void copyTo(faiss::IndexFlat *index);</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001247793178_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001247793178_p1212599383"><a name="en-us_topic_0000001247793178_p1212599383"></a><a name="en-us_topic_0000001247793178_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001247793178_p131714208358"><a name="en-us_topic_0000001247793178_p131714208358"></a><a name="en-us_topic_0000001247793178_p131714208358"></a>Copies the retrieval resources of <code>AscendIndexFlatL2</code> to the CPU side.</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793178_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001247793178_p112195910383"><a name="en-us_topic_0000001247793178_p112195910383"></a><a name="en-us_topic_0000001247793178_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001247793178_p874812810555"><a name="en-us_topic_0000001247793178_p874812810555"></a><a name="en-us_topic_0000001247793178_p874812810555"></a><code>faiss::IndexFlat *index</code>: CPU-side <code>Index</code> resource.</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793178_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001247793178_p17235973820"><a name="en-us_topic_0000001247793178_p17235973820"></a><a name="en-us_topic_0000001247793178_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001247793178_p973225082318"><a name="en-us_topic_0000001247793178_p973225082318"></a><a name="en-us_topic_0000001247793178_p973225082318"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793178_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001247793178_p182459113812"><a name="en-us_topic_0000001247793178_p182459113812"></a><a name="en-us_topic_0000001247793178_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001247793178_p132314362521"><a name="en-us_topic_0000001247793178_p132314362521"></a><a name="en-us_topic_0000001247793178_p132314362521"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001247793178_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001247793178_p423590386"><a name="en-us_topic_0000001247793178_p423590386"></a><a name="en-us_topic_0000001247793178_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001247793178_p182559163813"><a name="en-us_topic_0000001247793178_p182559163813"></a><a name="en-us_topic_0000001247793178_p182559163813"></a><code>index</code> must be a valid CPU <code>Index</code> pointer. The user must free the memory occupied by the <code>Index</code>.</p>
</td>
</tr>
</tbody>
</table>

## `operator =`<a name="en-us_TOPIC_0000001456695116"></a>

<a name="en-us_topic_0000001294432513_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001294432513_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001294432513_p12559123810"><a name="en-us_topic_0000001294432513_p12559123810"></a><a name="en-us_topic_0000001294432513_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001294432513_p1213215268503"><a name="en-us_topic_0000001294432513_p1213215268503"></a><a name="en-us_topic_0000001294432513_p1213215268503"></a>AscendIndexFlatL2&amp; operator=(const AscendIndexFlatL2&amp;) = delete;</p>
</td>
</tr>
<tr id="en-us_topic_0000001294432513_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001294432513_p1212599383"><a name="en-us_topic_0000001294432513_p1212599383"></a><a name="en-us_topic_0000001294432513_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001294432513_p131714208358"><a name="en-us_topic_0000001294432513_p131714208358"></a><a name="en-us_topic_0000001294432513_p131714208358"></a>Declares the assignment operator as deleted. In other words, this is a non-copyable type.</p>
</td>
</tr>
<tr id="en-us_topic_0000001294432513_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001294432513_p112195910383"><a name="en-us_topic_0000001294432513_p112195910383"></a><a name="en-us_topic_0000001294432513_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001294432513_p867213174418"><a name="en-us_topic_0000001294432513_p867213174418"></a><a name="en-us_topic_0000001294432513_p867213174418"></a><code>const AscendIndexFlatL2&amp;</code>: Constant <code>AscendIndexFlatL2</code>.</p>
</td>
</tr>
<tr id="en-us_topic_0000001294432513_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001294432513_p17235973820"><a name="en-us_topic_0000001294432513_p17235973820"></a><a name="en-us_topic_0000001294432513_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001294432513_p973225082318"><a name="en-us_topic_0000001294432513_p973225082318"></a><a name="en-us_topic_0000001294432513_p973225082318"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294432513_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001294432513_p182459113812"><a name="en-us_topic_0000001294432513_p182459113812"></a><a name="en-us_topic_0000001294432513_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001294432513_p132314362521"><a name="en-us_topic_0000001294432513_p132314362521"></a><a name="en-us_topic_0000001294432513_p132314362521"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001294432513_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001294432513_p423590386"><a name="en-us_topic_0000001294432513_p423590386"></a><a name="en-us_topic_0000001294432513_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001294432513_p182559163813"><a name="en-us_topic_0000001294432513_p182559163813"></a><a name="en-us_topic_0000001294432513_p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>
