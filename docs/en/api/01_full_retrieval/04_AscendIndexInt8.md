# `AscendIndexInt8`<a id="en-us_TOPIC_0000001506495841"></a>

## Function Description<a id="en-us_TOPIC_0000001506495913"></a>

<code>AscendIndexInt8</code> is the base class of the indexes that use INT8 feature vectors in the feature retrieval component. It defines interfaces for other INT8 indexes in feature retrieval.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must lock before use, or the retrieval interface may raise exceptions. It also does not support sharing one device across different threads. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `add`<a name="en-us_TOPIC_0000001506334825"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>void add(idx_t n, const int8_t *x);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Adds new feature vectors to the <code>AscendIndexInt8</code> base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>const int8_t *x</code>: Feature vectors to add to the base library.</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul9411102014349"></a><a name="ul9411102014349"></a><ul id="ul9411102014349"><li>The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table6211414109"></a>
<table><tbody><tr id="row19219141603"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p132101414015"><a name="p132101414015"></a><a name="p132101414015"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1821101419018"><a name="p1821101419018"></a><a name="p1821101419018"></a><code>void add(idx_t n, const char *x);</code></p>
</td>
</tr>
<tr id="row02111141013"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p152212147010"><a name="p152212147010"></a><a name="p152212147010"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p222191411013"><a name="p222191411013"></a><a name="p222191411013"></a>Adds new feature vectors to the <code>AscendIndexInt8</code> base library. When you add features with <code>add</code>, the default IDs of the corresponding features are [0, <code>ntotal</code>).</p>
</td>
</tr>
<tr id="row11224141604"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p7221714203"><a name="p7221714203"></a><a name="p7221714203"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p11221148019"><a name="p11221148019"></a><a name="p11221148019"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p022151410016"><a name="p022151410016"></a><a name="p022151410016"></a><code>const char *x</code>: Feature vectors to add to the base library.</p>
</td>
</tr>
<tr id="row122251416018"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p162281415018"><a name="p162281415018"></a><a name="p162281415018"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p1922171411011"><a name="p1922171411011"></a><a name="p1922171411011"></a>None</p>
</td>
</tr>
<tr id="row3225141020"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p52214145020"><a name="p52214145020"></a><a name="p52214145020"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1222614206"><a name="p1222614206"></a><a name="p1222614206"></a>None</p>
</td>
</tr>
<tr id="row1922131418013"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p16221814502"><a name="p16221814502"></a><a name="p16221814502"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul13290113043617"></a><a name="ul13290113043617"></a><ul id="ul13290113043617"><li>The length of pointer <code>x</code> must be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
> - The <code>add</code> interface cannot be used together with the <code>add_with_ids</code> interface.
> - After you use the <code>add</code> interface, the <code>labels</code> in the search results may repeat. If your service has requirements for labels, you are advised to use the <code>add_with_ids</code> interface.

## `add_with_ids`<a name="en-us_TOPIC_0000001506614905"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p95747912314"><a name="p95747912314"></a><a name="p95747912314"></a><code>void add_with_ids(idx_t n, const int8_t *x, const idx_t *ids);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Adds new feature vectors to the <code>AscendIndexInt8</code> base library and specifies the feature IDs.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>const int8_t *x</code>: Feature vectors to add to the base library.</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><code>const idx_t *ids</code>: IDs of the feature vectors to add to the base library. The IDs must be unique within the <code>Index</code> instance.</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul6110252163513"></a><a name="ul6110252163513"></a><ul id="ul6110252163513"><li>The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table38814511704"></a>
<table><tbody><tr id="row138812511016"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p178817511000"><a name="p178817511000"></a><a name="p178817511000"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p19881351509"><a name="p19881351509"></a><a name="p19881351509"></a><code>void add_with_ids(idx_t n, const char *x, const idx_t *ids);</code></p>
</td>
</tr>
<tr id="row88855119016"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p10885519011"><a name="p10885519011"></a><a name="p10885519011"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p72832460516"><a name="p72832460516"></a><a name="p72832460516"></a>Adds new feature vectors to the <code>AscendIndexInt8</code> base library and specifies the feature IDs.</p>
</td>
</tr>
<tr id="row88885115010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p17881515011"><a name="p17881515011"></a><a name="p17881515011"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p6882511707"><a name="p6882511707"></a><a name="p6882511707"></a><code>idx_t n</code>: Number of feature vectors to add to the base library.</p>
<p id="p168916512008"><a name="p168916512008"></a><a name="p168916512008"></a><code>const char *x</code>: Feature vectors to add to the base library.</p>
<p id="p16897517012"><a name="p16897517012"></a><a name="p16897517012"></a><code>const idx_t *ids</code>: IDs corresponding to the feature vectors to add to the base library. The IDs must be unique within the <code>Index</code> instance.</p>
</td>
</tr>
<tr id="row6895513016"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p689195119019"><a name="p689195119019"></a><a name="p689195119019"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p118915120011"><a name="p118915120011"></a><a name="p118915120011"></a>None</p>
</td>
</tr>
<tr id="row1689551609"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p7893511403"><a name="p7893511403"></a><a name="p7893511403"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p3896511015"><a name="p3896511015"></a><a name="p3896511015"></a>None</p>
</td>
</tr>
<tr id="row18915120017"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p7898511307"><a name="p7898511307"></a><a name="p7898511307"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul61401059163611"></a><a name="ul61401059163611"></a><ul id="ul61401059163611"><li>The length of pointer <code>x</code> must be <code>dims * n</code>, and the length of pointer <code>ids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>The valid range of the total number of base-library vectors is <code>0 &lt; n &lt; 1e9</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `assign`<a name="en-us_TOPIC_0000001506495721"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p63341832173418"><a name="p63341832173418"></a><a name="p63341832173418"></a><code>void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Feature-vector retrieval interface of <code>AscendIndexInt8</code>. It returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><code>idx_t n</code>: Number of query feature vectors.</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><code>const int8_t *x</code>: Feature-vector data.</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><code>idx_t k</code>: Number of most similar results to return.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul3740916183818"></a><a name="ul3740916183818"></a><ul id="ul3740916183818"><li>The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the length of <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>n</code> must be greater than <code>0</code> and less than <code>1e9</code>.</li><li><code>k</code> must be greater than <code>0</code> and less than or equal to <code>4096</code>.</li><li><code>n * k</code> must be less than <code>1e10</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndexInt8`<a name="en-us_TOPIC_0000001506614993"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p156319619286"><a name="p156319619286"></a><a name="p156319619286"></a><code>AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndexInt8</code>. It creates an <code>AscendIndexInt8</code> with dimension <code>dims</code>. The dimension of the vector set managed by a single <code>Index</code> is unique. Device-side resources are set according to the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><code>int dims</code>: Dimension of a set of feature vectors managed by <code>AscendIndexInt8</code>.</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><code>faiss::MetricType metric</code>: Distance metric used by <code>AscendIndexInt8</code> when performing feature-vector similarity retrieval. Currently supported values are <code>faiss::MetricType::METRIC_L2</code> and <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>AscendIndexInt8Config config</code>: Device-side resource configuration.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>dims</code> must be an integer that is not smaller than <code>64</code> and not larger than <code>1024</code>, and it must be divisible by <code>64</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table103312407520"></a>
<table><tbody><tr id="row9331540657"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p933940155"><a name="p933940155"></a><a name="p933940155"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a><code>AscendIndexInt8(const AscendIndexInt8&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row163311401851"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p13311408510"><a name="p13311408510"></a><a name="p13311408510"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1233154015517"><a name="p1233154015517"></a><a name="p1233154015517"></a>Declares this <code>Index</code> copy constructor as deleted. Therefore, the type is non-copyable.</p>
</td>
</tr>
<tr id="row203364017512"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p53304012513"><a name="p53304012513"></a><a name="p53304012513"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndexInt8&amp;</code>: <code>AscendIndexInt8</code> object.</p>
</td>
</tr>
<tr id="row33318406512"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1833840458"><a name="p1833840458"></a><a name="p1833840458"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p9331140556"><a name="p9331140556"></a><a name="p9331140556"></a>None</p>
</td>
</tr>
<tr id="row1533740858"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p733184020518"><a name="p733184020518"></a><a name="p733184020518"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p93315408518"><a name="p93315408518"></a><a name="p93315408518"></a>None</p>
</td>
</tr>
<tr id="row7339405511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p143319401358"><a name="p143319401358"></a><a name="p143319401358"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p163315401851"><a name="p163315401851"></a><a name="p163315401851"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table1882220715614"></a>
<table><tbody><tr id="row282214719612"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p10822167367"><a name="p10822167367"></a><a name="p10822167367"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndexInt8();</code></p>
</td>
</tr>
<tr id="row128221171266"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p98221971361"><a name="p98221971361"></a><a name="p98221971361"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p19822271613"><a name="p19822271613"></a><a name="p19822271613"></a>Destructor of <code>AscendIndexInt8</code>. It destroys the <code>AscendIndexInt8</code> object and releases resources.</p>
</td>
</tr>
<tr id="row2082217362"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p138221771067"><a name="p138221771067"></a><a name="p138221771067"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
</td>
</tr>
<tr id="row15822977619"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1682212712617"><a name="p1682212712617"></a><a name="p1682212712617"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18221278617"><a name="p18221278617"></a><a name="p18221278617"></a>None</p>
</td>
</tr>
<tr id="row382247166"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p88221378615"><a name="p88221378615"></a><a name="p88221378615"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p682214717616"><a name="p682214717616"></a><a name="p682214717616"></a>None</p>
</td>
</tr>
<tr id="row198221076614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p38221974612"><a name="p38221974612"></a><a name="p38221974612"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p148221171060"><a name="p148221171060"></a><a name="p148221171060"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getDeviceList`<a name="en-us_TOPIC_0000001672982421"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a><code>std::vector&lt;int&gt; getDeviceList() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>Returns the device-side Ascend AI Processor settings managed by <code>Index</code>. Subclasses inherit from it and implement it. This base class does not provide a corresponding implementation and returns only an empty <code>vector&lt;int&gt;</code>.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Device-side Ascend AI Processor settings managed by <code>Index</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getDim`<a name="en-us_TOPIC_0000001690599922"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p192220207614"><a name="p192220207614"></a><a name="p192220207614"></a><code>int getDim() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3221620764"><a name="p3221620764"></a><a name="p3221620764"></a>Gets the dimension of the feature vector set managed by <code>AscendIndexInt8</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8220620861"><a name="p8220620861"></a><a name="p8220620861"></a>Dimension of the feature vector set managed by <code>AscendIndexInt8</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p172160208615"><a name="p172160208615"></a><a name="p172160208615"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getNTotal`<a name="en-us_TOPIC_0000001738718517"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p192220207614"><a name="p192220207614"></a><a name="p192220207614"></a><code>faiss::idx_t getNTotal() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3221620764"><a name="p3221620764"></a><a name="p3221620764"></a>Gets the number of feature vectors that <code>AscendIndexInt8</code> has added to the base vector set.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p45014471818"><a name="p45014471818"></a><a name="p45014471818"></a>Number of feature vectors that <code>AscendIndexInt8</code> has added to the base vector set.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p172160208615"><a name="p172160208615"></a><a name="p172160208615"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getMetricType`<a name="en-us_TOPIC_0000001738678653"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58591491393"><a name="p58591491393"></a><a name="p58591491393"></a><code>faiss::MetricType getMetricType() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18599491290"><a name="p18599491290"></a><a name="p18599491290"></a>Gets the distance metric type used by <code>AscendIndexInt8</code> when performing feature vector similarity retrieval.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1685710491096"><a name="p1685710491096"></a><a name="p1685710491096"></a>Distance metric type used by <code>AscendIndexInt8</code> when performing feature vector similarity retrieval.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p785724919915"><a name="p785724919915"></a><a name="p785724919915"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `isTrained`<a name="en-us_TOPIC_0000001690759666"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p79107141490"><a name="p79107141490"></a><a name="p79107141490"></a><code>bool isTrained() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5909201417911"><a name="p5909201417911"></a><a name="p5909201417911"></a>Determines whether <code>AscendIndexInt8</code> is trained.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1790810141395"><a name="p1790810141395"></a><a name="p1790810141395"></a>Trained state of <code>AscendIndexInt8</code>. <code>true</code> means trained, and <code>false</code> means not trained.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p129074145914"><a name="p129074145914"></a><a name="p129074145914"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `operator =`<a name="en-us_TOPIC_0000001506414841"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8&amp; operator=(const AscendIndexInt8&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares this <code>Index</code> assignment operator as deleted. Therefore, the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendIndexInt8&amp;</code>: Constant <code>AscendIndexInt8</code>.</p>
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

## `reclaimMemory`<a name="en-us_TOPIC_0000001506615133"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a><code>virtual size_t reclaimMemory();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>A virtual function defined in the base class. See the subclass for details.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16621234163"><a name="p16621234163"></a><a name="p16621234163"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `remove_ids`<a name="en-us_TOPIC_0000001456695088"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>size_t remove_ids(const faiss::IDSelector &amp;sel);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Implements the interface for deleting the specified feature vectors from the base vector set in <code>AscendIndexInt8</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.3.1 "><p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><code>const faiss::IDSelector &amp;sel</code>: Feature vectors to delete. For details on usage and definition, see the corresponding Faiss documentation.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Number of deleted feature vectors.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `reserveMemory`<a name="en-us_TOPIC_0000001506615065"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a><code>virtual void reserveMemory(size_t numVecs);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>A virtual function defined in the base class. See the subclass for details.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a><code>size_t numVecs</code>: Number of base vectors for which to reserve memory.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6587140131617"><a name="p6587140131617"></a><a name="p6587140131617"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `search`<a name="en-us_TOPIC_0000001506414889"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p17821720121118"><a name="p17821720121118"></a><a name="p17821720121118"></a><code>void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Feature-vector search interface of <code>AscendIndexInt8</code>. It returns the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><code>idx_t n</code>: Number of query feature vectors.</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><code>const int8_t *x</code>: Feature-vector data.</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><code>idx_t k</code>: Number of most similar results to return.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid distances with <code>65504</code> or <code>-65504</code> depending on the metric.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query. When fewer than <code>k</code> valid retrieval results are available, fill the remaining invalid labels with <code>-1</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul4165643183912"></a><a name="ul4165643183912"></a><ul id="ul4165643183912"><li>The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>n</code> must be greater than <code>0</code> and less than <code>1e9</code>.</li><li><code>k</code> must be greater than <code>0</code> and less than or equal to <code>4096</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table88671631181418"></a>
<table><tbody><tr id="row6867133191414"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p14867153118141"><a name="p14867153118141"></a><a name="p14867153118141"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p28671831101414"><a name="p28671831101414"></a><a name="p28671831101414"></a><code>void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;</code></p>
</td>
</tr>
<tr id="row8867631151417"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1086733161419"><a name="p1086733161419"></a><a name="p1086733161419"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p10867231161419"><a name="p10867231161419"></a><a name="p10867231161419"></a>Feature-vector search interface of <code>AscendIndexInt8</code>. It returns the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
</td>
</tr>
<tr id="row1686713131418"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p286783116148"><a name="p286783116148"></a><a name="p286783116148"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p13867203110149"><a name="p13867203110149"></a><a name="p13867203110149"></a><code>idx_t n</code>: Number of query feature vectors.</p>
<p id="p11867173116148"><a name="p11867173116148"></a><a name="p11867173116148"></a><code>const char *x</code>: Feature-vector data.</p>
<p id="p20867031131410"><a name="p20867031131410"></a><a name="p20867031131410"></a><code>idx_t k</code>: Number of most similar results to return.</p>
</td>
</tr>
<tr id="row188673319140"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p14867193115144"><a name="p14867193115144"></a><a name="p14867193115144"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p2086715317143"><a name="p2086715317143"></a><a name="p2086715317143"></a><code>float *distances</code>: Distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p88672310144"><a name="p88672310144"></a><a name="p88672310144"></a><code>idx_t *labels</code>: IDs of the top <code>k</code> nearest vectors to the query.</p>
</td>
</tr>
<tr id="row1786719315149"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p08674311147"><a name="p08674311147"></a><a name="p08674311147"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p886710319142"><a name="p886710319142"></a><a name="p886710319142"></a>None.</p>
</td>
</tr>
<tr id="row11867231121415"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p16867931161410"><a name="p16867931161410"></a><a name="p16867931161410"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul176971234124016"></a><a name="ul176971234124016"></a><ul id="ul176971234124016"><li>The length of query feature-vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>n</code> must be greater than <code>0</code> and less than <code>1e9</code>.</li><li><code>k</code> must be greater than <code>0</code> and less than or equal to <code>4096</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `train`<a name="en-us_TOPIC_0000001456534956"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a><code>virtual void train(idx_t n, const int8_t *x);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>A virtual function defined in the base class. See the subclass for details.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><code>idx_t n</code>: Number of feature vectors in the training set.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><code>const int8_t *x</code>: Feature-vector data.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p255165217139"><a name="p255165217139"></a><a name="p255165217139"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `updateCentroids`<a name="en-us_TOPIC_0000001506414833"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p107821866408"><a name="p107821866408"></a><a name="p107821866408"></a><code>virtual void updateCentroids(idx_t n, const int8_t *x);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p283153493915"><a name="p283153493915"></a><a name="p283153493915"></a>A virtual function defined in the base class. See the subclass for details.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><code>idx_t n</code>: Number of feature vectors in the training set.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><code>const int8_t *x</code>: Feature-vector data.</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p211213924718"><a name="p211213924718"></a><a name="p211213924718"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table2023134918146"></a>
<table><tbody><tr id="row5231649201420"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p4233499147"><a name="p4233499147"></a><a name="p4233499147"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p18234493149"><a name="p18234493149"></a><a name="p18234493149"></a><code>virtual void updateCentroids(idx_t n, const char *x);</code></p>
</td>
</tr>
<tr id="row7232497144"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p8238490147"><a name="p8238490147"></a><a name="p8238490147"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p105115579251"><a name="p105115579251"></a><a name="p105115579251"></a>A virtual function defined in the base class. See the subclass for details.</p>
</td>
</tr>
<tr id="row1023164911414"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p1723349161412"><a name="p1723349161412"></a><a name="p1723349161412"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p323149141417"><a name="p323149141417"></a><a name="p323149141417"></a><code>idx_t n</code>: Number of feature vectors in the training set.</p>
<p id="p16231949131414"><a name="p16231949131414"></a><a name="p16231949131414"></a><code>const char *x</code>: Feature-vector data.</p>
</td>
</tr>
<tr id="row1231749111414"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p1723114911419"><a name="p1723114911419"></a><a name="p1723114911419"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p323449121419"><a name="p323449121419"></a><a name="p323449121419"></a>None</p>
</td>
</tr>
<tr id="row72374910141"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p32311490142"><a name="p32311490142"></a><a name="p32311490142"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p82354971417"><a name="p82354971417"></a><a name="p82354971417"></a>None</p>
</td>
</tr>
<tr id="row11230494140"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p1923124951413"><a name="p1923124951413"></a><a name="p1923124951413"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p157561925193210"><a name="p157561925193210"></a><a name="p157561925193210"></a>None</p>
</td>
</tr>
</tbody>
</table>
