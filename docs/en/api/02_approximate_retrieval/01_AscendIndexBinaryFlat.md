# `AscendIndexBinaryFlat`<a name="en-us_TOPIC_0000001506334701"></a>

## Function Description<a name="en-us_TOPIC_0000001456694988"></a>

The `AscendIndexBinaryFlat` class inherits from Faiss `IndexBinary` and is used for binary feature retrieval.

It supports only <term>Atlas Inference Series products</term>.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `add`<a name="en-us_TOPIC_0000001456854896"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p4703164217446"><a name="p4703164217446"></a><a name="p4703164217446"></a><code>void add(idx_t n, const uint8_t *x) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Adds feature vectors to the base library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b812832571217"><a name="b812832571217"></a><a name="b812832571217"></a><code>idx_t n</code></strong>: Number of feature vectors to add to the base library.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b261412289122"><a name="b261412289122"></a><a name="b261412289122"></a><code>const uint8_t *x</code></strong>: Feature vectors to add to the base library.</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p13415621173615"><a name="p13415621173615"></a><a name="p13415621173615"></a>The length of pointer <code>x</code> must be <code>dims/8 * n</code>. Otherwise, out-of-bounds reads or writes may occur or the program may crash.</p>
<p id="p1727814248218"><a name="p1727814248218"></a><a name="p1727814248218"></a><code>n &gt; 0</code>. The <code>add</code> operation must ensure that the final base library size <code>ntotal</code> is the smaller of the <i><span class="varname" id="varname63161016193917"><a name="varname63161016193917"></a><a name="varname63161016193917"></a>actual chip memory capacity</span></i> and <code>1e9</code>.</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
> - The `add` API cannot be used together with the `add_with_ids` API.
> - After you use the `add` API, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` API.

## `add_with_ids`<a name="en-us_TOPIC_0000001506414809"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p15923147164612"><a name="p15923147164612"></a><a name="p15923147164612"></a><code>void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p359315567338"><a name="p359315567338"></a><a name="p359315567338"></a>Adds feature vectors to the base library and specifies the corresponding IDs.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b21773411615"><a name="b21773411615"></a><a name="b21773411615"></a><code>idx_t n</code></strong>: Number of feature vectors to add to the base library.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1733711363162"><a name="b1733711363162"></a><a name="b1733711363162"></a><code>const uint8_t *x</code></strong>: Feature vectors to add to the base library.</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b990063701613"><a name="b990063701613"></a><a name="b990063701613"></a><code>const idx_t *xids</code></strong>: IDs of the feature vectors to add to the base library.</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p019510419535"><a name="p019510419535"></a><a name="p019510419535"></a><code>0 &lt; n</code>. The <code>add</code> operation must ensure that the final base library size <code>n</code> is the smaller of the <i><span class="varname" id="varname63161016193917"><a name="varname63161016193917"></a><a name="varname63161016193917"></a>actual chip memory capacity</span></i> and <code>1e9</code>.</p>
<p id="p819694119533"><a name="p819694119533"></a><a name="p819694119533"></a>The length of pointer <code>x</code> must be <code>dims/8 * n</code>, and the length of pointer <code>xids</code> must be <code>n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. You need to ensure that <code>xids</code> is valid according to your service scenario. If duplicate IDs exist in the base library, the labels in the search results cannot be mapped to specific base-library vectors.</p>
</td>
</tr>
</tbody>
</table>

## `AscendIndexBinaryFlat`<a name="en-us_TOPIC_0000001456535056"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p33682318435"><a name="p33682318435"></a><a name="p33682318435"></a><code>AscendIndexBinaryFlat(int dims, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndexBinaryFlat</code>. It creates an <code>AscendIndexBinaryFlat</code> with dimension <code>dims</code> and sets Device-side resources based on the values configured in <code>config</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b48571551268"><a name="b48571551268"></a><a name="b48571551268"></a><code>int dims</code></strong>: Dimension of a set of feature vectors managed by <code>AscendIndexBinaryFlat</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b167641631570"><a name="b167641631570"></a><a name="b167641631570"></a><code>AscendIndexBinaryFlatConfig config</code></strong>: Device-side resource configuration.</p>
<p id="p1280191519597"><a name="p1280191519597"></a><a name="p1280191519597"></a><strong id="b769792117303"><a name="b769792117303"></a><a name="b769792117303"></a><code>bool usedFloat</code></strong>: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the <a href="#en-us_TOPIC_0000001456375288"><code>search</code></a> API. The default value is <code>false</code>. Set it to <code>true</code> to enable the performance improvement.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>dims</code> ∈ { 256, 512, 1024 }</p>
</td>
</tr>
</tbody>
</table>

<a name="table191641015539"></a>
<table><tbody><tr id="row8164101513314"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p14164141519314"><a name="p14164141519314"></a><a name="p14164141519314"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p1092313612565"><a name="p1092313612565"></a><a name="p1092313612565"></a><code>AscendIndexBinaryFlat(const faiss::IndexBinaryFlat *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></p>
</td>
</tr>
<tr id="row171644151312"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p101644157311"><a name="p101644157311"></a><a name="p101644157311"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p216571511319"><a name="p216571511319"></a><a name="p216571511319"></a>Constructor of <code>AscendIndexBinaryFlat</code>. It creates an Ascend retrieval index based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row816511155319"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p141652152034"><a name="p141652152034"></a><a name="p141652152034"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p1216513152314"><a name="p1216513152314"></a><a name="p1216513152314"></a><strong id="b19571543583"><a name="b19571543583"></a><a name="b19571543583"></a><code>const faiss::IndexBinaryFlat *index</code></strong>: CPU-side index resource.</p>
<p id="p141651115738"><a name="p141651115738"></a><a name="p141651115738"></a><strong id="b171651315732"><a name="b171651315732"></a><a name="b171651315732"></a><code>AscendIndexBinaryFlatConfig config</code></strong>: Device-side resource configuration.</p>
<p id="p1516531514317"><a name="p1516531514317"></a><a name="p1516531514317"></a><strong id="b156172041183020"><a name="b156172041183020"></a><a name="b156172041183020"></a><code>bool usedFloat</code></strong>: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the <a href="#en-us_TOPIC_0000001456375288"><code>search</code></a> API. The default value is <code>false</code>. Set it to <code>true</code> to enable the performance improvement.</p>
</td>
</tr>
<tr id="row5165515438"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p4165515138"><a name="p4165515138"></a><a name="p4165515138"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p5165161518319"><a name="p5165161518319"></a><a name="p5165161518319"></a>None</p>
</td>
</tr>
<tr id="row1165141515316"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p41650151032"><a name="p41650151032"></a><a name="p41650151032"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p12165315634"><a name="p12165315634"></a><a name="p12165315634"></a>None</p>
</td>
</tr>
<tr id="row2165101519312"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p7165201516316"><a name="p7165201516316"></a><a name="p7165201516316"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><p id="p616501515317"><a name="p616501515317"></a><a name="p616501515317"></a><code>index</code> must be a valid CPU index pointer. <code>index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;ntotal</code> is the smaller of the <i><span class="varname" id="varname1935761793413"><a name="varname1935761793413"></a><a name="varname1935761793413"></a>actual chip memory capacity</span></i> and <code>1e9</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table142022518319"></a>
<table><tbody><tr id="row720152517313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p17201257311"><a name="p17201257311"></a><a name="p17201257311"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p96251456568"><a name="p96251456568"></a><a name="p96251456568"></a><code>AscendIndexBinaryFlat(const faiss::IndexBinaryIDMap *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</code></p>
</td>
</tr>
<tr id="row42092517313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p220152511318"><a name="p220152511318"></a><a name="p220152511318"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p22002516311"><a name="p22002516311"></a><a name="p22002516311"></a>Constructor of <code>AscendIndexBinaryFlat</code>. It creates an Ascend retrieval index based on an existing <code>index</code>.</p>
</td>
</tr>
<tr id="row1520625935"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p920725739"><a name="p920725739"></a><a name="p920725739"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p620132517318"><a name="p620132517318"></a><a name="p620132517318"></a><strong id="b12011251937"><a name="b12011251937"></a><a name="b12011251937"></a><code>const faiss::IndexBinaryIDMap *index</code></strong>: CPU-side index resource.</p>
<p id="p152011259319"><a name="p152011259319"></a><a name="p152011259319"></a><strong id="b12013257310"><a name="b12013257310"></a><a name="b12013257310"></a><code>AscendIndexBinaryFlatConfig config</code></strong>: Device-side resource configuration.</p>
<p id="p5201425933"><a name="p5201425933"></a><a name="p5201425933"></a><strong id="b2019215542303"><a name="b2019215542303"></a><a name="b2019215542303"></a><code>bool usedFloat</code></strong>: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the <a href="#en-us_TOPIC_0000001456375288"><code>search</code></a> API. The default value is <code>false</code>. Set it to <code>true</code> to enable the performance improvement.</p>
</td>
</tr>
<tr id="row1120122517310"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p02017253319"><a name="p02017253319"></a><a name="p02017253319"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p52142514310"><a name="p52142514310"></a><a name="p52142514310"></a>None</p>
</td>
</tr>
<tr id="row8211825339"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3217255318"><a name="p3217255318"></a><a name="p3217255318"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p02116255316"><a name="p02116255316"></a><a name="p02116255316"></a>None</p>
</td>
</tr>
<tr id="row6216254312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p22182519314"><a name="p22182519314"></a><a name="p22182519314"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p152291327101513"><a name="p152291327101513"></a><a name="p152291327101513"></a><code>index</code> must be a valid <code>faiss::IndexBinaryIDMap</code> pointer. <code>index-&gt;index</code> must be a valid <code>IndexBinaryFlat</code> pointer. <code>index-&gt;index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;index-&gt;ntotal</code> is the smaller of the <i><span class="varname" id="varname9229152721512"><a name="varname9229152721512"></a><a name="varname9229152721512"></a>actual chip memory capacity</span></i> and <code>1e9</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table145324411437"></a>
<table><tbody><tr id="row75329411438"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p145326412034"><a name="p145326412034"></a><a name="p145326412034"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6828175217434"><a name="p6828175217434"></a><a name="p6828175217434"></a><code>AscendIndexBinaryFlat(const AscendIndexBinaryFlat &amp;) = delete;</code></p>
</td>
</tr>
<tr id="row0532841735"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1532154111318"><a name="p1532154111318"></a><a name="p1532154111318"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p165321741735"><a name="p165321741735"></a><a name="p165321741735"></a>Declares the copy constructor of <code>AscendIndexBinaryFlat</code> as deleted. Therefore, <code>AscendIndexBinaryFlat</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row55324411131"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17532641633"><a name="p17532641633"></a><a name="p17532641633"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b162111517184312"><a name="b162111517184312"></a><a name="b162111517184312"></a><code>const AscendIndexBinaryFlat &amp;</code></strong>: Constant <code>AscendIndexBinaryFlat</code>.</p>
</td>
</tr>
<tr id="row19532144117319"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p353214114316"><a name="p353214114316"></a><a name="p353214114316"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p953217411937"><a name="p953217411937"></a><a name="p953217411937"></a>None</p>
</td>
</tr>
<tr id="row2532164118313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p653210411131"><a name="p653210411131"></a><a name="p653210411131"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p105324416311"><a name="p105324416311"></a><a name="p105324416311"></a>None</p>
</td>
</tr>
<tr id="row16532041331"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1653294114316"><a name="p1653294114316"></a><a name="p1653294114316"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1253284115310"><a name="p1253284115310"></a><a name="p1253284115310"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~AscendIndexBinaryFlat`<a name="en-us_TOPIC_0000001506495917"></a>

<a name="table13115573310"></a>
<table><tbody><tr id="row133117571634"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p93116571312"><a name="p93116571312"></a><a name="p93116571312"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1223817411653"><a name="p1223817411653"></a><a name="p1223817411653"></a><code>virtual ~AscendIndexBinaryFlat() = default;</code></p>
</td>
</tr>
<tr id="row131111571314"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p19311857938"><a name="p19311857938"></a><a name="p19311857938"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p20311175712315"><a name="p20311175712315"></a><a name="p20311175712315"></a>Destructor of <code>AscendIndexBinaryFlat</code>. It destroys the <code>AscendIndexBinaryFlat</code> object and releases resources.</p>
</td>
</tr>
<tr id="row1631185720315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p331111571035"><a name="p331111571035"></a><a name="p331111571035"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p823816417516"><a name="p823816417516"></a><a name="p823816417516"></a>None</p>
</td>
</tr>
<tr id="row131110576311"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p123112571132"><a name="p123112571132"></a><a name="p123112571132"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p113111757932"><a name="p113111757932"></a><a name="p113111757932"></a>None</p>
</td>
</tr>
<tr id="row2031118575316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2311757432"><a name="p2311757432"></a><a name="p2311757432"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p133117571934"><a name="p133117571934"></a><a name="p133117571934"></a>None</p>
</td>
</tr>
<tr id="row0311205718311"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p03118571131"><a name="p03118571131"></a><a name="p03118571131"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1231195718316"><a name="p1231195718316"></a><a name="p1231195718316"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `copyFrom`<a name="en-us_TOPIC_0000001506414941"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyFrom(const faiss::IndexBinaryFlat *index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies data from an existing <code>Index</code> to <code>AscendIndexBinaryFlat</code>, clears the current base library of <code>AscendIndexBinaryFlat</code>, and retains the original Device-side resource configuration.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b166777494441"><a name="b166777494441"></a><a name="b166777494441"></a><code>const faiss::IndexBinaryFlat *index</code></strong>: <code>faiss::IndexBinaryFlat</code> pointer.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid <code>IndexBinaryFlat</code> pointer. <code>index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;ntotal</code> is the smaller of the <i><span class="varname" id="varname1935761793413"><a name="varname1935761793413"></a><a name="varname1935761793413"></a>actual chip memory capacity</span></i> and <code>1e9</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table1570816514419"></a>
<table><tbody><tr id="row87089510415"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2070816511342"><a name="p2070816511342"></a><a name="p2070816511342"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p370818511140"><a name="p370818511140"></a><a name="p370818511140"></a><code>void copyFrom(const faiss::IndexBinaryIDMap *index);</code></p>
</td>
</tr>
<tr id="row1970816519416"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p157086511048"><a name="p157086511048"></a><a name="p157086511048"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p17086511346"><a name="p17086511346"></a><a name="p17086511346"></a>Copies data from an existing <code>index</code> to <code>AscendIndexBinaryFlat</code>, clears the current base library of <code>AscendIndexBinaryFlat</code>, and retains the original Device-side resource configuration.</p>
</td>
</tr>
<tr id="row67081551148"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p37082511045"><a name="p37082511045"></a><a name="p37082511045"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1670865117413"><a name="p1670865117413"></a><a name="p1670865117413"></a><strong id="b95691227174516"><a name="b95691227174516"></a><a name="b95691227174516"></a><code>const faiss::IndexBinaryIDMap *index</code></strong>: <code>faiss::IndexBinaryIDMap</code> pointer.</p>
</td>
</tr>
<tr id="row117082511940"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p197081551340"><a name="p197081551340"></a><a name="p197081551340"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1070811513418"><a name="p1070811513418"></a><a name="p1070811513418"></a>None</p>
</td>
</tr>
<tr id="row1170805111412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p370805118413"><a name="p370805118413"></a><a name="p370805118413"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p117085512418"><a name="p117085512418"></a><a name="p117085512418"></a>None</p>
</td>
</tr>
<tr id="row1370895113414"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p16708951942"><a name="p16708951942"></a><a name="p16708951942"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p77081951843"><a name="p77081951843"></a><a name="p77081951843"></a><code>index</code> must be a valid <code>faiss::IndexBinaryIDMap</code> pointer. <code>index-&gt;index</code> must be a valid <code>IndexBinaryFlat</code> pointer. <code>index-&gt;index-&gt;d</code> ∈ {256, 512, 1024}. <code>index-&gt;index-&gt;ntotal</code> is the smaller of the <i><span class="varname" id="varname670818511944"><a name="varname670818511944"></a><a name="varname670818511944"></a>actual chip memory capacity</span></i> and <code>1e9</code>.</p>
</td>
</tr>
</tbody>
</table>

## `copyTo`<a name="en-us_TOPIC_0000001456855048"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p425342585420"><a name="p425342585420"></a><a name="p425342585420"></a><code>void copyTo(faiss::IndexBinaryFlat *index) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies data from an existing <code>AscendIndexBinaryFlat</code> to <code>faiss::IndexBinaryFlat index</code>, and clears the original resources of <code>index</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b194718510463"><a name="b194718510463"></a><a name="b194718510463"></a><code>faiss::IndexBinaryFlat *index</code></strong>: <code>faiss::IndexBinaryFlat</code> pointer.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>index</code> must be a valid <code>IndexBinaryFlat</code> pointer. The user must release the resources of the copied <code>index</code>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table19831553111512"></a>
<table><tbody><tr id="row1183118539158"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p10831135301517"><a name="p10831135301517"></a><a name="p10831135301517"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void copyTo(faiss::IndexBinaryIDMap *index) const;</code></p>
</td>
</tr>
<tr id="row1831153151517"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p158311553101511"><a name="p158311553101511"></a><a name="p158311553101511"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p68311653201512"><a name="p68311653201512"></a><a name="p68311653201512"></a>Copies data from an existing <code>AscendIndexBinaryFlat</code> to <code>faiss::IndexBinaryIDMap index</code>, and clears the original resources of <code>index</code>.</p>
</td>
</tr>
<tr id="row8831125312154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p10831145314156"><a name="p10831145314156"></a><a name="p10831145314156"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p108311153141519"><a name="p108311153141519"></a><a name="p108311153141519"></a><strong id="b172171419114612"><a name="b172171419114612"></a><a name="b172171419114612"></a><code>faiss::IndexBinaryIDMap *index</code></strong>: <code>faiss::IndexBinaryIDMap</code> pointer.</p>
</td>
</tr>
<tr id="row11831195315154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1683110535159"><a name="p1683110535159"></a><a name="p1683110535159"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p58323537152"><a name="p58323537152"></a><a name="p58323537152"></a>None</p>
</td>
</tr>
<tr id="row1083225391518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1683225314159"><a name="p1683225314159"></a><a name="p1683225314159"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1683235310157"><a name="p1683235310157"></a><a name="p1683235310157"></a>None</p>
</td>
</tr>
<tr id="row1983215318157"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11832175317159"><a name="p11832175317159"></a><a name="p11832175317159"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4832115316152"><a name="p4832115316152"></a><a name="p4832115316152"></a><code>index</code> must be a valid <code>IndexBinaryIDMap</code> pointer. The user must release the copied <code>Index</code> resources.</p>
</td>
</tr>
</tbody>
</table>

## `operator=`<a name="en-us_TOPIC_0000001456535072"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16196201484419"><a name="p16196201484419"></a><a name="p16196201484419"></a><code>AscendIndexBinaryFlat &amp;operator = (const AscendIndexBinaryFlat &amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the assignment operator of <code>AscendIndexBinaryFlat</code> as deleted. Therefore, <code>AscendIndexBinaryFlat</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b105275248101"><a name="b105275248101"></a><a name="b105275248101"></a><code>const AscendIndexBinaryFlat &amp;</code></strong>: Constant <code>AscendIndexBinaryFlat</code>.</p>
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

## `remove_ids`<a name="en-us_TOPIC_0000001506495769"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p723211434816"><a name="p723211434816"></a><a name="p723211434816"></a><code>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Deletes the specified feature vectors from the base library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b94414427185"><a name="b94414427185"></a><a name="b94414427185"></a><code>const faiss::IDSelector &amp;sel</code></strong>: Feature vectors to delete. For details about usage and definition, see the relevant Faiss documentation.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Number of feature vectors deleted successfully, with invalid IDs ignored.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `reset`<a name="en-us_TOPIC_0000001456855028"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>void reset() override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Clears the base-library vectors of this <code>AscendIndexBinaryFlat</code>.</p>
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

## `search`<a id="en-us_TOPIC_0000001456375288"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p18443123012413"><a name="p18443123012413"></a><a name="p18443123012413"></a><code>void search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels, const SearchParameters *params) const override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Feature vector query API. It returns the IDs and corresponding distances of the <code>k</code> most similar features based on the input feature vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><strong id="b6402181191915"><a name="b6402181191915"></a><a name="b6402181191915"></a><code>idx_t n</code></strong>: Number of query vectors.</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><strong id="b8615513161912"><a name="b8615513161912"></a><a name="b8615513161912"></a><code>const uint8_t *x</code></strong>: Query vectors.</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><strong id="b82719159198"><a name="b82719159198"></a><a name="b82719159198"></a><code>idx_t k</code></strong>: Number of most similar results to return.</p>
<p id="p191711443182415"><a name="p191711443182415"></a><a name="p191711443182415"></a><strong id="b10561743122410"><a name="b10561743122410"></a><a name="b10561743122410"></a><code>const SearchParameters *params</code></strong>: Optional Faiss parameters. The default value is <code>nullptr</code>, and this parameter is not supported for now.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b74651934121914"><a name="b74651934121914"></a><a name="b74651934121914"></a><code>int32_t *distances</code></strong>: Distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b11761113620191"><a name="b11761113620191"></a><a name="b11761113620191"></a><code>idx_t *labels</code></strong>: IDs of the <code>k</code> nearest vectors.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul38481656216"></a><a name="ul38481656216"></a><ul id="ul38481656216"><li>The length of feature vector data <code>x</code> must be <code>dims/8 * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>0 &lt; n ≤ 1e9</code>, <code>0 &lt; k ≤ 1e5</code> (the <code>n ≤ 1e9</code> limit is far beyond the actual available resources, so you are advised to choose an appropriate number of query vectors according to your service scenario).</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1659211341612"></a>
<table><tbody><tr id="row1259231351612"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.1.1"><p id="p11592161301616"><a name="p11592161301616"></a><a name="p11592161301616"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.1.1 "><p id="p17215919132819"><a name="p17215919132819"></a><a name="p17215919132819"></a><code>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;</code></p>
</td>
</tr>
<tr id="row8592513191612"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.2.1"><p id="p859216134169"><a name="p859216134169"></a><a name="p859216134169"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.2.1 "><p id="p1617317122911"><a name="p1617317122911"></a><a name="p1617317122911"></a>Feature vector query API. It returns the IDs and corresponding distances of the <code>k</code> most similar features based on the input feature vectors. This API is used for the retrieval mode in which binary features are stored in the base library and float features are used for retrieval.</p>
</td>
</tr>
<tr id="row8592121311162"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.3.1"><p id="p5592151320162"><a name="p5592151320162"></a><a name="p5592151320162"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.3.1 "><p id="p142581642102914"><a name="p142581642102914"></a><a name="p142581642102914"></a><strong id="b36001143196"><a name="b36001143196"></a><a name="b36001143196"></a><code>idx_t n</code></strong>: Number of query vectors.</p>
<p id="p1025813425296"><a name="p1025813425296"></a><a name="p1025813425296"></a><strong id="b525218718197"><a name="b525218718197"></a><a name="b525218718197"></a><code>const float *x</code></strong>: Query vectors.</p>
<p id="p1825814422294"><a name="p1825814422294"></a><a name="p1825814422294"></a><strong id="b12147951917"><a name="b12147951917"></a><a name="b12147951917"></a><code>idx_t k</code></strong>: Number of most similar results to return.</p>
</td>
</tr>
<tr id="row19592181320167"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.4.1"><p id="p9592111318169"><a name="p9592111318169"></a><a name="p9592111318169"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.4.1 "><p id="p66920493291"><a name="p66920493291"></a><a name="p66920493291"></a><strong id="b1785151311912"><a name="b1785151311912"></a><a name="b1785151311912"></a><code>float *distances</code></strong>: Distance values between the query vectors and the top <code>k</code> nearest vectors.</p>
<p id="p14692144913291"><a name="p14692144913291"></a><a name="p14692144913291"></a><strong id="b133591761916"><a name="b133591761916"></a><a name="b133591761916"></a><code>idx_t *labels</code></strong>: IDs of the <code>k</code> nearest vectors.</p>
</td>
</tr>
<tr id="row6592171319163"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.5.1"><p id="p45921313161611"><a name="p45921313161611"></a><a name="p45921313161611"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.5.1 "><p id="p220841917285"><a name="p220841917285"></a><a name="p220841917285"></a>None</p>
</td>
</tr>
<tr id="row19593513111612"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.6.1"><p id="p459391311616"><a name="p459391311616"></a><a name="p459391311616"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.6.1 "><a name="ul127218309196"></a><a name="ul127218309196"></a><ul id="ul127218309196"><li>The length of feature vector data <code>x</code> must be <code>dims * n</code>, and the lengths of <code>distances</code> and <code>labels</code> must be <code>k * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>0 &lt; n ≤ 1e9</code>, <code>0 &lt; k ≤ 1e5</code> (the <code>n ≤ 1e9</code> limit is far beyond the actual available resources, so you are advised to choose an appropriate number of query vectors according to your service scenario).</li></ul>
</td>
</tr>
</tbody>
</table>

## `setRemoveFast`<a name="en-us_TOPIC_0000002024780673"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a><code>static void setRemoveFast(bool removeFast);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p347711919317"><a name="p347711919317"></a><a name="p347711919317"></a>Sets whether to quickly delete vectors from the base library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b943925813513"><a name="b943925813513"></a><a name="b943925813513"></a><code>bool removeFast</code></strong>: Set it to <code>true</code> to use fast deletion, or <code>false</code> not to use it.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>Fast deletion improves the performance of deleting the base library, but it slightly reduces the performance of adding data to the base library. If you do not call this API, fast deletion is disabled by default. This API can be called only once, and you must call it before you construct the index object.</p>
</td>
</tr>
</tbody>
</table>
