# `AscendIndexILFlat`<a name="en-us_TOPIC_0000002514896041"></a>

## Function Description<a name="en-us_TOPIC_0000002482656058"></a>

`AscendIndexILFlat` is the standard-mode scenario of `ILFlat`. You need to use `Init` to initialize the corresponding resources. After initialization, it allocates a contiguous block of memory to store the base library. After use, call the `Finalize` interface to release the resources.

`AscendIndexILFlat` supports only <term>Atlas inference products</term> and only the inner product distance type in the standard deployment mode. `AscendIndexILFlat` depends on the Flat and AICPU operators. For details, see <a href="../../05_user_guide.md#flat">Flat</a> and <a href="../../05_user_guide.md#aicpu">AICPU</a>.

Multithreaded concurrent calls are supported. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

## `AddFeatures`<a name="en-us_TOPIC_0000002514776041"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR AddFeatures(int n, const float *features);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Adds <code>n</code> feature vectors to the feature library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b645815911297"><a name="b645815911297"></a><a name="b645815911297"></a><code>int n</code></strong>: Number of feature vectors to insert.</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b1053019911245"><a name="b1053019911245"></a><a name="b1053019911245"></a><code>const float *features</code></strong>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1835741212302"><a name="b1835741212302"></a><a name="b1835741212302"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul10674191294110"></a><a name="ul10674191294110"></a><ul id="ul10674191294110"><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><code>features</code> must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table392463914228"></a>
<table><tbody><tr id="row17924183911228"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p159241739182219"><a name="p159241739182219"></a><a name="p159241739182219"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19924039112212"><a name="p19924039112212"></a><a name="p19924039112212"></a><code>APP_ERROR AddFeatures(int n, const float16_t *features);</code></p>
</td>
</tr>
<tr id="row13924439172216"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p792417393229"><a name="p792417393229"></a><a name="p792417393229"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1792413962211"><a name="p1792413962211"></a><a name="p1792413962211"></a>Adds <code>n</code> feature vectors to the feature library.</p>
</td>
</tr>
<tr id="row792418398229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1692410397224"><a name="p1692410397224"></a><a name="p1692410397224"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1192416399222"><a name="p1192416399222"></a><a name="p1192416399222"></a><strong id="b1992493913229"><a name="b1992493913229"></a><a name="b1992493913229"></a><code>int n</code></strong>: Number of feature vectors to insert.</p>
<p id="p292416399222"><a name="p292416399222"></a><a name="p292416399222"></a><strong id="b1453142822318"><a name="b1453142822318"></a><a name="b1453142822318"></a><code>const float16_t *features</code></strong>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
</td>
</tr>
<tr id="row5924163962213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p0924439162217"><a name="p0924439162217"></a><a name="p0924439162217"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p19924103932211"><a name="p19924103932211"></a><a name="p19924103932211"></a>None</p>
</td>
</tr>
<tr id="row14924163932212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p18924143910228"><a name="p18924143910228"></a><a name="p18924143910228"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p179241397228"><a name="p179241397228"></a><a name="p179241397228"></a><strong id="b892483952211"><a name="b892483952211"></a><a name="b892483952211"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row159242391222"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2924153992215"><a name="p2924153992215"></a><a name="p2924153992215"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul16924339172210"></a><a name="ul16924339172210"></a><ul id="ul16924339172210"><li><strong id="b1592433919225"><a name="b1592433919225"></a><a name="b1592433919225"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><code>features</code> must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndexILFlat`<a name="en-us_TOPIC_0000002516511133"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>AscendIndexILFlat();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndexILFlat</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a>None</p>
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

<a name="table161511529133912"></a>
<table><tbody><tr id="row1615110293394"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2151429113910"><a name="p2151429113910"></a><a name="p2151429113910"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16794134614447"><a name="p16794134614447"></a><a name="p16794134614447"></a><code>AscendIndexILFlat(const AscendIndexILFlat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row51517295398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p21514294391"><a name="p21514294391"></a><a name="p21514294391"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015122918399"><a name="p2015122918399"></a><a name="p2015122918399"></a>Declares the copy constructor of <code>AscendIndexILFlat</code> as deleted. Therefore, <code>AscendIndexILFlat</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row815120292398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7151122933917"><a name="p7151122933917"></a><a name="p7151122933917"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b2450181274519"><a name="b2450181274519"></a><a name="b2450181274519"></a><code>const AscendIndexILFlat&amp;</code></strong>: <code>AscendIndexILFlat</code> object.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a><code>virtual ~AscendIndexILFlat();</code></p>
</td>
</tr>
<tr id="row1926221314401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1926218134408"><a name="p1926218134408"></a><a name="p1926218134408"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82621213184020"><a name="p82621213184020"></a><a name="p82621213184020"></a>Destructor of <code>AscendIndexILFlat</code>.</p>
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

## `ComputeDistance`<a name="en-us_TOPIC_0000002482736032"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19771939144811"><a name="p19771939144811"></a><a name="p19771939144811"></a><code>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Returns the distances between <code>n</code> feature vectors and all feature vectors in the base library. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b75131329183314"><a name="b75131329183314"></a><a name="b75131329183314"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b6862131183312"><a name="b6862131183312"></a><a name="b6862131183312"></a><code>const float16_t *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b1041123433313"><a name="b1041123433313"></a><a name="b1041123433313"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b376183643316"><a name="b376183643316"></a><a name="b376183643316"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1750033215518"><a name="p1750033215518"></a><a name="p1750033215518"></a><strong id="b668194911337"><a name="b668194911337"></a><a name="b668194911337"></a><code>float *distances</code></strong>: External memory. It stores the distances between query vectors and base library vectors. The total length should be <code>n * nTotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, <code>ntotal</code> rounded up to a multiple of 16).</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul167968714447"></a><a name="ul167968714447"></a><ul id="ul167968714447"><li><strong id="b1141410121444"><a name="b1141410121444"></a><a name="b1141410121444"></a><code>n</code></strong>: The recommended value should be in <code>(0, capacity]</code>.</li><li><strong id="b429253634917"><a name="b429253634917"></a><a name="b429253634917"></a><code>distances</code></strong>: The required buffer length is <code>n * ntotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, the result of rounding <code>ntotal</code> up to a multiple of 16. The valid comparison distances for each query are stored in the first <code>ntotal</code> positions, and the padded data has no practical meaning).</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li><li><code>queries</code> and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table17574555124816"></a>
<table><tbody><tr id="row757435594819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p155742557480"><a name="p155742557480"></a><a name="p155742557480"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p0574255104813"><a name="p0574255104813"></a><a name="p0574255104813"></a><code>APP_ERROR ComputeDistance(int n, const float *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row14574135514811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p205741955124812"><a name="p205741955124812"></a><a name="p205741955124812"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p45741955174814"><a name="p45741955174814"></a><a name="p45741955174814"></a>Returns the distances between <code>n</code> feature vectors and all feature vectors in the base library. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</p>
</td>
</tr>
<tr id="row85751555194813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p35754551488"><a name="p35754551488"></a><a name="p35754551488"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1757535519484"><a name="p1757535519484"></a><a name="p1757535519484"></a><strong id="b257555544811"><a name="b257555544811"></a><a name="b257555544811"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p1575115520489"><a name="p1575115520489"></a><a name="p1575115520489"></a><strong id="b158293141686"><a name="b158293141686"></a><a name="b158293141686"></a><code>const float *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p1957512553485"><a name="p1957512553485"></a><a name="p1957512553485"></a><strong id="b7575155164810"><a name="b7575155164810"></a><a name="b7575155164810"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p20575185520489"><a name="p20575185520489"></a><a name="p20575185520489"></a><strong id="b1657585510484"><a name="b1657585510484"></a><a name="b1657585510484"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row1557595510487"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1257545518481"><a name="p1257545518481"></a><a name="p1257545518481"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p115751555184810"><a name="p115751555184810"></a><a name="p115751555184810"></a><strong id="b13575195584812"><a name="b13575195584812"></a><a name="b13575195584812"></a><code>float *distances</code></strong>: External memory. It stores the distances between query vectors and base library vectors. The total length should be <code>n * nTotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, <code>ntotal</code> rounded up to a multiple of 16).</p>
</td>
</tr>
<tr id="row7575175554817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1757575514814"><a name="p1757575514814"></a><a name="p1757575514814"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p125758557485"><a name="p125758557485"></a><a name="p125758557485"></a><strong id="b19575165519489"><a name="b19575165519489"></a><a name="b19575165519489"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row9575755204810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12575185574818"><a name="p12575185574818"></a><a name="p12575185574818"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7575125514817"></a><a name="ul7575125514817"></a><ul id="ul7575125514817"><li><strong id="b18575155510484"><a name="b18575155510484"></a><a name="b18575155510484"></a><code>n</code></strong>: The recommended value should be in <code>(0, capacity]</code>.</li><li><strong id="b75751155114811"><a name="b75751155114811"></a><a name="b75751155114811"></a><code>distances</code></strong>: The required buffer length is <code>n * ntotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, the result of rounding <code>ntotal</code> up to a multiple of 16. The valid comparison distances for each query are stored in the first <code>ntotal</code> positions, and the padded data has no practical meaning).</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li><li><code>queries</code> and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `ComputeDistanceByIdx`<a name="en-us_TOPIC_0000002514896043"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1196718166412"><a name="p1196718166412"></a><a name="p1196718166412"></a><code>APP_ERROR ComputeDistanceByIdx(int n, const float *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a><code>ComputeDistance</code> calculates the distances between query vectors and all base library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distances between query vectors and the base library vectors at the specified indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1178514265435"><a name="b1178514265435"></a><a name="b1178514265435"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b811819197314"><a name="b811819197314"></a><a name="b811819197314"></a><code>const float *queries</code></strong>: Feature vectors to query. The valid length is <code>n * dim</code>, and <code>dim</code> must match the dimension specified during initialization.</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a><code>const int *num</code></strong>: Number of base library feature vectors to compare for each query. The length is <code>n</code>.</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b79815523517"><a name="b79815523517"></a><a name="b79815523517"></a><code>const idx_t *indices</code></strong>: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum <code>num</code> value. The length of <code>indices</code> is <code>n * max(num)</code>. If the input is on the host, <code>indices</code> is a host pointer. If the input is on the device, <code>indices</code> is a device pointer.</p>
<p id="p13553919567"><a name="p13553919567"></a><a name="p13553919567"></a><strong id="b184841525865"><a name="b184841525865"></a><a name="b184841525865"></a><code>MEMORY_TYPE memoryType</code></strong>: Policy for where the input and output are stored. The default is <code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>. The available policies are as follows:</p>
<a name="ul125365550127"></a><a name="ul125365550127"></a><ul id="ul125365550127"><li><code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>: Input on the host, output on the host.</li><li><code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE</code>: Input on the device, output on the device.</li><li><code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST</code>: Input on the device, output on the host.</li><li><code>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE</code>: Input on the host, output on the device.</li></ul>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b156251035184313"><a name="b156251035184313"></a><a name="b156251035184313"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b25863377438"><a name="b25863377438"></a><a name="b25863377438"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p057182814222"><a name="p057182814222"></a><a name="p057182814222"></a><strong id="b0194557446"><a name="b0194557446"></a><a name="b0194557446"></a><code>float *distances</code></strong>: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum <code>num</code> value. The total length is <code>n * max(num)</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b140412864414"><a name="b140412864414"></a><a name="b140412864414"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a><code>num</code></strong>: User-specified, with length <code>n</code>, and each query's <code>num</code> value must be in <code>[0, ntotal]</code>.</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>.</li><li>API parameter configuration example: <code>n = 3</code>, <code>num[3] = {1, 3, 5}</code> indicates that the three queries compare with <code>1</code>, <code>3</code>, and <code>5</code> base library vectors respectively. Since <code>max(num) = 5</code>, the storage space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example, <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When selecting a <code>memoryType</code> storage policy, <code>queries</code> and <code>distances</code> must be pointers to the corresponding location, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table93703718308"></a>
<table><tbody><tr id="row20370173302"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p143716720307"><a name="p143716720307"></a><a name="p143716720307"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1037115763020"><a name="p1037115763020"></a><a name="p1037115763020"></a><code>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row103719723013"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p0371207153017"><a name="p0371207153017"></a><a name="p0371207153017"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p53715711306"><a name="p53715711306"></a><a name="p53715711306"></a><code>ComputeDistance</code> calculates the distances between query vectors and all base library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distances between query vectors and the base library vectors at the specified indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</p>
</td>
</tr>
<tr id="row123716710302"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p16371147193013"><a name="p16371147193013"></a><a name="p16371147193013"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p23711377308"><a name="p23711377308"></a><a name="p23711377308"></a><strong id="b83711372301"><a name="b83711372301"></a><a name="b83711372301"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p173714714309"><a name="p173714714309"></a><a name="p173714714309"></a><strong id="b1926533153016"><a name="b1926533153016"></a><a name="b1926533153016"></a><code>const float16_t *queries</code></strong>: Feature vectors to query. The valid length is <code>n * dim</code>, and <code>dim</code> must match the dimension specified during initialization.</p>
<p id="p1237110743020"><a name="p1237110743020"></a><a name="p1237110743020"></a><strong id="b133711723012"><a name="b133711723012"></a><a name="b133711723012"></a><code>const int *num</code></strong>: Number of base library feature vectors to compare for each query. The length is <code>n</code>.</p>
<p id="p037167153019"><a name="p037167153019"></a><a name="p037167153019"></a><strong id="b12371167113014"><a name="b12371167113014"></a><a name="b12371167113014"></a><code>const idx_t *indices</code></strong>: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum <code>num</code> value. The length of <code>indices</code> is <code>n * max(num)</code>. If the input is on the host, <code>indices</code> is a host pointer. If the input is on the device, <code>indices</code> is a device pointer.</p>
<p id="p123717711303"><a name="p123717711303"></a><a name="p123717711303"></a><strong id="b1137116717301"><a name="b1137116717301"></a><a name="b1137116717301"></a><code>MEMORY_TYPE memoryType</code></strong>: Policy for where the input and output are stored. The default is <code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>. The available policies are as follows:</p>
<a name="ul183711373302"></a><a name="ul183711373302"></a><ul id="ul183711373302"><li><code>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST</code>: Input on the host, output on the host.</li><li><code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE</code>: Input on the device, output on the device.</li><li><code>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST</code>: Input on the device, output on the host.</li><li><code>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE</code>: Input on the host, output on the device.</li></ul>
<p id="p173715717301"><a name="p173715717301"></a><a name="p173715717301"></a><strong id="b18371874303"><a name="b18371874303"></a><a name="b18371874303"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p1637117793019"><a name="p1637117793019"></a><a name="p1637117793019"></a><strong id="b153711719302"><a name="b153711719302"></a><a name="b153711719302"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row837117183012"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p1737119715302"><a name="p1737119715302"></a><a name="p1737119715302"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p4371378300"><a name="p4371378300"></a><a name="p4371378300"></a><strong id="b137117716306"><a name="b137117716306"></a><a name="b137117716306"></a><code>float *distances</code></strong>: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum <code>num</code> value. The total length is <code>n * max(num)</code>.</p>
</td>
</tr>
<tr id="row037177153010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p183711274309"><a name="p183711274309"></a><a name="p183711274309"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p63711776304"><a name="p63711776304"></a><a name="p63711776304"></a><strong id="b1337118763012"><a name="b1337118763012"></a><a name="b1337118763012"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row193711676307"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p137114753010"><a name="p137114753010"></a><a name="p137114753010"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul13371476304"></a><a name="ul13371476304"></a><ul id="ul13371476304"><li><strong id="b737127163015"><a name="b737127163015"></a><a name="b737127163015"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><strong id="b103719712303"><a name="b103719712303"></a><a name="b103719712303"></a><code>num</code></strong>: User-specified, with length <code>n</code>, and each query's <code>num</code> value must be in <code>[0, ntotal]</code>.</li><li><strong id="b33718723018"><a name="b33718723018"></a><a name="b33718723018"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>.</li><li>API parameter configuration example: <code>n = 3</code>, <code>num[3] = {1, 3, 5}</code> indicates that the three queries compare with <code>1</code>, <code>3</code>, and <code>5</code> base library vectors respectively. Since <code>max(num) = 5</code>, the storage space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example, <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul10372137123013"></a><a name="ul10372137123013"></a><ul id="ul10372137123013"><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `Finalize`<a name="en-us_TOPIC_0000002482656060"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>void Finalize();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Releases feature library management resources.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1570793612442"><a name="b1570793612442"></a><a name="b1570793612442"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `GetFeatures`<a name="en-us_TOPIC_0000002484074790"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR GetFeatures(int n, float *features, const idx_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the host.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a><code>int n</code></strong>: Number of base library vectors to get.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a><code>const idx_t *indices</code></strong>: Indices corresponding to the feature vectors, with length <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b38573713437"><a name="b38573713437"></a><a name="b38573713437"></a><code>float *features</code></strong>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>, and you can get <code>ntotal</code> by calling <code>GetNTotal</code>.</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table018415716495"></a>
<table><tbody><tr id="row51841657124915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3184757144914"><a name="p3184757144914"></a><a name="p3184757144914"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16184957174915"><a name="p16184957174915"></a><a name="p16184957174915"></a><code>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices);</code></p>
</td>
</tr>
<tr id="row121844578499"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1184205717496"><a name="p1184205717496"></a><a name="p1184205717496"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6184457154911"><a name="p6184457154911"></a><a name="p6184457154911"></a>Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the host.</p>
</td>
</tr>
<tr id="row20184257154913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18184657114919"><a name="p18184657114919"></a><a name="p18184657114919"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1418413578493"><a name="p1418413578493"></a><a name="p1418413578493"></a><strong id="b16184115711494"><a name="b16184115711494"></a><a name="b16184115711494"></a><code>int n</code></strong>: Number of base library vectors to get.</p>
<p id="p11184757204918"><a name="p11184757204918"></a><a name="p11184757204918"></a><strong id="b718445714916"><a name="b718445714916"></a><a name="b718445714916"></a><code>const idx_t *indices</code></strong>: Indices corresponding to the feature vectors, with length <code>n</code>.</p>
</td>
</tr>
<tr id="row19184195714498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p3184175711496"><a name="p3184175711496"></a><a name="p3184175711496"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p12184657154920"><a name="p12184657154920"></a><a name="p12184657154920"></a><strong id="b12804115914544"><a name="b12804115914544"></a><a name="b12804115914544"></a><code>float16_t *features</code></strong>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
</td>
</tr>
<tr id="row1918411573494"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p19184357194919"><a name="p19184357194919"></a><a name="p19184357194919"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p14184757184914"><a name="p14184757184914"></a><a name="p14184757184914"></a><strong id="b191841957184915"><a name="b191841957184915"></a><a name="b191841957184915"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row16184195774920"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p181841657104916"><a name="p181841657104916"></a><a name="p181841657104916"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul181842571494"></a><a name="ul181842571494"></a><ul id="ul181842571494"><li><strong id="b6184105764918"><a name="b6184105764918"></a><a name="b6184105764918"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>, and you can get <code>ntotal</code> by calling <code>GetNTotal</code>.</li><li><strong id="b111847574493"><a name="b111847574493"></a><a name="b111847574493"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetFeaturesOnDevice`<a name="en-us_TOPIC_0000002516516843"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR GetFeaturesOnDevice (int n, float16_t *features, const idx_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the Device.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a><code>int n</code></strong>: Number of base library vectors to get.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a><code>const idx_t *indices</code></strong>: Indices corresponding to the feature vectors, with length <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b885718291130"><a name="b885718291130"></a><a name="b885718291130"></a><code>float16_t *features</code></strong>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension. Device-side pointer.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>, and you can get <code>ntotal</code> by calling <code>GetNTotal</code>.</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table15312115612410"></a>
<table><tbody><tr id="row1831211561843"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2031217561042"><a name="p2031217561042"></a><a name="p2031217561042"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10312185617418"><a name="p10312185617418"></a><a name="p10312185617418"></a><code>APP_ERROR GetFeaturesOnDevice (int n, float *features, const idx_t *indices);</code></p>
</td>
</tr>
<tr id="row123121356046"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p531245612416"><a name="p531245612416"></a><a name="p531245612416"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p431217567418"><a name="p431217567418"></a><a name="p431217567418"></a>Queries the feature vectors with the specified indices for <code>n</code> entries. Output is on the Device.</p>
</td>
</tr>
<tr id="row531213561245"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p431225615416"><a name="p431225615416"></a><a name="p431225615416"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p63123561342"><a name="p63123561342"></a><a name="p63123561342"></a><strong id="b13312356848"><a name="b13312356848"></a><a name="b13312356848"></a><code>int n</code></strong>: Number of base library vectors to get.</p>
<p id="p1731215620414"><a name="p1731215620414"></a><a name="p1731215620414"></a><strong id="b143129561748"><a name="b143129561748"></a><a name="b143129561748"></a><code>const idx_t *indices</code></strong>: Indices corresponding to the feature vectors, with length <code>n</code>.</p>
</td>
</tr>
<tr id="row53126562043"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1431212564417"><a name="p1431212564417"></a><a name="p1431212564417"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p193121656244"><a name="p193121656244"></a><a name="p193121656244"></a><strong id="b1679419391057"><a name="b1679419391057"></a><a name="b1679419391057"></a><code>float *features</code></strong>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension. Device-side pointer.</p>
</td>
</tr>
<tr id="row10312056346"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p73124561444"><a name="p73124561444"></a><a name="p73124561444"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p931217561412"><a name="p931217561412"></a><a name="p931217561412"></a><strong id="b1931275616410"><a name="b1931275616410"></a><a name="b1931275616410"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row123127565418"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1931214561141"><a name="p1931214561141"></a><a name="p1931214561141"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4312115612410"></a><a name="ul4312115612410"></a><ul id="ul4312115612410"><li><strong id="b331255610416"><a name="b331255610416"></a><a name="b331255610416"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>, and you can get <code>ntotal</code> by calling <code>GetNTotal</code>.</li><li><strong id="b2312185619410"><a name="b2312185619410"></a><a name="b2312185619410"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetNTotal`<a name="en-us_TOPIC_0000002514776043"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p0336129171210"><a name="p0336129171210"></a><a name="p0336129171210"></a><code>int GetNTotal() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Queries the theoretical maximum number of feature vectors in the current feature library. If the feature vector indices are inserted consecutively, <code>ntotal</code> is equal to the number of feature vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a><strong id="b445021732816"><a name="b445021732816"></a><a name="b445021732816"></a><code>int ntotal</code></strong>: The theoretical maximum number of feature vectors, that is, the maximum base library index plus <code>1</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a><code>int</code></strong>: The theoretical maximum number of feature vectors, that is, the maximum base library index plus <code>1</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Init`<a name="en-us_TOPIC_0000002482736034"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>Initialization function of <code>AscendIndexILFlat</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1517311219268"><a name="b1517311219268"></a><a name="b1517311219268"></a><code>int dim</code></strong>: Dimension of the feature vectors managed by <code>AscendIndexILFlat</code>.</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8628752620"><a name="b8628752620"></a><a name="b8628752620"></a><code>int capacity</code></strong>: Maximum base library capacity. The API allocates <code>capacity * dim * sizeof(fp16)</code> bytes of memory based on the <code>capacity</code> value.</p>
<p id="p1450765311416"><a name="p1450765311416"></a><a name="p1450765311416"></a><strong id="b5404134231712"><a name="b5404134231712"></a><a name="b5404134231712"></a><code>faiss::MetricType metricType</code></strong>: Feature distance type, including inner product, Euclidean distance, and cosine similarity.</p>
<p id="p1291682015184"><a name="p1291682015184"></a><a name="p1291682015184"></a><strong id="b6916192019187"><a name="b6916192019187"></a><a name="b6916192019187"></a><code>const std::vector&lt;int&gt; &amp;deviceList</code></strong>: Device-side resource configuration.</p>
<p id="p1411722401512"><a name="p1411722401512"></a><a name="p1411722401512"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a><code>int64_t resourceSize</code></strong>: Device-side preset memory pool size, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>-1</code>, which means <code>128 MB</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b13793144414268"><a name="b13793144414268"></a><a name="b13793144414268"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1768605017262"></a><a name="ul1768605017262"></a><ul id="ul1768605017262"><li><code>dim</code> ∈ {32, 64, 128, 256, 384, 512}.</li><li><code>metricType</code>: <code>AscendIndexILFlat</code> currently implements only the inner product distance, so it supports only <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>.</li><li><code>capacity</code>: The maximum memory that the API can allocate for the base library is <code>12,288,000,000</code> bytes, and the allowed range of <code>capacity</code> is <code>[0, 12000000]</code>.<a name="ul138816512117"></a><a name="ul138816512117"></a><ul id="ul138816512117"><li>For example, for a base library vector set with 512 dimensions and the FP16 type, the maximum supported <code>capacity</code> is 12 million (<code>12288000000 / (512 * sizeof(fp16))</code>).</li><li>For a base library vector set with 256 dimensions and the FP16 type, <code>capacity</code> can still be set to at most 12 million, even though the memory limit supports a larger value.</li></ul>
</li><li>Only single-card configuration is supported. Multi-card configuration is not supported yet, and <strong id="b16270210371"><a name="b16270210371"></a><a name="b16270210371"></a><code>deviceList.size() == 1</code></strong> must hold.</li><li><code>resourceSize</code> can be set to <code>-1</code> or any value in <code>[134217728, 4294967296]</code>, which is equivalent to <code>[128 MB, 4096 MB]</code>. This parameter is determined jointly by the base library size and the search batch size. When the base library contains at least 10 million vectors and the batch size is at least 16, you are advised to set it to <code>1024 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `operator =`<a name="en-us_TOPIC_0000002482794858"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4759192852812"><a name="p4759192852812"></a><a name="p4759192852812"></a><code>AscendIndexILFlat&amp; operator=(const AscendIndexILFlat &amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5787143522812"><a name="p5787143522812"></a><a name="p5787143522812"></a>Declares this <code>Index</code> assignment operator as deleted. Therefore, the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p175601140172811"><a name="p175601140172811"></a><a name="p175601140172811"></a><strong id="b1023216252912"><a name="b1023216252912"></a><a name="b1023216252912"></a><code>const AscendIndexILFlat &amp;</code></strong>: <code>AscendIndexILFlat</code> object.</p>
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

## `RemoveFeatures`<a name="en-us_TOPIC_0000002482917750"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p411713313214"><a name="p411713313214"></a><a name="p411713313214"></a><code>APP_ERROR RemoveFeatures(int n, const idx_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Deletes <code>n</code> feature vectors with the specified indices from the vector library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b10676733203011"><a name="b10676733203011"></a><a name="b10676733203011"></a><code>int n</code></strong>: Number of feature vectors to delete.</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b1248654013016"><a name="b1248654013016"></a><a name="b1248654013016"></a><code>const idx_t *indices</code></strong>: Indices of the feature vectors. The length is <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b727535618302"><a name="b727535618302"></a><a name="b727535618302"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>, and you can get <code>ntotal</code> by calling <code>GetNTotal</code>.</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><code>indices</code> must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `Search`<a name="en-us_TOPIC_0000002514896045"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p95721657104813"><a name="p95721657104813"></a><a name="p95721657104813"></a><code>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Returns the indices and corresponding distances of the <code>topk</code> base library vectors that are closest to the query vectors. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b2797917153611"><a name="b2797917153611"></a><a name="b2797917153611"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1445642010368"><a name="b1445642010368"></a><a name="b1445642010368"></a><code>const float16_t *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1177612213614"><a name="b1177612213614"></a><a name="b1177612213614"></a><code>int topk</code></strong>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b18837162415364"><a name="b18837162415364"></a><a name="b18837162415364"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b1174653015367"><a name="b1174653015367"></a><a name="b1174653015367"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b187841356153711"><a name="b187841356153711"></a><a name="b187841356153711"></a><code>float *distances</code></strong>: External memory. It stores the cosine distances corresponding to the <code>topk * n</code> base library feature vectors that are most similar to the query. The length is <code>n * topk</code>.</p>
<p id="p1154614405264"><a name="p1154614405264"></a><a name="p1154614405264"></a><strong id="b12933115811373"><a name="b12933115811373"></a><a name="b12933115811373"></a><code>idx_t *indices</code></strong>: External memory. It returns the indices corresponding to the <code>topk</code> base library vectors that are most similar to the query. The length is <code>n * topk</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b111527319385"><a name="b111527319385"></a><a name="b111527319385"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1346615102548"></a><a name="ul1346615102548"></a><ul id="ul1346615102548"><li><strong id="b5538111715545"><a name="b5538111715545"></a><a name="b5538111715545"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><strong id="b18681518105411"><a name="b18681518105411"></a><a name="b18681518105411"></a><code>topk</code></strong>: The value must be in <code>(0, 1024]</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><code>indices</code>, <code>queries</code>, and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table838713119461"></a>
<table><tbody><tr id="row33871117462"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p83873114461"><a name="p83873114461"></a><a name="p83873114461"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p33878119466"><a name="p33878119466"></a><a name="p33878119466"></a><code>APP_ERROR Search(int n, const float *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row1438891104611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p143889110469"><a name="p143889110469"></a><a name="p143889110469"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1388121164616"><a name="p1388121164616"></a><a name="p1388121164616"></a>Returns the indices and corresponding distances of the <code>topk</code> base library vectors that are closest to the query vectors. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped distances are returned.</p>
</td>
</tr>
<tr id="row1038821104617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p9388121124613"><a name="p9388121124613"></a><a name="p9388121124613"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p113881518469"><a name="p113881518469"></a><a name="p113881518469"></a><strong id="b123881015468"><a name="b123881015468"></a><a name="b123881015468"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p6388111204615"><a name="p6388111204615"></a><a name="p6388111204615"></a><strong id="b4285489460"><a name="b4285489460"></a><a name="b4285489460"></a><code>const float *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p1738816112464"><a name="p1738816112464"></a><a name="p1738816112464"></a><strong id="b838811114614"><a name="b838811114614"></a><a name="b838811114614"></a><code>int topk</code></strong>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.</p>
<p id="p1338831154612"><a name="p1338831154612"></a><a name="p1338831154612"></a><strong id="b1438814115461"><a name="b1438814115461"></a><a name="b1438814115461"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p183888134612"><a name="p183888134612"></a><a name="p183888134612"></a><strong id="b438816119468"><a name="b438816119468"></a><a name="b438816119468"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row1938815124610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p13388151134618"><a name="p13388151134618"></a><a name="p13388151134618"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p538810111463"><a name="p538810111463"></a><a name="p538810111463"></a><strong id="b1338861114611"><a name="b1338861114611"></a><a name="b1338861114611"></a><code>float *distances</code></strong>: External memory. It stores the cosine distances corresponding to the <code>topk * n</code> base library feature vectors that are most similar to the query. The length is <code>n * topk</code>.</p>
<p id="p1638841124616"><a name="p1638841124616"></a><a name="p1638841124616"></a><strong id="b9388191154616"><a name="b9388191154616"></a><a name="b9388191154616"></a><code>idx_t *indices</code></strong>: External memory. It returns the indices corresponding to the <code>topk</code> base library vectors that are most similar to the query. The length is <code>n * topk</code>.</p>
</td>
</tr>
<tr id="row1938811154610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1838821174615"><a name="p1838821174615"></a><a name="p1838821174615"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p338814194614"><a name="p338814194614"></a><a name="p338814194614"></a><strong id="b1138813119469"><a name="b1138813119469"></a><a name="b1138813119469"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row8388618462"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1338815119460"><a name="p1338815119460"></a><a name="p1338815119460"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1138817154617"></a><a name="ul1138817154617"></a><ul id="ul1138817154617"><li><strong id="b3388181144610"><a name="b3388181144610"></a><a name="b3388181144610"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><strong id="b13881714467"><a name="b13881714467"></a><a name="b13881714467"></a><code>topk</code></strong>: The value must be in <code>(0, 1024]</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul193898112469"></a><a name="ul193898112469"></a><ul id="ul193898112469"><li><code>indices</code>, <code>queries</code>, and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchByThreshold`<a name="en-us_TOPIC_0000002482656062"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p111481632134920"><a name="p111481632134920"></a><a name="p111481632134920"></a><code>APP_ERROR SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1893114655310"><a name="p1893114655310"></a><a name="p1893114655310"></a>Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1378124514396"><a name="b1378124514396"></a><a name="b1378124514396"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1840232010507"><a name="b1840232010507"></a><a name="b1840232010507"></a><code>const float *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>.</p>
<p id="p12923104514555"><a name="p12923104514555"></a><a name="p12923104514555"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a><code>float threshold</code></strong>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a><code>int topk</code></strong>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b662915439408"><a name="b662915439408"></a><a name="b662915439408"></a><code>int *num</code></strong>: Number of base library vectors that meet the threshold condition for each query. The length is <code>n</code>.</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b15701310524"><a name="b15701310524"></a><a name="b15701310524"></a><code>idx_t *indices</code></strong>: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.</p>
<p id="p03841120175217"><a name="p03841120175217"></a><a name="p03841120175217"></a><strong id="b1581125094017"><a name="b1581125094017"></a><a name="b1581125094017"></a><code>float *distances</code></strong>: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b8300175210408"><a name="b8300175210408"></a><a name="b8300175210408"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a><code>topk</code></strong>: The value must be in <code>(0, 1024]</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table910711421721"></a>
<table><tbody><tr id="row13108642623"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1610817426217"><a name="p1610817426217"></a><a name="p1610817426217"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1013445511210"><a name="p1013445511210"></a><a name="p1013445511210"></a><code>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row141085421821"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1510864219218"><a name="p1510864219218"></a><a name="p1510864219218"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1810818421423"><a name="p1810818421423"></a><a name="p1810818421423"></a>Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</p>
</td>
</tr>
<tr id="row12108942725"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p131081342022"><a name="p131081342022"></a><a name="p131081342022"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21081142426"><a name="p21081142426"></a><a name="p21081142426"></a><strong id="b810811421722"><a name="b810811421722"></a><a name="b810811421722"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p16108194212219"><a name="p16108194212219"></a><a name="p16108194212219"></a><strong id="b026811345317"><a name="b026811345317"></a><a name="b026811345317"></a><code>const float16_t *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>.</p>
<p id="p410824217215"><a name="p410824217215"></a><a name="p410824217215"></a><strong id="b1510814425212"><a name="b1510814425212"></a><a name="b1510814425212"></a><code>float threshold</code></strong>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.</p>
<p id="p121087424218"><a name="p121087424218"></a><a name="p121087424218"></a><strong id="b31086422216"><a name="b31086422216"></a><a name="b31086422216"></a><code>int topk</code></strong>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.</p>
<p id="p21081342623"><a name="p21081342623"></a><a name="p21081342623"></a><strong id="b20108134216211"><a name="b20108134216211"></a><a name="b20108134216211"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p51081421027"><a name="p51081421027"></a><a name="p51081421027"></a><strong id="b101081421825"><a name="b101081421825"></a><a name="b101081421825"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row1010816424210"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p81081742228"><a name="p81081742228"></a><a name="p81081742228"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p121088421922"><a name="p121088421922"></a><a name="p121088421922"></a><strong id="b11108142324"><a name="b11108142324"></a><a name="b11108142324"></a><code>int *num</code></strong>: Number of base library vectors that meet the threshold condition for each query. The length is <code>n</code>.</p>
<p id="p1310817428215"><a name="p1310817428215"></a><a name="p1310817428215"></a><strong id="b12433151942"><a name="b12433151942"></a><a name="b12433151942"></a><code>idx_t *indices</code></strong>: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.</p>
<p id="p6108194217212"><a name="p6108194217212"></a><a name="p6108194217212"></a><strong id="b81081429211"><a name="b81081429211"></a><a name="b81081429211"></a><code>float *distances</code></strong>: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</p>
</td>
</tr>
<tr id="row1810854219219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p310810423210"><a name="p310810423210"></a><a name="p310810423210"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p810819421524"><a name="p810819421524"></a><a name="p810819421524"></a><strong id="b710815421022"><a name="b710815421022"></a><a name="b710815421022"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row1110811421218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p191086421521"><a name="p191086421521"></a><a name="p191086421521"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul2010818425217"></a><a name="ul2010818425217"></a><ul id="ul2010818425217"><li><strong id="b1810834211215"><a name="b1810834211215"></a><a name="b1810834211215"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><strong id="b610974212213"><a name="b610974212213"></a><a name="b610974212213"></a><code>topk</code></strong>: The value must be in <code>(0, 1024]</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul01091542821"></a><a name="ul01091542821"></a><ul id="ul01091542821"><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SetNTotal`<a name="en-us_TOPIC_0000002514776045"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR SetNTotal(int n);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p7313759183119"><a name="p7313759183119"></a><a name="p7313759183119"></a>Provides an interface for adjusting the <code>ntotal</code> count externally.</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>After base library vectors are added, the <code>Index</code> internally updates the <code>ntotal</code> value according to the largest inserted index, but it does not record which regions in the range <code>[0, ntotal]</code> are invalid. Therefore, the <code>RemoveFeatures</code> operation does not change the <code>ntotal</code> value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set <code>ntotal</code> manually. This reduces the operator workload within a controllable range and improves interface performance.</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>For example, if 100 vectors are inserted and the base library indices range from <code>0</code> to <code>99</code>, <code>ntotal = 100</code>. If you delete the base library entries with indices from <code>80</code> to <code>90</code>, the <code>ntotal</code> value inside the <code>Index</code> remains unchanged and can only be set to a value in <code>[ntotal, capacity]</code>. If you then delete the base library entries with indices from <code>90</code> to <code>99</code>, you can manually set <code>ntotal</code> to a value in <code>[80, capacity]</code>. When you set it to <code>80</code>, the amount of base library data involved in comparison decreases by 20 vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1215142783714"><a name="b1215142783714"></a><a name="b1215142783714"></a><code>int n</code></strong>: The maximum base library index managed by the service side, plus <code>1</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</p>
</td>
</tr>
</tbody>
</table>

## `UpdateFeatures`<a name="en-us_TOPIC_0000002516314733"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p119217478565"><a name="p119217478565"></a><a name="p119217478565"></a><code>APP_ERROR UpdateFeatures (int n, const float16_t *features, const idx_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Updates <code>n</code> feature vectors with the specified indices in the feature library. If a feature vector does not exist at an index, the API adds it. If a feature vector already exists at an index, the API updates it.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a><code>int n</code></strong>: Number of feature vectors to insert.</p>
<p id="p042220329586"><a name="p042220329586"></a><a name="p042220329586"></a><strong id="b17419938175818"><a name="b17419938175818"></a><a name="b17419938175818"></a><code>const float16_t *features</code></strong>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p18422153235817"><a name="p18422153235817"></a><a name="p18422153235817"></a><strong id="b5921164425815"><a name="b5921164425815"></a><a name="b5921164425815"></a><code>const idx_t *indices</code></strong>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>.</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table19567183517113"></a>
<table><tbody><tr id="row145678353110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1056713351120"><a name="p1056713351120"></a><a name="p1056713351120"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13567835018"><a name="p13567835018"></a><a name="p13567835018"></a><code>APP_ERROR UpdateFeatures(int n, const float *features, const idx_t *indices);</code></p>
</td>
</tr>
<tr id="row1256719351818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p175675351511"><a name="p175675351511"></a><a name="p175675351511"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p20567123520115"><a name="p20567123520115"></a><a name="p20567123520115"></a>Updates <code>n</code> feature vectors with the specified indices in the feature library. If a feature vector does not exist at an index, the API adds it. If a feature vector already exists at an index, the API updates it.</p>
</td>
</tr>
<tr id="row756713352110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1456793520110"><a name="p1456793520110"></a><a name="p1456793520110"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p2567133518120"><a name="p2567133518120"></a><a name="p2567133518120"></a><strong id="b65671335119"><a name="b65671335119"></a><a name="b65671335119"></a><code>int n</code></strong>: Number of feature vectors to insert.</p>
<p id="p165673357114"><a name="p165673357114"></a><a name="p165673357114"></a><strong id="b3277341929"><a name="b3277341929"></a><a name="b3277341929"></a><code>const float *features</code></strong>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p175671135216"><a name="p175671135216"></a><a name="p175671135216"></a><strong id="b2567173515117"><a name="b2567173515117"></a><a name="b2567173515117"></a><code>const idx_t *indices</code></strong>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</p>
</td>
</tr>
<tr id="row456710351212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1556715352111"><a name="p1556715352111"></a><a name="p1556715352111"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1956716351718"><a name="p1956716351718"></a><a name="p1956716351718"></a>None</p>
</td>
</tr>
<tr id="row155678359112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p115677351519"><a name="p115677351519"></a><a name="p115677351519"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1056773515119"><a name="p1056773515119"></a><a name="p1056773515119"></a><strong id="b1456793518119"><a name="b1456793518119"></a><a name="b1456793518119"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row65673351912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9567135715"><a name="p9567135715"></a><a name="p9567135715"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1567203512115"></a><a name="ul1567203512115"></a><ul id="ul1567203512115"><li><strong id="b556783510117"><a name="b556783510117"></a><a name="b556783510117"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>.</li><li><strong id="b1456713512117"><a name="b1456713512117"></a><a name="b1456713512117"></a><code>n</code></strong>: The value must be in <code>(0, capacity]</code>.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>
