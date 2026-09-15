# `AscendIndexCluster`<a id="en-us_TOPIC_0000001614744825"></a>

## Overview<a name="en-us_TOPIC_0000001564586790"></a>

<code>AscendIndexCluster</code> requires <a href="#init"><code>Init</code></a> to initialize the specified resources. After initialization, it allocates a complete memory space to store the base library. After use, call [Finalize](#finalize) to release the resources.

<code>AscendIndexCluster</code> supports only the vector inner-product distance type in standard mode on <term>Atlas inference products</term>. It depends on Flat and AICPU operators. For details, see [Flat](../../05_user_guide.md#flat) and [AICPU](../../05_user_guide.md#aicpu).

It supports multithreaded concurrent calls. To enable this feature, set the <code>MX_INDEX_MULTITHREAD</code> environment variable to <code>1</code>, that is, run <code>export MX_INDEX_MULTITHREAD=1</code>. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

## `AddFeatures`<a name="en-us_TOPIC_0000001614746533"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122877269522"><a name="p122877269522"></a><a name="p122877269522"></a><code>APP_ERROR AddFeatures(int n, const float *features, const uint32_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16287112617527"><a name="p16287112617527"></a><a name="p16287112617527"></a>Inserts <code>n</code> feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, This API updates it.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1728872675217"><a name="p1728872675217"></a><a name="p1728872675217"></a><code>int n</code>: Number of feature vectors to insert.</p>
<p id="p172889261527"><a name="p172889261527"></a><a name="p172889261527"></a><code>const float *features</code>: Feature vectors to insert. The length is <code>n</code> multiplied by the vector dimension <code>dim</code>.</p>
<p id="p6288192619521"><a name="p6288192619521"></a><a name="p6288192619521"></a><code>const uint32_t *indices</code>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1628882619521"><a name="p1628882619521"></a><a name="p1628882619521"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p528872613525"><a name="p528872613525"></a><a name="p528872613525"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li><code>indices</code>: The index of each feature must be in [0, <code>capacity</code> ), and <code>indices</code> must be continuous.</li><li><code>n</code>: Must be in (0, <code>capacity</code> ].</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table772538154310"></a>
<table><tbody><tr id="row97256854317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p372568194310"><a name="p372568194310"></a><a name="p372568194310"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19725148124318"><a name="p19725148124318"></a><a name="p19725148124318"></a><code>APP_ERROR AddFeatures(int n, const uint16_t *features, const int64_t *indices);</code></p>
</td>
</tr>
<tr id="row9725983433"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1372519814431"><a name="p1372519814431"></a><a name="p1372519814431"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p47259810431"><a name="p47259810431"></a><a name="p47259810431"></a>Inserts <code>n</code> feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, This API updates it.</p>
</td>
</tr>
<tr id="row1272528104315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p167251682439"><a name="p167251682439"></a><a name="p167251682439"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1872513819432"><a name="p1872513819432"></a><a name="p1872513819432"></a><code>int n</code>: Number of feature vectors to insert.</p>
<p id="p87251087438"><a name="p87251087438"></a><a name="p87251087438"></a><code>const uint16_t *features</code>: Feature vectors to insert. The length is <code>n</code> multiplied by the vector dimension <code>dim</code>.</p>
<p id="p672518884310"><a name="p672518884310"></a><a name="p672518884310"></a><code>const int64_t *indices</code>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</p>
</td>
</tr>
<tr id="row187251389432"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p20725788435"><a name="p20725788435"></a><a name="p20725788435"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p3725198194318"><a name="p3725198194318"></a><a name="p3725198194318"></a>None</p>
</td>
</tr>
<tr id="row672517820435"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p57254812434"><a name="p57254812434"></a><a name="p57254812434"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p107256824313"><a name="p107256824313"></a><a name="p107256824313"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row1972548114318"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9725582431"><a name="p9725582431"></a><a name="p9725582431"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul87251988433"></a><a name="ul87251988433"></a><ul id="ul87251988433"><li><code>indices</code>: The index of each feature must be in [0, <code>capacity</code> ).</li><li><code>n</code>: Must be in (0, <code>capacity</code> ].</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndexCluster`<a name="en-us_TOPIC_0000001564746410"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p9608143314716"><a name="p9608143314716"></a><a name="p9608143314716"></a><code>AscendIndexCluster();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1760814333474"><a name="p1760814333474"></a><a name="p1760814333474"></a>Constructor of <code>AscendIndexCluster</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p204291821488"><a name="p204291821488"></a><a name="p204291821488"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p194301211486"><a name="p194301211486"></a><a name="p194301211486"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1743014244819"><a name="p1743014244819"></a><a name="p1743014244819"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20430172184812"><a name="p20430172184812"></a><a name="p20430172184812"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table15621560282"></a>
<table><tbody><tr id="row1256265642816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p20562165614286"><a name="p20562165614286"></a><a name="p20562165614286"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p29285682519"><a name="p29285682519"></a><a name="p29285682519"></a><code>AscendIndexCluster(const AscendIndexCluster&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row756235619282"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p95621656152818"><a name="p95621656152818"></a><a name="p95621656152818"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p538993116244"><a name="p538993116244"></a><a name="p538993116244"></a>Declares this <code>Index</code> copy constructor as deleted, making the type non-copyable.</p>
</td>
</tr>
<tr id="row356225619283"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1956245682817"><a name="p1956245682817"></a><a name="p1956245682817"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p998472882614"><a name="p998472882614"></a><a name="p998472882614"></a><code>const AscendIndexCluster&amp;</code>: <code>AscendIndexCluster</code> object.</p>
</td>
</tr>
<tr id="row55621556102815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p135627560287"><a name="p135627560287"></a><a name="p135627560287"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row0562256142813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1656225622819"><a name="p1656225622819"></a><a name="p1656225622819"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row145621856162814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19562105616284"><a name="p19562105616284"></a><a name="p19562105616284"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~AscendIndexCluster`<a name="en-us_TOPIC_0000002399598393"></a>

<a name="table179216322487"></a>
<table><tbody><tr id="row2092173214484"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3929320480"><a name="p3929320480"></a><a name="p3929320480"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p118119814816"><a name="p118119814816"></a><a name="p118119814816"></a><code>virtual ~AscendIndexCluster() = default;</code></p>
</td>
</tr>
<tr id="row092163217481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p493532184819"><a name="p493532184819"></a><a name="p493532184819"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2811384485"><a name="p2811384485"></a><a name="p2811384485"></a>Destructor of <code>AscendIndexCluster</code>.</p>
</td>
</tr>
<tr id="row1193163244813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p19316328485"><a name="p19316328485"></a><a name="p19316328485"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1581178114813"><a name="p1581178114813"></a><a name="p1581178114813"></a>None</p>
</td>
</tr>
<tr id="row9931932104818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p793173212484"><a name="p793173212484"></a><a name="p793173212484"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p148111782488"><a name="p148111782488"></a><a name="p148111782488"></a>None</p>
</td>
</tr>
<tr id="row89333214481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p89316323489"><a name="p89316323489"></a><a name="p89316323489"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p58111481480"><a name="p58111481480"></a><a name="p58111481480"></a>None</p>
</td>
</tr>
<tr id="row49311328486"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p993113234818"><a name="p993113234818"></a><a name="p993113234818"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4811148104812"><a name="p4811148104812"></a><a name="p4811148104812"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `ComputeDistanceByIdx`<a name="en-us_TOPIC_0000002446061685"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p68995910575"><a name="p68995910575"></a><a name="p68995910575"></a><code>APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num, const uint32_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a><code>ComputeDistance</code> calculates the distance between the query vectors and all base-library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distance between the query vectors and the base-library vectors at the given indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the interface returns the mapped top-<code>k</code> results.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><code>int n</code>: Number of query feature vectors.</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><code>const uint16_t *queries</code>: Query feature vectors. The valid length is <code>n * dim</code>, and <code>dim</code> must be the same as the dimension specified during initialization.</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><code>const int *num</code>: Number of base-library feature vectors to compare for each query. The length is <code>n</code>.</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><code>const uint32_t *indices</code>: Indices of the base-library feature vectors to compare. The number of base-library vectors to compare can differ for each query. Valid vector indices must be stored continuously from front to back, and the space usage must be padded according to the maximum <code>num</code>. The length of <code>indices</code> is <code>n * max(num)</code>.</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><code>unsigned int tableLen</code>: Mapping-table length. The default value is <code>0</code>, which means that no mapping is performed. Currently, the supported mapping-table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><code>const float *table</code>: Mapping-table pointer that points to valid mapped values of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p057182814222"><a name="p057182814222"></a><a name="p057182814222"></a><code>float *distances</code>: Distances between the query vectors and the selected base-library vectors. For each query, valid distances are recorded continuously from front to back, and the space usage is padded according to the maximum <code>num</code>. The total length is <code>n * max(num)</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong>: Must be in the range (0, <i><span class="varname" id="varname82723561324"><a name="varname82723561324"></a><a name="varname82723561324"></a>capacity</span></i>].</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>: User-specified. The length is <strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>n</strong>, and the <strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>num</strong> value for each query must be in [0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>].</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong>: The index of each feature must be in [0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>).</li><li>Example parameter values: <strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong> = 3, <strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>[3] = {1, 3, 5} means that the three queries compare against 1, 3, and 5 base-library vectors respectively. If <strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>max(num)</strong> = 5, then the space pointed to by <strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong> is aligned to 5, and the total size is <strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>3 * 5 * sizeof(idx_t)</strong> bytes, for example {{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}.</li><li>When both <span class="parmname" id="en-us_topic_0000001456535116_parmname391111228612"><a name="en-us_topic_0000001456535116_parmname391111228612"></a><a name="en-us_topic_0000001456535116_parmname391111228612"></a>tableLen</span> and <span class="parmname" id="en-us_topic_0000001456535116_parmname19267121920619"><a name="en-us_topic_0000001456535116_parmname19267121920619"></a><a name="en-us_topic_0000001456535116_parmname19267121920619"></a>table</span> meet the requirements, the interface maps the computed <strong id="en-us_topic_0000001456535116_b5371616181211"><a name="en-us_topic_0000001456535116_b5371616181211"></a><a name="en-us_topic_0000001456535116_b5371616181211"></a>distance</strong> values:<p id="en-us_topic_0000001456535116_p1129513513121"><a name="en-us_topic_0000001456535116_p1129513513121"></a><a name="en-us_topic_0000001456535116_p1129513513121"></a>First, normalize <strong id="en-us_topic_0000001456535116_b13840714131216"><a name="en-us_topic_0000001456535116_b13840714131216"></a><a name="en-us_topic_0000001456535116_b13840714131216"></a>distance</strong> to a floating-point value <strong id="en-us_topic_0000001456535116_b7555131016121"><a name="en-us_topic_0000001456535116_b7555131016121"></a><a name="en-us_topic_0000001456535116_b7555131016121"></a>f1</strong> in [0, 1]. Then multiply <strong id="en-us_topic_0000001456535116_b199806129123"><a name="en-us_topic_0000001456535116_b199806129123"></a><a name="en-us_topic_0000001456535116_b199806129123"></a>f1</strong> by <span class="parmname" id="en-us_topic_0000001456535116_parmname14917143791"><a name="en-us_topic_0000001456535116_parmname14917143791"></a><a name="en-us_topic_0000001456535116_parmname14917143791"></a>tableLen</span> and round it down to obtain an integer index in [0, <strong id="en-us_topic_0000001456535116_b1399121919123"><a name="en-us_topic_0000001456535116_b1399121919123"></a><a name="en-us_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]. Next, use the integer index as an offset to read the corresponding <strong id="en-us_topic_0000001456535116_b12230192011219"><a name="en-us_topic_0000001456535116_b12230192011219"></a><a name="en-us_topic_0000001456535116_b12230192011219"></a>score</strong> from the memory space pointed to by <span class="parmname" id="en-us_topic_0000001456535116_parmname266193771110"><a name="en-us_topic_0000001456535116_parmname266193771110"></a><a name="en-us_topic_0000001456535116_parmname266193771110"></a>table</span>. This completes the mapping and stores <strong id="en-us_topic_0000001456535116_b1622952141216"><a name="en-us_topic_0000001456535116_b1622952141216"></a><a name="en-us_topic_0000001456535116_b1622952141216"></a>score</strong> in <span class="parmname" id="en-us_topic_0000001456535116_parmname106381556121113"><a name="en-us_topic_0000001456535116_parmname106381556121113"></a><a name="en-us_topic_0000001456535116_parmname106381556121113"></a>distance</span>.</p>
<p id="en-us_topic_0000001456535116_p340315471018"><a name="en-us_topic_0000001456535116_p340315471018"></a><a name="en-us_topic_0000001456535116_p340315471018"></a>The index mapping formula can be abstracted as ((CosDistance + 1) / 2) * tableLen.</p>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `ComputeDistanceByThreshold`<a name="en-us_TOPIC_0000001615066169"></a>

> [!NOTE]
> This API must be used together with [AddFeatures\(int n, const float \*features, const uint32\_t \*indices\);](#addfeatures).

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.1.1 "><p id="p15352133820537"><a name="p15352133820537"></a><a name="p15352133820537"></a><code>APP_ERROR ComputeDistanceByThreshold(const std::vector&lt;uint32_t&gt; &amp;queryIdxArr, uint32_t codeStartIdx,  uint32_t codeNum, float threshold, bool aboveFilter, std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr, std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.2.1 "><p id="p1935283855318"><a name="p1935283855318"></a><a name="p1935283855318"></a>Calculates the distances between the queried feature vectors in the base library and the specified base-library feature vectors, then filters by threshold and returns the distances and labels that meet the conditions.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.3.1 "><p id="p1535283885320"><a name="p1535283885320"></a><a name="p1535283885320"></a><code>const std::vector&lt;uint32_t&gt; &amp;queryIdxArr</code>: Indices of the vectors to query in the base library.</p>
<p id="p1835213385530"><a name="p1835213385530"></a><a name="p1835213385530"></a><code>uint32_t codeStartIdx</code>: Starting index of the base library vectors for distance calculation.</p>
<p id="p23526383537"><a name="p23526383537"></a><a name="p23526383537"></a><code>uint32_t codeNum</code>: Number of base-library vectors for distance calculation.</p>
<p id="p203521138195313"><a name="p203521138195313"></a><a name="p203521138195313"></a><code>float threshold</code>: Threshold used for filtering. Distances smaller than the threshold are filtered out.</p>
<p id="p1435223815538"><a name="p1435223815538"></a><a name="p1435223815538"></a><code>bool aboveFilter</code>: Reserved parameter.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.4.1 "><p id="p13522382534"><a name="p13522382534"></a><a name="p13522382534"></a><code>std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr</code>: Two-dimensional array that returns the distances between each query vector and the base-library vectors that meet the threshold condition.</p>
<p id="p83524381530"><a name="p83524381530"></a><a name="p83524381530"></a><code>std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr</code>: Two-dimensional array that returns the indices of the base-library vectors that meet the threshold condition for each query vector.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.5.1 "><p id="p735215385536"><a name="p735215385536"></a><a name="p735215385536"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.6.1 "><a name="ul1624216519526"></a><a name="ul1624216519526"></a><ul id="ul1624216519526"><li>The lengths of <code>queryIdxArr</code>, <code>resDistArr</code>, and <code>resIdxArr</code> must be the same, that is, <code>queryIdxArr.size() == resDistArr.size()</code>.</li><li><code>queryIdxArr.size()</code> must be greater than <code>0</code> and less than or equal to <code>ntotal</code>.</li><li><code>codeNum</code> must be greater than <code>0</code> and less than or equal to <code>ntotal</code>.</li><li><code>codeStartIdx + codeNum</code> must not exceed <code>ntotal</code> (the base-library size).</li><li><code>codeStartIdx</code> must be greater than or equal to <code>0</code> and less than or equal to <code>ntotal</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `Finalize`<a name="en-us_TOPIC_0000001614906601"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p377414915513"><a name="p377414915513"></a><a name="p377414915513"></a><code>void Finalize();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Releases feature-library management resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p877484919511"><a name="p877484919511"></a><a name="p877484919511"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14774134917519"><a name="p14774134917519"></a><a name="p14774134917519"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1277416496515"><a name="p1277416496515"></a><a name="p1277416496515"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16774184918516"><a name="p16774184918516"></a><a name="p16774184918516"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `GetFeatures`<a name="en-us_TOPIC_0000002412742482"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR GetFeatures(int n, uint16_t *features, const int64_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Retrieves <code>n</code> feature vectors at the specified indices.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><code>int n</code>: Number of base-library vectors to retrieve.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><code>const int64_t *indices</code>: Indices corresponding to the feature vectors. The length is <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><code>uint16_t *features</code>: Feature vectors corresponding to the queried indices. The length is <code>n * vector dimension dim</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ), and <code>ntotal</code> can be obtained through the <code>GetNTotal</code> interface.</li><li><code>n</code>: Must be in [0, <code>capacity</code> ].</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetNTotal`<a name="en-us_TOPIC_0000002412582646"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p61958153167"><a name="p61958153167"></a><a name="p61958153167"></a><code>int GetNTotal() const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Queries the theoretical maximum number of feature vectors in the current feature library. If the inserted feature-vector indices are continuous, <code>ntotal</code> is equal to the number of feature vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a><code>int ntotal</code>: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><code>int</code>: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Init`<a name="en-us_TOPIC_0000001614866169"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p463919713918"><a name="p463919713918"></a><a name="p463919713918"></a><code>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p4783164014487"><a name="p4783164014487"></a><a name="p4783164014487"></a>Initialization function of <code>AscendIndexCluster</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14783940124815"><a name="p14783940124815"></a><a name="p14783940124815"></a><code>int dim</code>: Dimension of the feature vectors managed by <code>AscendIndexCluster</code>.</p>
<p id="p178314407484"><a name="p178314407484"></a><a name="p178314407484"></a><code>int capacity</code>: Maximum base-library capacity. The interface allocates <code>capacity * dim * sizeof(fp16)</code> bytes of memory based on the value of <code>capacity</code>.</p>
<p id="p1478354004816"><a name="p1478354004816"></a><a name="p1478354004816"></a><code>faiss::MetricType metricType</code>: Feature-distance category, including vector inner product, Euclidean distance, and cosine similarity.</p>
<p id="p978314010482"><a name="p978314010482"></a><a name="p978314010482"></a><code>const std::vector&lt;int&gt; &amp;deviceList</code>: Device-side resource configuration.</p>
<p id="p278364014481"><a name="p278364014481"></a><a name="p278364014481"></a><code>int64_t resourceSize</code>: Size of the preallocated memory pool on the device side, in bytes. This memory stores intermediate results during computation and is used to avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>-1</code>, which means <code>128 MB</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1378314016485"><a name="p1378314016485"></a><a name="p1378314016485"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p4783174017482"><a name="p4783174017482"></a><a name="p4783174017482"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3783154018480"></a><a name="ul3783154018480"></a><ul id="ul3783154018480"><li><code>dim</code> ∈ {32, 64, 128, 256, 384, 512}.</li><li><code>metricType</code>: <code>AscendIndexCluster</code> currently implements only vector inner-product distance, which means that only <code>faiss::MetricType::METRIC_INNER_PRODUCT</code> is supported.</li><li>The maximum memory that can be allocated for the base library is <code>12,288,000,000</code> bytes, and the value range of <code>capacity</code> is [0, 12000000].</li><li>For example, for a base-library vector with 512 dimensions and the FP16 type, the maximum supported <code>capacity</code> is 12 million ( <code>12288000000 / (512 * sizeof(fp_16))</code> ).</li><li>For base-library vectors with 256 dimensions and the FP16 type, even though the memory constraint supports a larger <code>capacity</code>, the maximum <code>capacity</code> can still be only 12 million.</li><li>Only single-card configuration is supported. Multi-card configuration is not supported yet, so <code>deviceList.size()</code> must equal <code>1</code>.</li><li><code>resourceSize</code> can be <code>-1</code> or a value in [134217728, 4294967296], which is equivalent to [128 MB, 4096 MB]. This parameter is determined jointly by the base-library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `operator =`<a name="en-us_TOPIC_0000001897100377"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19940753122510"><a name="p19940753122510"></a><a name="p19940753122510"></a><code>AscendIndexCluster&amp; operator=(const AscendIndexCluster&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3198107264"><a name="p3198107264"></a><a name="p3198107264"></a>Declares this <code>Index</code> copy assignment operator as deleted, making the type non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6538114402614"><a name="p6538114402614"></a><a name="p6538114402614"></a><code>const AscendIndexCluster&amp;</code>: <code>AscendIndexCluster</code> object.</p>
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

## `RemoveFeatures`<a name="en-us_TOPIC_0000002446181741"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p457211594914"><a name="p457211594914"></a><a name="p457211594914"></a><code>APP_ERROR RemoveFeatures(int n, const int64_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Removes <code>n</code> feature vectors at the specified indices from the vector library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><code>int n</code>: Number of feature vectors to remove.</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><code>const int64_t *indices</code>: Indices corresponding to the feature vectors. The length is <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><code>indices</code>: The index of each feature must be in [0, <code>ntotal</code> ), and <code>ntotal</code> can be obtained through the <code>GetNTotal</code> interface.</li><li><code>n</code>: Must be in [0, <code>capacity</code> ].</li><li><code>indices</code> must be a non-null pointer, and its length must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchByThreshold`<a name="en-us_TOPIC_0000002446061689"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p126941862232"><a name="p126941862232"></a><a name="p126941862232"></a><code>APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num, int64_t * indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p138361625164412"><a name="p138361625164412"></a><a name="p138361625164412"></a>Adds threshold filtering on top of <code>Search</code> and returns only the results that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the interface returns the mapped top-<code>k</code> results.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><code>int n</code>: Number of feature vectors to query.</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><code>const uint16_t *queries</code>: Query feature vectors. The length is <code>n * dim</code>.</p>
<p id="p11518191412248"><a name="p11518191412248"></a><a name="p11518191412248"></a><code>float threshold</code>: Threshold used for filtering. The interface does not restrict the value range. If you pass a mapping table, the interface first maps the distance to a score and then filters by <code>threshold</code>.</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><code>int topk</code>: Sorts the comparison distances between the query and the base library, then returns the top <code>k</code> results.</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><code>unsigned int tableLen</code>: Mapping-table length. The default value is <code>0</code>, which means that no mapping is performed. Currently, the supported mapping-table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><code>const float *table</code>: Mapping-table pointer that points to valid mapped values of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18962112242616"><a name="p18962112242616"></a><a name="p18962112242616"></a><code>int *num</code>: Number of base-library vectors that meet the threshold condition for each query feature vector. The length is <code>n</code>.</p>
<p id="p1796272252611"><a name="p1796272252611"></a><a name="p1796272252611"></a><code>int64_t *indices</code>: Indices of base-library vectors that meet the threshold condition. For each query, matching indices are recorded from front to back and the space is padded according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.</p>
<p id="p296222222618"><a name="p296222222618"></a><a name="p296222222618"></a><code>float *distances</code>: Distances between the base-library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><code>n</code>: Must be in the range (0, <code>capacity</code> ].</li><li><code>topk</code>: Must be in the range (0, 1024].</li><li>When both <code>tableLen</code> and <code>table</code> meet the requirements, the interface maps the computed <code>distance</code> values:<p id="en-us_topic_0000001456535116_p1129513513121"><a name="en-us_topic_0000001456535116_p1129513513121"></a><a name="en-us_topic_0000001456535116_p1129513513121"></a>First, normalize <code>distance</code> to a floating-point value <code>f1</code> in [0, 1]. Then multiply <code>f1</code> by <code>tableLen</code> and round it down to obtain an integer index in [0, <code>tableLen</code>]. Next, use the integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>. This completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="en-us_topic_0000001456535116_p340315471018"><a name="en-us_topic_0000001456535116_p340315471018"></a><a name="en-us_topic_0000001456535116_p340315471018"></a>The index mapping formula can be abstracted as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SetNTotal`<a name="en-us_TOPIC_0000002412742486"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR SetNTotal(int n);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p7313759183119"><a name="p7313759183119"></a><a name="p7313759183119"></a>Provides an external way to adjust the <code>ntotal</code> count.</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>After base-library vectors are added, the <code>Index</code> internally updates <code>ntotal</code> according to the maximum inserted index. However, it does not record which areas in the range [0, <code>ntotal</code> ] are invalid space. Therefore, the <code>RemoveFeatures</code> operation does not change the value of <code>ntotal</code>. If you explicitly record the maximum base-library index after add and remove operations in the service layer, you can set <code>ntotal</code> manually. This can reduce the amount of work performed by the operators within a controllable range and improve interface performance.</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>For example, if you currently insert 100 vectors with base-library indices from 0 to 99, then <code>ntotal = 100</code>. If you delete the base-library vectors with indices from 80 to 90, the internal <code>ntotal</code> of <code>Index</code> remains unchanged and can only be set to a value in [ <code>ntotal</code>, <code>capacity</code> ]. If you then delete the base-library vectors with indices from 90 to 99, you can manually set <code>ntotal</code> to a value in [80, <code>capacity</code> ]. When you set it to <code>80</code>, the amount of base-library data participating in the comparison is effectively reduced by 20 vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><code>int n</code>: Maximum base-library index plus 1, managed by the user in the service layer.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><code>APP_ERROR</code>: Return status of the call. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a><code>n</code>: Must be in the range [0, <code>capacity</code> ].</p>
</td>
</tr>
</tbody>
</table>
