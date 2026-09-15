# `IndexILFlat`<a name="en-us_TOPIC_0000001506614925"></a>

## Function Description<a name="en-us_TOPIC_0000001506414785"></a>

`IndexILFlat` inherits from `IndexIL` and is a pure Device-side retrieval solution. It uses resources such as the Ascend AI Processor and AI Core to enable each API. The program must be compiled on the Host into a binary, and then the binary and related runtime dependencies are deployed to the Device for execution. `IndexILFlat` uses the <a href="#init">`Init`</a> interface to initialize the specified resources. After initialization, it allocates a contiguous block of memory to store the base library. After use, call the <a href="#finalize">`Finalize`</a> interface to release the resources.

`IndexILFlat` currently receives only functional and performance maintenance on <term>Atlas inference products</term>. The base library and query vectors must be normalized by the user, and the interfaces currently support only the inner product distance. For details, see <a href="#indexilflat">`IndexILFlat`</a>. Successful execution of this algorithm depends on the OM file of the TIK operator. In a pure-Device scenario, ensure that the deployed OM file is generated from the Index SDK deliverable and has not been tampered with.

Multithreaded concurrent calls are supported. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

## `AddFeatures`<a name="en-us_TOPIC_0000001456854852"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Inserts <code>n</code> feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, the API updates it.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b645815911297"><a name="b645815911297"></a><a name="b645815911297"></a><code>int n</code></strong>: Number of feature vectors to insert.</p>
<p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b118193183015"><a name="b118193183015"></a><a name="b118193183015"></a><code>const float16_t *features</code></strong>: Feature vectors to insert. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b119812563013"><a name="b119812563013"></a><a name="b119812563013"></a><code>const idx_t *indices</code></strong>: Indices of the feature vectors to insert. The valid length is <code>n</code>.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul10674191294110"></a><a name="ul10674191294110"></a><ul id="ul10674191294110"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a><code>indices</code></strong>: Each feature index must be in <code>[0, capacity)</code>.</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `ComputeDistance`<a name="en-us_TOPIC_0000001456535116"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1750033215518"><a name="p1750033215518"></a><a name="p1750033215518"></a><strong id="b668194911337"><a name="b668194911337"></a><a name="b668194911337"></a><code>float *distances</code></strong>: External memory that stores the distances between query vectors and base library vectors. The total length should be <code>n * nTotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, <code>ntotal</code> rounded up to a multiple of 16).</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul167968714447"></a><a name="ul167968714447"></a><ul id="ul167968714447"><li><strong id="b1141410121444"><a name="b1141410121444"></a><a name="b1141410121444"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><strong id="b429253634917"><a name="b429253634917"></a><a name="b429253634917"></a><code>distances</code></strong>: The required buffer length is <code>n * ntotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, the result of rounding <code>ntotal</code> up to a multiple of 16. The valid comparison distances for each query are stored in the first <code>ntotal</code> positions, and the padded data has no practical meaning).<p id="p0715202405018"><a name="p0715202405018"></a><a name="p0715202405018"></a>You are advised to use the <strong id="b1894113045017"><a name="b1894113045017"></a><a name="b1894113045017"></a><code>aclrtmalloc</code></strong> interface, which can allocate full physical memory for use and optimize processing latency.</p>
</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li><li><code>queries</code> and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `ComputeDistanceByIdx`<a name="en-us_TOPIC_0000001456694920"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p7384196195012"><a name="p7384196195012"></a><a name="p7384196195012"></a><code>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Similar to <code>ComputeDistance</code>, except that <code>ComputeDistance</code> calculates the distances between query vectors and all base library vectors, whereas <code>ComputeDistanceByIdx</code> calculates only the distances between query vectors and the base library vectors at the given indices. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), the mapped <code>topk</code> results are returned.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1178514265435"><a name="b1178514265435"></a><a name="b1178514265435"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b79742028204313"><a name="b79742028204313"></a><a name="b79742028204313"></a><code>const float16_t *queries</code></strong>: Feature vectors to query. The valid length is <code>n * dim</code>, and <code>dim</code> must match the dimension specified during initialization.</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a><code>const int *num</code></strong>: Number of base library feature vectors to compare for each query. The length is <code>n</code>.</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b632503394315"><a name="b632503394315"></a><a name="b632503394315"></a><code>const idx_t *indices</code></strong>: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum <code>num</code> value. The length of <code>indices</code> is <code>n * max(num)</code>.</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a><code>num</code></strong>: User-specified length <code>n</code>, and each <code>num</code> value must be in <code>[0, ntotal]</code>.</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a><code>indices</code></strong>: Each feature index must be in <code>[0, ntotal)</code>.</li><li>API parameter configuration example: <code>n = 3</code>, <code>num[3] = {1, 3, 5}</code> indicates that the three queries compare with <code>1</code>, <code>3</code>, and <code>5</code> base library vectors respectively. Since <code>max(num) = 5</code>, the storage space pointed to by <code>indices</code> is aligned to 5, and the total size is <code>3 * 5 * sizeof(idx_t)</code> bytes, for example, <code>{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `ComputeDistanceByThreshold`<a name="en-us_TOPIC_0000001506615117"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p137564208498"><a name="p137564208498"></a><a name="p137564208498"></a><code>APP_ERROR ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Adds threshold filtering on top of <code>ComputeDistance</code> and returns only the distances that meet the threshold condition. If you pass a valid mapping table (<code>tableLen &gt; 0</code> and <code>table</code> is a non-null pointer), <code>distances</code> contains the mapped results after threshold filtering.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b6232619123815"><a name="b6232619123815"></a><a name="b6232619123815"></a><code>int n</code></strong>: Number of feature vectors to query.</p>
<p id="p144875610364"><a name="p144875610364"></a><a name="p144875610364"></a><strong id="b6143623203816"><a name="b6143623203816"></a><a name="b6143623203816"></a><code>float16_t *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
<p id="p1924692795017"><a name="p1924692795017"></a><a name="p1924692795017"></a><strong id="b183957257385"><a name="b183957257385"></a><a name="b183957257385"></a><code>float threshold</code></strong>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b124943270387"><a name="b124943270387"></a><a name="b124943270387"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b13344132915381"><a name="b13344132915381"></a><a name="b13344132915381"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b5394194013386"><a name="b5394194013386"></a><a name="b5394194013386"></a><code>int *num</code></strong>: Number of base library vectors that meet the threshold condition for each query, with length <code>n</code>.</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b787564210382"><a name="b787564210382"></a><a name="b787564210382"></a><code>idx_t *indices</code></strong>: Indices of the base library vectors that meet the threshold condition. Each query records valid indices contiguously from front to back, and the space is padded according to <code>ntotalPad</code>. The total length of <code>indices</code> is <code>n * nTotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, <code>ntotal</code> rounded up to a multiple of 16).</p>
<p id="p03841120175217"><a name="p03841120175217"></a><a name="p03841120175217"></a><strong id="b17674164983818"><a name="b17674164983818"></a><a name="b17674164983818"></a><code>float *distances</code></strong>: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of <code>indices</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b292575217384"><a name="b292575217384"></a><a name="b292575217384"></a><code>APP_ERROR</code></strong>: Return status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul192831928125717"></a><a name="ul192831928125717"></a><ul id="ul192831928125717"><li><strong id="b657843016578"><a name="b657843016578"></a><a name="b657843016578"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><strong id="b841218507570"><a name="b841218507570"></a><a name="b841218507570"></a><code>indices</code></strong>: The required buffer length is <code>n * ntotalPad</code> (<code>ntotalPad</code> is <code>(ntotal + 15) / 16 * 16</code>, that is, the result of rounding <code>ntotal</code> up to a multiple of 16. For the <code>i</code>-th query, valid base library indices are stored in the first <code>*(num + i)</code> positions of each <code>ntotalPad</code> block, and the padded data has no practical meaning).</li><li><strong id="b19683195213576"><a name="b19683195213576"></a><a name="b19683195213576"></a><code>distances</code></strong>: The required buffer length is <code>n * ntotalPad</code>.</li><li><code>indices</code> and <code>distances</code> are advised to use the <strong id="b4371184115813"><a name="b4371184115813"></a><a name="b4371184115813"></a><code>aclrtmalloc</code></strong> interface, which can allocate full physical memory for use and optimize processing latency.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `Finalize`<a name="en-us_TOPIC_0000001506414845"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR Finalize() override;</code></p>
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

## `GetFeatures`<a name="en-us_TOPIC_0000001456854992"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Queries the feature vectors with the specified indices for <code>n</code> entries.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a><code>int n</code></strong>: Number of base library vectors to get.</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a><code>const idx_t *indices</code></strong>: Indices corresponding to the <code>n</code> base library vectors to get.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b043314127333"><a name="b043314127333"></a><a name="b043314127333"></a><code>float16_t *features</code></strong>: Feature vectors corresponding to the queried indices. The length is <code>n * dim</code>, where <code>dim</code> is the vector dimension.</p>
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

## `GetNTotal`<a name="en-us_TOPIC_0000001456375336"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a><code>int GetNTotal() const override;</code></p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a><code>int ntotal</code></strong>: See the description.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `IndexILFlat`<a name="en-us_TOPIC_0000001456694872"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a><code>IndexILFlat();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>Constructor of <code>IndexILFlat</code>.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table194381755582"></a>
<table><tbody><tr id="row1438055581"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11438155155815"><a name="p11438155155815"></a><a name="p11438155155815"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1887018312271"><a name="p1887018312271"></a><a name="p1887018312271"></a><code>IndexILFlat(const IndexILFlat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row20438551584"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p74381159583"><a name="p74381159583"></a><a name="p74381159583"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p087012313276"><a name="p087012313276"></a><a name="p087012313276"></a>Declares the copy constructor of <code>IndexILFlat</code> as deleted. Therefore, <code>IndexILFlat</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row24385511589"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p54381519581"><a name="p54381519581"></a><a name="p54381519581"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1786920310278"><a name="p1786920310278"></a><a name="p1786920310278"></a><strong id="b1129362910278"><a name="b1129362910278"></a><a name="b1129362910278"></a><code>const IndexILFlat&amp;</code></strong>: <code>IndexILFlat</code> object.</p>
</td>
</tr>
<tr id="row84387585820"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1643812512585"><a name="p1643812512585"></a><a name="p1643812512585"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p154381517589"><a name="p154381517589"></a><a name="p154381517589"></a>None</p>
</td>
</tr>
<tr id="row043813535813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1443815510581"><a name="p1443815510581"></a><a name="p1443815510581"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184381457585"><a name="p184381457585"></a><a name="p184381457585"></a>None</p>
</td>
</tr>
<tr id="row2043811515580"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1643935185813"><a name="p1643935185813"></a><a name="p1643935185813"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~IndexILFlat`<a name="en-us_TOPIC_0000001456375172"></a>

<a name="table11904175418"></a>
<table><tbody><tr id="row49051251216"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11905115615"><a name="p11905115615"></a><a name="p11905115615"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15905125316"><a name="p15905125316"></a><a name="p15905125316"></a><code>virtual ~IndexILFlat();</code></p>
</td>
</tr>
<tr id="row139053510117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p179056510119"><a name="p179056510119"></a><a name="p179056510119"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p59051353114"><a name="p59051353114"></a><a name="p59051353114"></a>Destructor of <code>IndexILFlat</code>.</p>
</td>
</tr>
<tr id="row17905135915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18905456118"><a name="p18905456118"></a><a name="p18905456118"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6905656112"><a name="p6905656112"></a><a name="p6905656112"></a>None</p>
</td>
</tr>
<tr id="row199051557118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p129051050117"><a name="p129051050117"></a><a name="p129051050117"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p109051250114"><a name="p109051250114"></a><a name="p109051250114"></a>None</p>
</td>
</tr>
<tr id="row149051757115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p169052055120"><a name="p169052055120"></a><a name="p169052055120"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1690513511119"><a name="p1690513511119"></a><a name="p1690513511119"></a>None</p>
</td>
</tr>
<tr id="row29058514119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15905151318"><a name="p15905151318"></a><a name="p15905151318"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p139051951417"><a name="p139051951417"></a><a name="p139051951417"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Init`<a name="en-us_TOPIC_0000001456375212"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize = -1) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>Initializes feature library parameters and allocates base library memory resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1517311219268"><a name="b1517311219268"></a><a name="b1517311219268"></a><code>int dim</code></strong>: Feature vector dimension.</p>
<p id="p1889154465814"><a name="p1889154465814"></a><a name="p1889154465814"></a><strong id="b637319417265"><a name="b637319417265"></a><a name="b637319417265"></a><code>AscendMetricType metricType</code></strong>: Feature distance type, including inner product, Euclidean distance, and cosine similarity.</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8628752620"><a name="b8628752620"></a><a name="b8628752620"></a><code>int capacity</code></strong>: Maximum base library capacity. The API allocates <code>capacity * dim * sizeof(fp16)</code> bytes of memory based on the <code>capacity</code> value.</p>
<p id="p1411722401512"><a name="p1411722401512"></a><a name="p1411722401512"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a><code>int64_t resourceSize</code></strong>: Preallocates Device-side cache resources. When a retrieval API is called, it can use these resources directly instead of calling the <code>aclrtmalloc</code> interface to allocate memory, which improves performance.</p>
<p id="p117241413167"><a name="p117241413167"></a><a name="p117241413167"></a>The default value is <code>-1</code>, which means that the cache resource is allocated with the default size of <code>128 MB</code>. You can configure the actual size more precisely based on the retrieval workload and Device-side resource usage.</p>
<p id="p1703214386"><a name="p1703214386"></a><a name="p1703214386"></a>For example, if the query batch size is <code>64</code>, the base library contains 1,000,000 vectors, and one FP32 value occupies 4 bytes, set <code>resourceSize</code> to <code>64 * 1000000 * 4 = 256,000,000</code> bytes. Note that the maximum cache resource supported by the interface is <code>4 GB</code>.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1768605017262"></a><a name="ul1768605017262"></a><ul id="ul1768605017262"><li><code>dim</code> ∈ {32, 64, 128, 256, 384, 512, 1024}.</li><li><code>metricType</code>: <code>IndexILFlat</code> currently implements only the inner product distance, so it supports only <code>AscendMetricType::ASCEND_METRIC_INNER_PRODUCT</code>.</li><li><code>capacity</code>: The maximum memory that the API can allocate for the base library is <code>12,288,000,000</code> bytes, and the allowed range of <code>capacity</code> is <code>(0, 12000000]</code>.<a name="ul138816512117"></a><a name="ul138816512117"></a><ul id="ul138816512117"><li>For example, for a base library vector set with 512 dimensions and the FP16 type, the maximum supported <code>capacity</code> is 12 million (<code>12288000000 / (512 * sizeof(fp16))</code>).</li><li>For a base library vector set with 256 dimensions and the FP16 type, <code>capacity</code> can still be set to at most 12 million, even though the memory limit supports a larger value.</li></ul>
</li><li><code>resourceSize</code> can be set to <code>-1</code> or any value in <code>[134217728, 4294967296]</code>, in bytes, which is equivalent to <code>[128 MB, 4096 MB]</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `operator =`<a name="en-us_TOPIC_0000001897140809"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4759192852812"><a name="p4759192852812"></a><a name="p4759192852812"></a><code>IndexILFlat&amp; operator=(const IndexILFlat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5787143522812"><a name="p5787143522812"></a><a name="p5787143522812"></a>Declares this <code>Index</code> assignment operator as deleted. Therefore, the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p175601140172811"><a name="p175601140172811"></a><a name="p175601140172811"></a><strong id="b9347444182810"><a name="b9347444182810"></a><a name="b9347444182810"></a><code>const IndexILFlat&amp;</code></strong>: <code>IndexILFlat</code> object.</p>
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

## `RemoveFeatures`<a name="en-us_TOPIC_0000001506414837"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR RemoveFeatures(int n, const idx_t *indices) override;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Deletes <code>n</code> feature vectors with specified indices from the vector library.</p>
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

## `Search`<a name="en-us_TOPIC_0000001456854856"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1346615102548"></a><a name="ul1346615102548"></a><ul id="ul1346615102548"><li><strong id="b5538111715545"><a name="b5538111715545"></a><a name="b5538111715545"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><strong id="b18681518105411"><a name="b18681518105411"></a><a name="b18681518105411"></a><code>topk</code></strong>: The value must be in <code>[0, 1024]</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><code>indices</code>, <code>queries</code>, and <code>distances</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchByThreshold`<a name="en-us_TOPIC_0000001456694892"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1591144334913"><a name="p1591144334913"></a><a name="p1591144334913"></a><code>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</code></p>
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
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b24212483396"><a name="b24212483396"></a><a name="b24212483396"></a><code>const float16_t *queries</code></strong>: Feature vectors to query. The length is <code>n * dim</code>.</p>
<p id="p12923104514555"><a name="p12923104514555"></a><a name="p12923104514555"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a><code>float threshold</code></strong>: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by <code>threshold</code>.</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a><code>int topk</code></strong>: Sorts the comparison distances between the query vectors and the base library and returns <code>topk</code> results.</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a><code>unsigned int tableLen</code></strong>: Mapping table length. The default value is <code>0</code>, which means that mapping is not performed. The currently supported mapping table length is <code>10000</code>.</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a><code>const float *table</code></strong>: Mapping table pointer. It points to valid mapping values stored in a space of length <code>tableLen</code>. The currently supported redundant length is <code>48</code>, which means that the space pointed to by <code>table</code> has a length of <code>10048 * sizeof(float)</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b662915439408"><a name="b662915439408"></a><a name="b662915439408"></a><code>int *num</code></strong>: Number of base library vectors that meet the threshold condition for each query. The length is <code>n</code>.</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b452884674019"><a name="b452884674019"></a><a name="b452884674019"></a><code>idx_t *indices</code></strong>: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to <code>topk</code>. The total length of <code>indices</code> is <code>n * topk</code>.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a><code>n</code></strong>: The value must be in <code>[0, capacity]</code>.</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a><code>topk</code></strong>: The value must be in <code>[0, 1024]</code>.</li><li>If you pass <code>tableLen</code> and <code>table</code> and both satisfy the requirements, the API maps the computed <code>distance</code> values:<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>First, it normalizes <code>distance</code> to the floating-point value <code>f1</code> in <code>[0, 1]</code>. Then it multiplies <code>f1</code> by <code>tableLen</code> and rounds down to obtain an integer index in <code>[0, tableLen]</code>. Next, it uses that integer index as an offset to read the corresponding <code>score</code> from the memory space pointed to by <code>table</code>, which completes the mapping and stores <code>score</code> in <code>distance</code>.</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>The index mapping formula can be expressed as <code>((CosDistance + 1) / 2) * tableLen</code>.</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><code>indices</code>, <code>queries</code>, <code>distances</code>, and <code>num</code> must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SetNTotal`<a name="en-us_TOPIC_0000001456854892"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>APP_ERROR SetNTotal(int n) override;</code></p>
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
