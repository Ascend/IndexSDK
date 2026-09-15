# `AscendIndexMixSearchParams`<a name="en-us_TOPIC_0000002008910258"></a>

## Function Description<a name="en-us_TOPIC_0000002045034929"></a>

The `AscendIndexMixSearchParams.h` file provides the structures required by `AscendIndexGreat` and `AscendIndexVStar`.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must acquire a lock before use; otherwise, the search API may cause exceptions. Sharing a single device across different threads is also not supported.

## `AscendIndexGreatInitParams`<a name="en-us_TOPIC_0000002049404289"></a>

<a name="table17465519101616"></a>
<table><tbody><tr id="row13506161913166"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1550613193168"><a name="p1550613193168"></a><a name="p1550613193168"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p205061191163"><a name="p205061191163"></a><a name="p205061191163"></a><code>AscendIndexGreatInitParams();</code></p>
</td>
</tr>
<tr id="row1150611931616"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p8506201910163"><a name="p8506201910163"></a><a name="p8506201910163"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7506131941617"><a name="p7506131941617"></a><a name="p7506131941617"></a>Initialization parameter structure for KMode mode.</p>
</td>
</tr>
<tr id="row2050661921618"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p850612195161"><a name="p850612195161"></a><a name="p850612195161"></a><strong id="b85061319151619"><a name="b85061319151619"></a><a name="b85061319151619"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p9724249405"><a name="p9724249405"></a><a name="p9724249405"></a>None</p>
</td>
</tr>
<tr id="row11506619161619"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1850612196160"><a name="p1850612196160"></a><a name="p1850612196160"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p4506819101620"><a name="p4506819101620"></a><a name="p4506819101620"></a>None</p>
</td>
</tr>
<tr id="row850611991611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p85061319191610"><a name="p85061319191610"></a><a name="p85061319191610"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p35071919161614"><a name="p35071919161614"></a><a name="p35071919161614"></a>See <a href="#table10419189143817"><code>AscendIndexGreatInitParams</code></a> for default parameter values.</p>
</td>
</tr>
</tbody>
</table>

<a id="table10419189143817"></a>
<table><tbody><tr id="row54190910388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3419159133820"><a name="p3419159133820"></a><a name="p3419159133820"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p94191199389"><a name="p94191199389"></a><a name="p94191199389"></a><code>AscendIndexGreatInitParams(int dim, int degree, int convPQM, int evaluationType, int expandingFactor);</code></p>
</td>
</tr>
<tr id="row194192911388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1041949133817"><a name="p1041949133817"></a><a name="p1041949133817"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p24195913810"><a name="p24195913810"></a><a name="p24195913810"></a>Initialization parameter structure for KMode mode.</p>
</td>
</tr>
<tr id="row154191911388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p144191393386"><a name="p144191393386"></a><a name="p144191393386"></a><strong id="b144198910384"><a name="b144198910384"></a><a name="b144198910384"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul164151475215"></a><a name="ul164151475215"></a><ul id="ul164151475215"><li><strong id="b1741512792119"><a name="b1741512792119"></a><a name="b1741512792119"></a><code>int dim</code></strong>: Dimension of the feature vectors.</li><li><strong id="b84151270211"><a name="b84151270211"></a><a name="b84151270211"></a><code>int degree</code></strong>: Controls the fineness of the graph index during index construction. A larger value makes the graph index more fine-grained, requires more space, and yields higher retrieval accuracy.</li><li><strong id="b1415678214"><a name="b1415678214"></a><a name="b1415678214"></a><code>int convPQM</code></strong>: Number of PQ quantization vector segments.</li><li><strong id="b1241557112119"><a name="b1241557112119"></a><a name="b1241557112119"></a><code>int evaluationType</code></strong>: Distance evaluation algorithm type; <code>0</code> represents IP, and <code>1</code> represents L2.</li><li><strong id="b174151272218"><a name="b174151272218"></a><a name="b174151272218"></a><code>int expandingFactor</code></strong>: Number of neighbors connected when searching each layer during the initial graph-construction phase. Note that this is different from the retrieval-stage <code>expandingFactor</code>.</li></ul>
</td>
</tr>
<tr id="row141916973817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p184191897387"><a name="p184191897387"></a><a name="p184191897387"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p104199923810"><a name="p104199923810"></a><a name="p104199923810"></a>None</p>
</td>
</tr>
<tr id="row04193933812"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p114191898388"><a name="p114191898388"></a><a name="p114191898388"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><a name="ul10297151415264"></a><a name="ul10297151415264"></a><ul id="ul10297151415264"><li><code>dim</code> ∈ {128, 256, 512, 1024}. Default value: <code>256</code>.</li><li><code>degree</code> ∈ [50, 100]. Default value: <code>50</code>.</li><li><code>convPQM</code>: Must be at least <code>16</code>, must be a multiple of <code>8</code>, and must be divisible by <code>dim</code>. Default value: <code>128</code>.</li><li><code>evaluationType</code> ∈ {0, 1}. Default value: <code>0</code>.</li><li><code>expandingFactor</code> ∈ [200, 400]. Must be a multiple of <code>10</code>. Default value: <code>300</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndexVstarInitParams`<a name="en-us_TOPIC_0000002013246410"></a>

<a name="table20955195613391"></a>
<table><tbody><tr id="row179551256163915"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p49558566396"><a name="p49558566396"></a><a name="p49558566396"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p595585653912"><a name="p595585653912"></a><a name="p595585653912"></a><code>AscendIndexVstarInitParams();</code></p>
</td>
</tr>
<tr id="row199551956193911"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p1495545693913"><a name="p1495545693913"></a><a name="p1495545693913"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p1955145673916"><a name="p1955145673916"></a><a name="p1955145673916"></a>Initialization parameter structure for Vstar mode.</p>
</td>
</tr>
<tr id="row11955155643916"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p1395515616391"><a name="p1395515616391"></a><a name="p1395515616391"></a><strong id="b14955125693917"><a name="b14955125693917"></a><a name="b14955125693917"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p10955195616392"><a name="p10955195616392"></a><a name="p10955195616392"></a>None</p>
</td>
</tr>
<tr id="row15955156163911"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p7955956173915"><a name="p7955956173915"></a><a name="p7955956173915"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p18955135653919"><a name="p18955135653919"></a><a name="p18955135653919"></a>None</p>
</td>
</tr>
<tr id="row39558561396"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p2955656113911"><a name="p2955656113911"></a><a name="p2955656113911"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p1955195673914"><a name="p1955195673914"></a><a name="p1955195673914"></a>See <a href="#table42921559204019"><code>AscendIndexVstarHyperParams</code></a> for default parameter values.</p>
</td>
</tr>
</tbody>
</table>

<a id="table899624214019"></a>
<table><tbody><tr id="row129968429408"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p1499619428400"><a name="p1499619428400"></a><a name="p1499619428400"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p2099614274013"><a name="p2099614274013"></a><a name="p2099614274013"></a><code>AscendIndexVstarInitParams(int dim, int subSpaceDim, int nlist, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false, int64_t resourceSize = VSTAR_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row8996174214017"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p14996442104012"><a name="p14996442104012"></a><a name="p14996442104012"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p1999614423408"><a name="p1999614423408"></a><a name="p1999614423408"></a>Initialization parameter structure for Vstar mode.</p>
</td>
</tr>
<tr id="row1399694216401"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p999613425403"><a name="p999613425403"></a><a name="p999613425403"></a><strong id="b1999654274016"><a name="b1999654274016"></a><a name="b1999654274016"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p15980193215426"><a name="p15980193215426"></a><a name="p15980193215426"></a><strong id="b159801432184219"><a name="b159801432184219"></a><a name="b159801432184219"></a><code>int dim</code></strong>: Dimension of the feature vectors.</p>
<p id="p798019329424"><a name="p798019329424"></a><a name="p798019329424"></a><strong id="b17980332134218"><a name="b17980332134218"></a><a name="b17980332134218"></a><code>int subSpaceDim</code></strong>: Dimension after the first dimensionality reduction.</p>
<p id="p129801432144216"><a name="p129801432144216"></a><a name="p129801432144216"></a><strong id="b6980232164216"><a name="b6980232164216"></a><a name="b6980232164216"></a><code>int nlist</code></strong>: Number of first-level clusters.</p>
<p id="p1798043284218"><a name="p1798043284218"></a><a name="p1798043284218"></a><strong id="b179801332124218"><a name="b179801332124218"></a><a name="b179801332124218"></a><code>const std::vector&lt;int&gt;&amp; deviceList</code></strong>: Specified NPU physical IDs.</p>
<p id="p109801732134212"><a name="p109801732134212"></a><a name="p109801732134212"></a><strong id="b11713115919613"><a name="b11713115919613"></a><a name="b11713115919613"></a><code>bool verbose</code></strong>: Whether to enable the <code>verbose</code> option. When enabled, some operations provide additional print prompts. Default value: <code>false</code>.</p>
<p id="p1881318366219"><a name="p1881318366219"></a><a name="p1881318366219"></a><strong id="b1881318366219"><a name="b1881318366219"></a><a name="b1881318366219"></a><code>int64_t resourceSize</code></strong>: Size of the preallocated memory pool on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>VSTAR_DEFAULT_MEM</code> defined in the header file, with a size of 128 MB. This parameter is determined jointly by the base library size and the <code>search</code> batch size.</p>
</td>
</tr>
<tr id="row14997184218409"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p399794212404"><a name="p399794212404"></a><a name="p399794212404"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p119971642124010"><a name="p119971642124010"></a><a name="p119971642124010"></a>None</p>
</td>
</tr>
<tr id="row899774215406"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p199972042174013"><a name="p199972042174013"></a><a name="p199972042174013"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p999774294015"><a name="p999774294015"></a><a name="p999774294015"></a><code>dim</code> ∈ {128, 256, 512, 1024}. Default value: <code>1024</code>.</p>
<p id="p14393113811167"><a name="p14393113811167"></a><a name="p14393113811167"></a><code>subSpaceDim</code> ∈ {32, 64, 128}. <code>subSpaceDim</code> must be less than <code>dim</code>. Default value: <code>128</code>.</p>
<p id="p339314384161"><a name="p339314384161"></a><a name="p339314384161"></a><code>nlist</code> ∈ {256, 512, 1024}. Default value: <code>1024</code>.</p>
<p id="p174351643113118"><a name="p174351643113118"></a><a name="p174351643113118"></a>For <code>deviceList</code>, use the <strong id="b1949519225201"><a name="b1949519225201"></a><a name="b1949519225201"></a><code>npu-smi</code></strong> command to query the physical ID of the corresponding NPU card. Only one device ID is supported.</p>
<p id="p11413112513610"><a name="p11413112513610"></a><a name="p11413112513610"></a><code>resourceSize</code> ∈ [128M, 2048M].</p>
</td>
</tr>
</tbody>
</table>

## `AscendIndexVstarHyperParams`<a name="en-us_TOPIC_0000002013404694"></a>

<a name="table201855541164"></a>
<table><tbody><tr id="row1229205491611"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p15229145421610"><a name="p15229145421610"></a><a name="p15229145421610"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p19775933112813"><a name="p19775933112813"></a><a name="p19775933112813"></a><code>AscendIndexVstarHyperParams();</code></p>
</td>
</tr>
<tr id="row922985415161"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p172301854171617"><a name="p172301854171617"></a><a name="p172301854171617"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p19230195451619"><a name="p19230195451619"></a><a name="p19230195451619"></a>Hyperparameter structure for VSTAR mode.</p>
</td>
</tr>
<tr id="row5230155410161"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p202301754181615"><a name="p202301754181615"></a><a name="p202301754181615"></a><strong id="b1230125481610"><a name="b1230125481610"></a><a name="b1230125481610"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p14230654111611"><a name="p14230654111611"></a><a name="p14230654111611"></a>None</p>
</td>
</tr>
<tr id="row152301754191616"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p72301154101620"><a name="p72301154101620"></a><a name="p72301154101620"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p1323045491617"><a name="p1323045491617"></a><a name="p1323045491617"></a>None</p>
</td>
</tr>
<tr id="row52301454181614"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p1323065491615"><a name="p1323065491615"></a><a name="p1323065491615"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p572594452912"><a name="p572594452912"></a><a name="p572594452912"></a>See <a href="#table42921559204019"><code>AscendIndexVstarHyperParams</code></a> for default parameter values.</p>
</td>
</tr>
</tbody>
</table>

<a id="table42921559204019"></a>
<table><tbody><tr id="row1929245944010"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12921659194012"><a name="p12921659194012"></a><a name="p12921659194012"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p15292115964010"><a name="p15292115964010"></a><a name="p15292115964010"></a><code>AscendIndexVstarHyperParams(int nProbeL1, int nProbeL2, int l3SegmentNum);</code></p>
</td>
</tr>
<tr id="row62921559174010"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p82924592406"><a name="p82924592406"></a><a name="p82924592406"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p129205994013"><a name="p129205994013"></a><a name="p129205994013"></a>Hyperparameter structure for VSTAR mode.</p>
</td>
</tr>
<tr id="row929275984019"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p129316593406"><a name="p129316593406"></a><a name="p129316593406"></a><strong id="b11293359184015"><a name="b11293359184015"></a><a name="b11293359184015"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p1029355919406"><a name="p1029355919406"></a><a name="p1029355919406"></a><strong id="b13114027192810"><a name="b13114027192810"></a><a name="b13114027192810"></a><code>int nProbeL1</code></strong>: Number of clusters searched in the first-stage retrieval.</p>
<p id="p107015014422"><a name="p107015014422"></a><a name="p107015014422"></a><strong id="b18772202952812"><a name="b18772202952812"></a><a name="b18772202952812"></a><code>int nProbeL2</code></strong>: Number of clusters searched in the second-stage retrieval.</p>
<p id="p9701450174216"><a name="p9701450174216"></a><a name="p9701450174216"></a><strong id="b182823292820"><a name="b182823292820"></a><a name="b182823292820"></a><code>int l3SegmentNum</code></strong>: Number of segments in the third-stage retrieval, that is, the number of data segments searched from <code>nProbeL2</code>.</p>
</td>
</tr>
<tr id="row18293175964011"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p172930593402"><a name="p172930593402"></a><a name="p172930593402"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p19293459104014"><a name="p19293459104014"></a><a name="p19293459104014"></a>None</p>
</td>
</tr>
<tr id="row162937595403"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p1229345910409"><a name="p1229345910409"></a><a name="p1229345910409"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><a name="ul1287219505284"></a><a name="ul1287219505284"></a><ul id="ul1287219505284"><li><code>nProbeL1</code> ∈ [32, <code>nListL1</code>], and <code>nProbeL1</code> must be an integer multiple of <code>8</code>. Default value: <code>72</code>.</li><li><code>nProbeL2</code> ∈ (16, <code>nProbeL1 * n</code>]; when <code>dim</code> is <code>1024</code>, <code>n</code> is <code>16</code>, and for other dimensions <code>n</code> is <code>32</code>. <code>nProbeL2</code> must be an integer multiple of <code>8</code>. Default value: <code>64</code>.</li><li><code>l3SegmentNum</code> ∈ (100, 5000], and <code>l3SegmentNum</code> must be an integer multiple of <code>8</code>. Default value: <code>512</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndexHyperParams`<a name="en-us_TOPIC_0000002049325253"></a>

<a name="table93967711712"></a>
<table><tbody><tr id="row1042207151710"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.1.1"><p id="p74221719175"><a name="p74221719175"></a><a name="p74221719175"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.1.1 "><p id="p64221670173"><a name="p64221670173"></a><a name="p64221670173"></a><code>AscendIndexHyperParams();</code></p>
</td>
</tr>
<tr id="row44222771712"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.2.1"><p id="p2422173179"><a name="p2422173179"></a><a name="p2422173179"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.2.1 "><p id="p154225713171"><a name="p154225713171"></a><a name="p154225713171"></a>Hyperparameter structure for GREAT retrieval.</p>
</td>
</tr>
<tr id="row14231577178"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.3.1"><p id="p242314711720"><a name="p242314711720"></a><a name="p242314711720"></a><strong id="b3423177101713"><a name="b3423177101713"></a><a name="b3423177101713"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.3.1 "><p id="p14843135984413"><a name="p14843135984413"></a><a name="p14843135984413"></a>None</p>
</td>
</tr>
<tr id="row8423127161719"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.4.1"><p id="p542312710172"><a name="p542312710172"></a><a name="p542312710172"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.4.1 "><p id="p12423373171"><a name="p12423373171"></a><a name="p12423373171"></a>None</p>
</td>
</tr>
<tr id="row194231972176"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.5.1"><p id="p642315711713"><a name="p642315711713"></a><a name="p642315711713"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.5.1 "><p id="p1739165164518"><a name="p1739165164518"></a><a name="p1739165164518"></a>See <a href="#table1334182412417"><code>AscendIndexHyperParams</code></a> for default parameter values.</p>
</td>
</tr>
</tbody>
</table>

<a id="table1334182412417"></a>
<table><tbody><tr id="row7341224164110"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.1.1"><p id="p17341524124117"><a name="p17341524124117"></a><a name="p17341524124117"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.1.1 "><p id="p9341192424110"><a name="p9341192424110"></a><a name="p9341192424110"></a><code>AscendIndexHyperParams(const std::string&amp; mode, const AscendIndexVstarHyperParams&amp; vstarHyperParam, int expandingFactor);</code></p>
</td>
</tr>
<tr id="row12341102415417"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.2.1"><p id="p8341152417417"><a name="p8341152417417"></a><a name="p8341152417417"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.2.1 "><p id="p16341112444120"><a name="p16341112444120"></a><a name="p16341112444120"></a>Hyperparameter structure for GREAT retrieval.</p>
</td>
</tr>
<tr id="row183411924124115"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.3.1"><p id="p1634120247416"><a name="p1634120247416"></a><a name="p1634120247416"></a><strong id="b1534152414118"><a name="b1534152414118"></a><a name="b1534152414118"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.3.1 "><p id="p74235712170"><a name="p74235712170"></a><a name="p74235712170"></a><strong id="b8576358446"><a name="b8576358446"></a><a name="b8576358446"></a><code>const std::string&amp; mode</code></strong>: Specifies the algorithm mode.</p>
<p id="p757103514413"><a name="p757103514413"></a><a name="p757103514413"></a><strong id="b11138125032113"><a name="b11138125032113"></a><a name="b11138125032113"></a><code>const AscendIndexVstarHyperParams&amp; vstarHyperParam</code></strong>: For details, see <a href="#table42921559204019"><code>AscendIndexVstarHyperParams</code></a>.</p>
<p id="p1557203524420"><a name="p1557203524420"></a><a name="p1557203524420"></a><strong id="b105743519442"><a name="b105743519442"></a><a name="b105743519442"></a><code>int expandingFactor</code></strong>: Number of neighbors searched at each layer during retrieval. Note that this differs from the <code>expandingFactor</code> used during graph construction.</p>
</td>
</tr>
<tr id="row5341172424119"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.4.1"><p id="p15341172424120"><a name="p15341172424120"></a><a name="p15341172424120"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.4.1 "><p id="p734112464111"><a name="p734112464111"></a><a name="p734112464111"></a>None</p>
</td>
</tr>
<tr id="row14341224164113"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.5.1"><p id="p134119246417"><a name="p134119246417"></a><a name="p134119246417"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.5.1 "><a name="ul1944290182915"></a><a name="ul1944290182915"></a><ul id="ul1944290182915"><li><code>mode</code> ∈ {"KMode", "AKMode"}. Default value: <code>AKMode</code>.</li><li><code>expandingFactor</code> ∈ [10, 200]. Default value: <code>150</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table88027219236"></a>
<table><tbody><tr id="row280232117234"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.1.1"><p id="p10802112111235"><a name="p10802112111235"></a><a name="p10802112111235"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.1.1 "><p id="p4802132116235"><a name="p4802132116235"></a><a name="p4802132116235"></a><code>AscendIndexHyperParams(const std::string&amp; mode, int expandingFactor);</code></p>
</td>
</tr>
<tr id="row080292182312"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.2.1"><p id="p680220214236"><a name="p680220214236"></a><a name="p680220214236"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.2.1 "><p id="p20802142112316"><a name="p20802142112316"></a><a name="p20802142112316"></a>Hyperparameter structure for GREAT retrieval.</p>
</td>
</tr>
<tr id="row198021821192318"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.3.1"><p id="p1780216210239"><a name="p1780216210239"></a><a name="p1780216210239"></a><strong id="b1480272114235"><a name="b1480272114235"></a><a name="b1480272114235"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.3.1 "><p id="p15802112162316"><a name="p15802112162316"></a><a name="p15802112162316"></a><strong id="b8802721132317"><a name="b8802721132317"></a><a name="b8802721132317"></a><code>const std::string&amp; mode</code></strong>: Specifies the algorithm mode.</p>
<p id="p0802162182317"><a name="p0802162182317"></a><a name="p0802162182317"></a><strong id="b1980242122318"><a name="b1980242122318"></a><a name="b1980242122318"></a><code>int expandingFactor</code></strong>: Number of neighbors searched at each layer during retrieval. Note that this differs from the <code>expandingFactor</code> used during graph construction.</p>
</td>
</tr>
<tr id="row980282122314"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.4.1"><p id="p28021621192314"><a name="p28021621192314"></a><a name="p28021621192314"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.4.1 "><p id="p10802521122312"><a name="p10802521122312"></a><a name="p10802521122312"></a>None</p>
</td>
</tr>
<tr id="row1580282118238"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.5.1"><p id="p2802192192310"><a name="p2802192192310"></a><a name="p2802192192310"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.5.1 "><a name="ul1480216213235"></a><a name="ul1480216213235"></a><ul id="ul1480216213235"><li><code>mode</code> ∈ {"KMode", "AKMode"}. Default value: <code>AKMode</code>.</li><li><code>expandingFactor</code> ∈ [10, 200]. Default value: <code>150</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndexSearchParams`<a name="en-us_TOPIC_0000002044950949"></a>

<a name="table414612258177"></a>
<table><tbody><tr id="row118413250172"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2018462520172"><a name="p2018462520172"></a><a name="p2018462520172"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16184172511716"><a name="p16184172511716"></a><a name="p16184172511716"></a><code>AscendIndexSearchParams(size_t n, std::vector&lt;float&gt;&amp; queryData, int topK, std::vector&lt;float&gt;&amp; dists, std::vector&lt;int64_t&gt;&amp; labels);</code></p>
</td>
</tr>
<tr id="row16184162515173"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p61841251179"><a name="p61841251179"></a><a name="p61841251179"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p14184112541715"><a name="p14184112541715"></a><a name="p14184112541715"></a>Search parameter structure for retrieval.</p>
</td>
</tr>
<tr id="row191848251175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p4184125201718"><a name="p4184125201718"></a><a name="p4184125201718"></a><strong id="b7184122591715"><a name="b7184122591715"></a><a name="b7184122591715"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p126612528811"><a name="p126612528811"></a><a name="p126612528811"></a>None</p>
</td>
</tr>
<tr id="row1518572551717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p11851825121717"><a name="p11851825121717"></a><a name="p11851825121717"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p789264511813"><a name="p789264511813"></a><a name="p789264511813"></a>None</p>
</td>
</tr>
<tr id="row1185425101713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12185122518179"><a name="p12185122518179"></a><a name="p12185122518179"></a>Parameter Values</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p5761142186"><a name="p5761142186"></a><a name="p5761142186"></a><strong id="b4761114219820"><a name="b4761114219820"></a><a name="b4761114219820"></a><code>size_t n</code></strong>: Number of query feature vectors.</p>
<p id="p1941741042119"><a name="p1941741042119"></a><a name="p1941741042119"></a><strong id="b089323152313"><a name="b089323152313"></a><a name="b089323152313"></a><code>std::vector&lt;float&gt;&amp; queryData</code></strong>: Feature vector data.</p>
<p id="p19761124216818"><a name="p19761124216818"></a><a name="p19761124216818"></a><strong id="b117611142286"><a name="b117611142286"></a><a name="b117611142286"></a><code>int topK</code></strong>: Number of most similar results to return.</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b1671817282232"><a name="b1671817282232"></a><a name="b1671817282232"></a><code>std::vector&lt;float&gt;&amp; dists</code></strong>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.</p>
<p id="p5548356122111"><a name="p5548356122111"></a><a name="p5548356122111"></a><strong id="b287214390241"><a name="b287214390241"></a><a name="b287214390241"></a><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</p>
</td>
</tr>
<tr id="row4185425171719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1218512541712"><a name="p1218512541712"></a><a name="p1218512541712"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5481551192220"></a><a name="ul5481551192220"></a><ul id="ul5481551192220"><li><code>topK</code> ∈ (0, 4096].</li><li><code>n</code> ∈ (0, 10000].</li><li><code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>.</li><li><code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</li><li><code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</li></ul>
</td>
</tr>
</tbody>
</table>
