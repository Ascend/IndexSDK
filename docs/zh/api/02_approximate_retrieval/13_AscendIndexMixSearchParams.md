# AscendIndexMixSearchParams<a name="ZH-CN_TOPIC_0000002008910258"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002045034929"></a>

AscendIndexMixSearchParams.h文件，提供AscendIndexGreat和AscendIndexVStar需要的结构体。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

## AscendIndexGreatInitParams接口<a name="ZH-CN_TOPIC_0000002049404289"></a>

<a name="table17465519101616"></a>
<table><tbody><tr id="row13506161913166"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1550613193168"><a name="p1550613193168"></a><a name="p1550613193168"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p205061191163"><a name="p205061191163"></a><a name="p205061191163"></a>AscendIndexGreatInitParams();</p>
</td>
</tr>
<tr id="row1150611931616"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p8506201910163"><a name="p8506201910163"></a><a name="p8506201910163"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7506131941617"><a name="p7506131941617"></a><a name="p7506131941617"></a>KMode模式初始化参数结构体。</p>
</td>
</tr>
<tr id="row2050661921618"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p850612195161"><a name="p850612195161"></a><a name="p850612195161"></a><strong id="b85061319151619"><a name="b85061319151619"></a><a name="b85061319151619"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p9724249405"><a name="p9724249405"></a><a name="p9724249405"></a>无</p>
</td>
</tr>
<tr id="row11506619161619"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1850612196160"><a name="p1850612196160"></a><a name="p1850612196160"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p4506819101620"><a name="p4506819101620"></a><a name="p4506819101620"></a>无</p>
</td>
</tr>
<tr id="row850611991611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p85061319191610"><a name="p85061319191610"></a><a name="p85061319191610"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p35071919161614"><a name="p35071919161614"></a><a name="p35071919161614"></a>参数默认值见<a href="#table10419189143817">AscendIndexGreatInitParams</a>。</p>
</td>
</tr>
</tbody>
</table>

<a id="table10419189143817"></a>
<table><tbody><tr id="row54190910388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3419159133820"><a name="p3419159133820"></a><a name="p3419159133820"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p94191199389"><a name="p94191199389"></a><a name="p94191199389"></a>AscendIndexGreatInitParams(int dim, int degree, int convPQM, int evaluationType, int expandingFactor);</p>
</td>
</tr>
<tr id="row194192911388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1041949133817"><a name="p1041949133817"></a><a name="p1041949133817"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p24195913810"><a name="p24195913810"></a><a name="p24195913810"></a>KMode模式初始化参数结构体。</p>
</td>
</tr>
<tr id="row154191911388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p144191393386"><a name="p144191393386"></a><a name="p144191393386"></a><strong id="b144198910384"><a name="b144198910384"></a><a name="b144198910384"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul164151475215"></a><a name="ul164151475215"></a><ul id="ul164151475215"><li><strong id="b1741512792119"><a name="b1741512792119"></a><a name="b1741512792119"></a>int dim</strong>：特征向量的维度。</li><li><strong id="b84151270211"><a name="b84151270211"></a><a name="b84151270211"></a>int degree</strong>：在索引构建阶段控制图索引的精细程度，值越大图索引越精细，占用空间越大，检索时更准确。</li><li><strong id="b1415678214"><a name="b1415678214"></a><a name="b1415678214"></a>int convPQM</strong>：PQ量化向量分段数。</li><li><strong id="b1241557112119"><a name="b1241557112119"></a><a name="b1241557112119"></a>int evaluationType</strong>：距离评估算法类型，0代表IP，1代表L2。</li><li><strong id="b174151272218"><a name="b174151272218"></a><a name="b174151272218"></a>int expandingFactor</strong>：初始构图阶段，连接每一层搜索时邻居的数量。注意与检索阶段的<strong id="b165311219154112"><a name="b165311219154112"></a><a name="b165311219154112"></a>expandingFactor</strong>区分。</li></ul>
</td>
</tr>
<tr id="row141916973817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p184191897387"><a name="p184191897387"></a><a name="p184191897387"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p104199923810"><a name="p104199923810"></a><a name="p104199923810"></a>无</p>
</td>
</tr>
<tr id="row04193933812"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p114191898388"><a name="p114191898388"></a><a name="p114191898388"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><a name="ul10297151415264"></a><a name="ul10297151415264"></a><ul id="ul10297151415264"><li>dim ∈ {128, 256, 512, 1024}，默认值为<span class="parmvalue" id="parmvalue976982810518"><a name="parmvalue976982810518"></a><a name="parmvalue976982810518"></a>“256”</span>。</li><li>degree ∈ [50, 100]，默认值为<span class="parmvalue" id="parmvalue871233114510"><a name="parmvalue871233114510"></a><a name="parmvalue871233114510"></a>“50”</span>。</li><li>convPQM：大于等于16，并且convPQM是8的倍数且能被dim整除，默认值为<span class="parmvalue" id="parmvalue45298342517"><a name="parmvalue45298342517"></a><a name="parmvalue45298342517"></a>“128”</span>。</li><li>evaluationType ∈ {0，1}，默认值为<span class="parmvalue" id="parmvalue22121539158"><a name="parmvalue22121539158"></a><a name="parmvalue22121539158"></a>“0”</span>。</li><li>expandingFactor∈ [200, 400]，expandingFactor必须是10的倍数，默认值为<span class="parmvalue" id="parmvalue1327374515"><a name="parmvalue1327374515"></a><a name="parmvalue1327374515"></a>“300”</span>。</li></ul>
</td>
</tr>
</tbody>
</table>

## AscendIndexVstarInitParams接口<a name="ZH-CN_TOPIC_0000002013246410"></a>

<a name="table20955195613391"></a>
<table><tbody><tr id="row179551256163915"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p49558566396"><a name="p49558566396"></a><a name="p49558566396"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p595585653912"><a name="p595585653912"></a><a name="p595585653912"></a>AscendIndexVstarInitParams();</p>
</td>
</tr>
<tr id="row199551956193911"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p1495545693913"><a name="p1495545693913"></a><a name="p1495545693913"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p1955145673916"><a name="p1955145673916"></a><a name="p1955145673916"></a>Vstar模式初始化参数结构体。</p>
</td>
</tr>
<tr id="row11955155643916"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p1395515616391"><a name="p1395515616391"></a><a name="p1395515616391"></a><strong id="b14955125693917"><a name="b14955125693917"></a><a name="b14955125693917"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p10955195616392"><a name="p10955195616392"></a><a name="p10955195616392"></a>无</p>
</td>
</tr>
<tr id="row15955156163911"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p7955956173915"><a name="p7955956173915"></a><a name="p7955956173915"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p18955135653919"><a name="p18955135653919"></a><a name="p18955135653919"></a>无</p>
</td>
</tr>
<tr id="row39558561396"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p2955656113911"><a name="p2955656113911"></a><a name="p2955656113911"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p1955195673914"><a name="p1955195673914"></a><a name="p1955195673914"></a>参数默认值见<a href="#table42921559204019">AscendIndexVstarHyperParams</a>。</p>
</td>
</tr>
</tbody>
</table>

<a id="table899624214019"></a>
<table><tbody><tr id="row129968429408"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p1499619428400"><a name="p1499619428400"></a><a name="p1499619428400"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p2099614274013"><a name="p2099614274013"></a><a name="p2099614274013"></a>AscendIndexVstarInitParams(int dim, int subSpaceDim, int nlist, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false, int64_t resourceSize = VSTAR_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row8996174214017"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p14996442104012"><a name="p14996442104012"></a><a name="p14996442104012"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p1999614423408"><a name="p1999614423408"></a><a name="p1999614423408"></a>Vstar模式初始化参数结构体。</p>
</td>
</tr>
<tr id="row1399694216401"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p999613425403"><a name="p999613425403"></a><a name="p999613425403"></a><strong id="b1999654274016"><a name="b1999654274016"></a><a name="b1999654274016"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p15980193215426"><a name="p15980193215426"></a><a name="p15980193215426"></a><strong id="b159801432184219"><a name="b159801432184219"></a><a name="b159801432184219"></a>int dim</strong>：特征向量的维度。</p>
<p id="p798019329424"><a name="p798019329424"></a><a name="p798019329424"></a><strong id="b17980332134218"><a name="b17980332134218"></a><a name="b17980332134218"></a>int subSpaceDim</strong>：第一次降维后的维度大小。</p>
<p id="p129801432144216"><a name="p129801432144216"></a><a name="p129801432144216"></a><strong id="b6980232164216"><a name="b6980232164216"></a><a name="b6980232164216"></a>int nlist</strong>：一级聚类的数量。</p>
<p id="p1798043284218"><a name="p1798043284218"></a><a name="p1798043284218"></a><strong id="b179801332124218"><a name="b179801332124218"></a><a name="b179801332124218"></a>const std::vector&lt;int&gt;&amp; deviceList</strong>：指定的NPU physical ID。</p>
<p id="p109801732134212"><a name="p109801732134212"></a><a name="p109801732134212"></a><strong id="b11713115919613"><a name="b11713115919613"></a><a name="b11713115919613"></a>bool verbose</strong>：指定是否开启verbose选项，开启后部分操作提供额外的打印提示。默认值为“false”。</p>
<p id="p1881318366219"><a name="p1881318366219"></a><a name="p1881318366219"></a>int64_t resourceSize：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的<span class="parmvalue" id="parmvalue7166553349"><a name="parmvalue7166553349"></a><a name="parmvalue7166553349"></a>“VSTAR_DEFAULT_MEM”</span>，大小为128M。该参数通过底库大小和search的batch数共同确定。</p>
</td>
</tr>
<tr id="row14997154218409"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p399794212404"><a name="p399794212404"></a><a name="p399794212404"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p119971642124010"><a name="p119971642124010"></a><a name="p119971642124010"></a>无</p>
</td>
</tr>
<tr id="row899774215406"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p199972042174013"><a name="p199972042174013"></a><a name="p199972042174013"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p999774294015"><a name="p999774294015"></a><a name="p999774294015"></a>dim ∈ {128, 256, 512, 1024}，默认值为<span class="parmvalue" id="parmvalue1950315256254"><a name="parmvalue1950315256254"></a><a name="parmvalue1950315256254"></a>“1024”</span>。</p>
<p id="p14393113811167"><a name="p14393113811167"></a><a name="p14393113811167"></a>subSpaceDim ∈ {32，64，128}。subSpaceDim必须小于dim。默认值为“128”。</p>
<p id="p339314384161"><a name="p339314384161"></a><a name="p339314384161"></a>nlist∈ {256, 512, 1024}。默认值为“1024”。</p>
<p id="p174351643113118"><a name="p174351643113118"></a><a name="p174351643113118"></a>deviceList：请使用<strong id="b1949519225201"><a name="b1949519225201"></a><a name="b1949519225201"></a>npu-smi</strong>命令查询对应的NPU卡physical ID，仅支持一个device设备ID。</p>
<p id="p11413112513610"><a name="p11413112513610"></a><a name="p11413112513610"></a>resourceSize ∈ [128M, 2048M]。</p>
</td>
</tr>
</tbody>
</table>

## AscendIndexVstarHyperParams接口<a name="ZH-CN_TOPIC_0000002013404694"></a>

<a name="table201855541164"></a>
<table><tbody><tr id="row1229205491611"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p15229145421610"><a name="p15229145421610"></a><a name="p15229145421610"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p19775933112813"><a name="p19775933112813"></a><a name="p19775933112813"></a>AscendIndexVstarHyperParams();</p>
</td>
</tr>
<tr id="row922985415161"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p172301854171617"><a name="p172301854171617"></a><a name="p172301854171617"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p19230195451619"><a name="p19230195451619"></a><a name="p19230195451619"></a>VSTAR模式超参结构体。</p>
</td>
</tr>
<tr id="row5230155410161"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p202301754181615"><a name="p202301754181615"></a><a name="p202301754181615"></a><strong id="b1230125481610"><a name="b1230125481610"></a><a name="b1230125481610"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p14230654111611"><a name="p14230654111611"></a><a name="p14230654111611"></a>无</p>
</td>
</tr>
<tr id="row152301754191616"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p72301154101620"><a name="p72301154101620"></a><a name="p72301154101620"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p1323045491617"><a name="p1323045491617"></a><a name="p1323045491617"></a>无</p>
</td>
</tr>
<tr id="row52301454181614"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p1323065491615"><a name="p1323065491615"></a><a name="p1323065491615"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p572594452912"><a name="p572594452912"></a><a name="p572594452912"></a>参数默认值见<a href="#table42921559204019">AscendIndexVstarHyperParams</a>。</p>
</td>
</tr>
</tbody>
</table>

<a id="table42921559204019"></a>
<table><tbody><tr id="row1929245944010"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12921659194012"><a name="p12921659194012"></a><a name="p12921659194012"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p15292115964010"><a name="p15292115964010"></a><a name="p15292115964010"></a>AscendIndexVstarHyperParams(int nProbeL1, int nProbeL2, int l3SegmentNum);</p>
</td>
</tr>
<tr id="row62921559174010"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p82924592406"><a name="p82924592406"></a><a name="p82924592406"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p129205994013"><a name="p129205994013"></a><a name="p129205994013"></a>VSTAR模式超参结构体</p>
</td>
</tr>
<tr id="row929275984019"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p129316593406"><a name="p129316593406"></a><a name="p129316593406"></a><strong id="b11293359184015"><a name="b11293359184015"></a><a name="b11293359184015"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p1029355919406"><a name="p1029355919406"></a><a name="p1029355919406"></a><strong id="b13114027192810"><a name="b13114027192810"></a><a name="b13114027192810"></a>int nProbeL1</strong>：一阶段检索搜索的聚类数。</p>
<p id="p107015014422"><a name="p107015014422"></a><a name="p107015014422"></a><strong id="b18772202952812"><a name="b18772202952812"></a><a name="b18772202952812"></a>int nProbeL2</strong>：二阶段检索搜索的聚类数。</p>
<p id="p9701450174216"><a name="p9701450174216"></a><a name="p9701450174216"></a><strong id="b182823292820"><a name="b182823292820"></a><a name="b182823292820"></a>int l3SegmentNum</strong>：三阶段检索的段数量，从nProbeL2中用于搜索数据段数。</p>
</td>
</tr>
<tr id="row18293175964011"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p172930593402"><a name="p172930593402"></a><a name="p172930593402"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p19293459104014"><a name="p19293459104014"></a><a name="p19293459104014"></a>无</p>
</td>
</tr>
<tr id="row162937595403"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p1229345910409"><a name="p1229345910409"></a><a name="p1229345910409"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><a name="ul1287219505284"></a><a name="ul1287219505284"></a><ul id="ul1287219505284"><li>nProbeL1∈ [32, nListL1]，且nProbeL1必须是8的整数倍，默认值为<span class="parmvalue" id="parmvalue3992032812"><a name="parmvalue3992032812"></a><a name="parmvalue3992032812"></a>“72”</span>。</li><li>nProbeL2∈ (16, nProbeL1 * n]，当dim为1024时n为16，其余维度n为32，且nProbeL2必须是8的整数倍，默认值为<span class="parmvalue" id="parmvalue710178112811"><a name="parmvalue710178112811"></a><a name="parmvalue710178112811"></a>“64”</span>。</li><li>l3SegmentNum∈ (100, 5000]，且l3SegmentNum必须是8的整数倍。默认值为“512”。</li></ul>
</td>
</tr>
</tbody>
</table>

## AscendIndexHyperParams接口<a name="ZH-CN_TOPIC_0000002049325253"></a>

<a name="table93967711712"></a>
<table><tbody><tr id="row1042207151710"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.1.1"><p id="p74221719175"><a name="p74221719175"></a><a name="p74221719175"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.1.1 "><p id="p64221670173"><a name="p64221670173"></a><a name="p64221670173"></a>AscendIndexHyperParams();</p>
</td>
</tr>
<tr id="row44222771712"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.2.1"><p id="p2422173179"><a name="p2422173179"></a><a name="p2422173179"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.2.1 "><p id="p154225713171"><a name="p154225713171"></a><a name="p154225713171"></a>GREAT检索时的超参数结构体。</p>
</td>
</tr>
<tr id="row14231577178"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.3.1"><p id="p242314711720"><a name="p242314711720"></a><a name="p242314711720"></a><strong id="b3423177101713"><a name="b3423177101713"></a><a name="b3423177101713"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.3.1 "><p id="p14843135984413"><a name="p14843135984413"></a><a name="p14843135984413"></a>无</p>
</td>
</tr>
<tr id="row8423127161719"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.4.1"><p id="p542312710172"><a name="p542312710172"></a><a name="p542312710172"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.4.1 "><p id="p12423373171"><a name="p12423373171"></a><a name="p12423373171"></a>无</p>
</td>
</tr>
<tr id="row194231972176"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.5.1"><p id="p642315711713"><a name="p642315711713"></a><a name="p642315711713"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.5.1 "><p id="p1739165164518"><a name="p1739165164518"></a><a name="p1739165164518"></a>参数默认值见<a href="#table1334182412417">AscendIndexHyperParams</a>。</p>
</td>
</tr>
</tbody>
</table>

<a id="table1334182412417"></a>
<table><tbody><tr id="row7341224164110"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.1.1"><p id="p17341524124117"><a name="p17341524124117"></a><a name="p17341524124117"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.1.1 "><p id="p9341192424110"><a name="p9341192424110"></a><a name="p9341192424110"></a>AscendIndexHyperParams(const std::string&amp; mode, const AscendIndexVstarHyperParams&amp; vstarHyperParam, int expandingFactor);</p>
</td>
</tr>
<tr id="row12341102415417"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.2.1"><p id="p8341152417417"><a name="p8341152417417"></a><a name="p8341152417417"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.2.1 "><p id="p16341112444120"><a name="p16341112444120"></a><a name="p16341112444120"></a>GREAT检索时的超参数结构体。</p>
</td>
</tr>
<tr id="row183411924124115"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.3.1"><p id="p1634120247416"><a name="p1634120247416"></a><a name="p1634120247416"></a><strong id="b1534152414118"><a name="b1534152414118"></a><a name="b1534152414118"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.3.1 "><p id="p74235712170"><a name="p74235712170"></a><a name="p74235712170"></a><strong id="b8576358446"><a name="b8576358446"></a><a name="b8576358446"></a>const std::string&amp; mode</strong>：指定算法模式。</p>
<p id="p757103514413"><a name="p757103514413"></a><a name="p757103514413"></a><strong id="b11138125032113"><a name="b11138125032113"></a><a name="b11138125032113"></a>const AscendIndexVstarHyperParams&amp; vstarHyperParam：</strong>详细说明请参见<a href="#table42921559204019">AscendIndexVstarHyperParams</a>。</p>
<p id="p1557203524420"><a name="p1557203524420"></a><a name="p1557203524420"></a><strong id="b105743519442"><a name="b105743519442"></a><a name="b105743519442"></a>int expandingFactor</strong>：检索阶段每一层搜索时邻居的数量，注意与构图阶段的<strong id="b1878902774314"><a name="b1878902774314"></a><a name="b1878902774314"></a>expandingFactor</strong>区分。</p>
</td>
</tr>
<tr id="row5341172424119"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.4.1"><p id="p15341172424120"><a name="p15341172424120"></a><a name="p15341172424120"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.4.1 "><p id="p734112464111"><a name="p734112464111"></a><a name="p734112464111"></a>无</p>
</td>
</tr>
<tr id="row14341224164113"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.5.1"><p id="p134119246417"><a name="p134119246417"></a><a name="p134119246417"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.5.1 "><a name="ul1944290182915"></a><a name="ul1944290182915"></a><ul id="ul1944290182915"><li>mode∈ {“KMode”,“AKMode”}。默认值“AKMode”。</li><li>expandingFactor ∈ [10, 200]。默认值为<span class="parmvalue" id="parmvalue139863467317"><a name="parmvalue139863467317"></a><a name="parmvalue139863467317"></a>“150”</span>。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table88027219236"></a>
<table><tbody><tr id="row280232117234"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.1.1"><p id="p10802112111235"><a name="p10802112111235"></a><a name="p10802112111235"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.1.1 "><p id="p4802132116235"><a name="p4802132116235"></a><a name="p4802132116235"></a>AscendIndexHyperParams(const std::string&amp; mode, int expandingFactor);</p>
</td>
</tr>
<tr id="row080292182312"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.2.1"><p id="p680220214236"><a name="p680220214236"></a><a name="p680220214236"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.2.1 "><p id="p20802142112316"><a name="p20802142112316"></a><a name="p20802142112316"></a>GREAT检索时的超参数结构体。</p>
</td>
</tr>
<tr id="row198021821192318"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.3.1"><p id="p1780216210239"><a name="p1780216210239"></a><a name="p1780216210239"></a><strong id="b1480272114235"><a name="b1480272114235"></a><a name="b1480272114235"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.3.1 "><p id="p15802112162316"><a name="p15802112162316"></a><a name="p15802112162316"></a><strong id="b8802721132317"><a name="b8802721132317"></a><a name="b8802721132317"></a>const std::string&amp; mode</strong>：指定算法模式。</p>
<p id="p0802162182317"><a name="p0802162182317"></a><a name="p0802162182317"></a><strong id="b1980242122318"><a name="b1980242122318"></a><a name="b1980242122318"></a>int expandingFactor</strong>：检索阶段每一层搜索时邻居的数量，注意与构图阶段的<strong id="b1380220212230"><a name="b1380220212230"></a><a name="b1380220212230"></a>expandingFactor</strong>区分。</p>
</td>
</tr>
<tr id="row980282122314"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.4.1"><p id="p28021621192314"><a name="p28021621192314"></a><a name="p28021621192314"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.4.1 "><p id="p10802521122312"><a name="p10802521122312"></a><a name="p10802521122312"></a>无</p>
</td>
</tr>
<tr id="row1580282118238"><th class="firstcol" valign="top" width="19.35%" id="mcps1.1.3.5.1"><p id="p2802192192310"><a name="p2802192192310"></a><a name="p2802192192310"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="80.65%" headers="mcps1.1.3.5.1 "><a name="ul1480216213235"></a><a name="ul1480216213235"></a><ul id="ul1480216213235"><li>mode∈ {“KMode”,“AKMode”}。默认值“AKMode”。</li><li>expandingFactor ∈ [10, 200]。默认值为<span class="parmvalue" id="parmvalue1580252119235"><a name="parmvalue1580252119235"></a><a name="parmvalue1580252119235"></a>“150”</span>。</li></ul>
</td>
</tr>
</tbody>
</table>

## AscendIndexSearchParams接口<a name="ZH-CN_TOPIC_0000002044950949"></a>

<a name="table414612258177"></a>
<table><tbody><tr id="row118413250172"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2018462520172"><a name="p2018462520172"></a><a name="p2018462520172"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16184172511716"><a name="p16184172511716"></a><a name="p16184172511716"></a>AscendIndexSearchParams(size_t n, std::vector&lt;float&gt;&amp; queryData, int topK, std::vector&lt;float&gt;&amp; dists, std::vector&lt;int64_t&gt;&amp; labels);</p>
</td>
</tr>
<tr id="row16184162515173"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p61841251179"><a name="p61841251179"></a><a name="p61841251179"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p14184112541715"><a name="p14184112541715"></a><a name="p14184112541715"></a>检索时的搜索参数结构体。</p>
</td>
</tr>
<tr id="row191848251175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p4184125201718"><a name="p4184125201718"></a><a name="p4184125201718"></a><strong id="b7184122591715"><a name="b7184122591715"></a><a name="b7184122591715"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p126612528811"><a name="p126612528811"></a><a name="p126612528811"></a>无</p>
</td>
</tr>
<tr id="row1518572551717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p11851825121717"><a name="p11851825121717"></a><a name="p11851825121717"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p789264511813"><a name="p789264511813"></a><a name="p789264511813"></a>无</p>
</td>
</tr>
<tr id="row1185425101713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12185122518179"><a name="p12185122518179"></a><a name="p12185122518179"></a>参数值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p5761142186"><a name="p5761142186"></a><a name="p5761142186"></a><strong id="b4761114219820"><a name="b4761114219820"></a><a name="b4761114219820"></a>size_t n</strong>：查询的特征向量的条数。</p>
<p id="p1941741042119"><a name="p1941741042119"></a><a name="p1941741042119"></a><strong id="b089323152313"><a name="b089323152313"></a><a name="b089323152313"></a>std::vector&lt;float&gt;&amp; queryData</strong>：特征向量数据。</p>
<p id="p19761124216818"><a name="p19761124216818"></a><a name="p19761124216818"></a><strong id="b117611142286"><a name="b117611142286"></a><a name="b117611142286"></a>int topK：</strong>需要返回的最相似的结果个数。</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b1671817282232"><a name="b1671817282232"></a><a name="b1671817282232"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前“topK”个向量间的距离值。</p>
<p id="p5548356122111"><a name="p5548356122111"></a><a name="p5548356122111"></a><strong id="b287214390241"><a name="b287214390241"></a><a name="b287214390241"></a>std::vector&lt;int64_t&gt;&amp;</strong> <strong id="b148237305104"><a name="b148237305104"></a><a name="b148237305104"></a>labels</strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row4185425171719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1218512541712"><a name="p1218512541712"></a><a name="p1218512541712"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5481551192220"></a><a name="ul5481551192220"></a><ul id="ul5481551192220"><li>topK ∈ (0, 4096]。</li><li>n ∈ (0, 10000]。</li><li>queryData不能为空，且数据长度必须大于等于n * dim。</li><li>dists不能为空，且指向的数据长度必须大于等于n * topK。</li><li>labels不能为空，且指向的数据长度必须大于等于n * topK。</li></ul>
</td>
</tr>
</tbody>
</table>
