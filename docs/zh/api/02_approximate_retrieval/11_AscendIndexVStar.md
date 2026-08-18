# AscendIndexVStar<a name="ZH-CN_TOPIC_0000002044351677"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002044510693"></a>

昇腾自研向量检索算法，为用户提供昇腾侧高维大底库近似检索能力。使用自研矩阵近似策略，压缩特征向量后存底库，最后使用自研检索策略在底库中检索得到topK个最近似向量结果。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

此算法主要针对大底库场景的近似模糊搜索，相较暴力检索精度已有一定损失。在小底库场景，建议适当加大超参值，可改善精度损失问题。

## AscendIndexVStar接口<a name="ZH-CN_TOPIC_0000002044513265"></a>

> [!NOTE]
>
>- 创建Index实例时传入的参数params，需根据实际情况设置其中的params.dim。
>- params.subSpaceDim和params.nlist应与码本训练时对应参数保持一致。

<a name="table13851535141118"></a>
<table><tbody><tr id="row1444303516117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2443173517119"><a name="p2443173517119"></a><a name="p2443173517119"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p144316350115"><a name="p144316350115"></a><a name="p144316350115"></a>explicit AscendIndexVStar(const AscendIndexVstarInitParams&amp; params);</p>
</td>
</tr>
<tr id="row1944313352116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p8443035101113"><a name="p8443035101113"></a><a name="p8443035101113"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1544353591119"><a name="p1544353591119"></a><a name="p1544353591119"></a>AscendIndexVStar的构造函数，根据params中配置的值构造对应维度的Index。</p>
</td>
</tr>
<tr id="row14443153510119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14443935191116"><a name="p14443935191116"></a><a name="p14443935191116"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p11443153591119"><a name="p11443153591119"></a><a name="p11443153591119"></a><strong id="b16412146163213"><a name="b16412146163213"></a><a name="b16412146163213"></a>const AscendIndexVstarInitParams&amp; params</strong>：构造配置参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>。</p>
</td>
</tr>
<tr id="row20444153513115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p344411352117"><a name="p344411352117"></a><a name="p344411352117"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1044411350112"><a name="p1044411350112"></a><a name="p1044411350112"></a>无</p>
</td>
</tr>
<tr id="row1344473515113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p164444357118"><a name="p164444357118"></a><a name="p164444357118"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p154441335101114"><a name="p154441335101114"></a><a name="p154441335101114"></a>无</p>
</td>
</tr>
<tr id="row184441035191113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p174441135121113"><a name="p174441135121113"></a><a name="p174441135121113"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p127451568374"><a name="p127451568374"></a><a name="p127451568374"></a>具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>。</p>
</td>
</tr>
</tbody>
</table>

<a name="table11631734281"></a>
<table><tbody><tr id="row1997123419810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p39793416811"><a name="p39793416811"></a><a name="p39793416811"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p129783413815"><a name="p129783413815"></a><a name="p129783413815"></a>AscendIndexVStar(const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</p>
</td>
</tr>
<tr id="row209716341483"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p4979344810"><a name="p4979344810"></a><a name="p4979344810"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7972344815"><a name="p7972344815"></a><a name="p7972344815"></a>AscendIndexVStar的构造函数，根据deviceList构造未知输入数据维度和超参的Index。</p>
</td>
</tr>
<tr id="row119763411811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p109783412815"><a name="p109783412815"></a><a name="p109783412815"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p59716341788"><a name="p59716341788"></a><a name="p59716341788"></a><strong id="b398912386372"><a name="b398912386372"></a><a name="b398912386372"></a>const std::vector&lt;int&gt;&amp; deviceList</strong>：device侧设备ID。</p>
<p id="p1797113416815"><a name="p1797113416815"></a><a name="p1797113416815"></a><strong id="b173471141143717"><a name="b173471141143717"></a><a name="b173471141143717"></a>bool verbose</strong>：是否开启<span class="parmvalue" id="parmvalue20991318193711"><a name="parmvalue20991318193711"></a><a name="parmvalue20991318193711"></a>“verbose”</span>选项，开启后部分操作提供额外的打印提示。默认值为<span class="parmvalue" id="parmvalue810213723716"><a name="parmvalue810213723716"></a><a name="parmvalue810213723716"></a>“false”</span>。</p>
</td>
</tr>
<tr id="row99711348819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p59717343819"><a name="p59717343819"></a><a name="p59717343819"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p199717341384"><a name="p199717341384"></a><a name="p199717341384"></a>无</p>
</td>
</tr>
<tr id="row16971534688"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2971934789"><a name="p2971934789"></a><a name="p2971934789"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1097034289"><a name="p1097034289"></a><a name="p1097034289"></a>无</p>
</td>
</tr>
<tr id="row4987340814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p139812341810"><a name="p139812341810"></a><a name="p139812341810"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1087613504257"></a><a name="ul1087613504257"></a><ul id="ul1087613504257"><li><span class="parmvalue" id="parmvalue2072110402273"><a name="parmvalue2072110402273"></a><a name="parmvalue2072110402273"></a>“deviceList”</span>需要为合法有效的设备ID，当前仅支持一个device设备。</li><li>使用此构造函数创建Index实例后，需要先调用<span class="parmvalue" id="parmvalue158001625183711"><a name="parmvalue158001625183711"></a><a name="parmvalue158001625183711"></a>“LoadIndex”</span>加载事先落盘后的Index实例，然后再进行其他操作。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table8937623141615"></a>
<table><tbody><tr id="row5963112391615"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p19963192315165"><a name="p19963192315165"></a><a name="p19963192315165"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13963423141618"><a name="p13963423141618"></a><a name="p13963423141618"></a>AscendIndexVStar(const AscendIndexVStar&amp;) = delete;</p>
</td>
</tr>
<tr id="row59631723101617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p596315235169"><a name="p596315235169"></a><a name="p596315235169"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p19631423141620"><a name="p19631423141620"></a><a name="p19631423141620"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row11963112361618"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p796352311610"><a name="p796352311610"></a><a name="p796352311610"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p2963132371615"><a name="p2963132371615"></a><a name="p2963132371615"></a><strong id="b854016545917"><a name="b854016545917"></a><a name="b854016545917"></a>const AscendIndexVStar&amp;</strong>：AscendIndexVStar对象。</p>
</td>
</tr>
<tr id="row1963623181617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1296319234162"><a name="p1296319234162"></a><a name="p1296319234162"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p109637233167"><a name="p109637233167"></a><a name="p109637233167"></a>无</p>
</td>
</tr>
<tr id="row9963152391619"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p59636239166"><a name="p59636239166"></a><a name="p59636239166"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p5963723141619"><a name="p5963723141619"></a><a name="p5963723141619"></a>无</p>
</td>
</tr>
<tr id="row396342301616"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p39631623151614"><a name="p39631623151614"></a><a name="p39631623151614"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1496322301615"><a name="p1496322301615"></a><a name="p1496322301615"></a>无</p>
</td>
</tr>
</tbody>
</table>

## LoadIndex接口<a name="ZH-CN_TOPIC_0000002008232688"></a>

<a name="table950712481817"></a>
<table><tbody><tr id="row2539174813819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p15539148282"><a name="p15539148282"></a><a name="p15539148282"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p18539164814811"><a name="p18539164814811"></a><a name="p18539164814811"></a>APP_ERROR LoadIndex(const std::string&amp; indexPath, AscendIndexVStar* indexVStar = nullptr);</p>
</td>
</tr>
<tr id="row15539184819811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p85401048783"><a name="p85401048783"></a><a name="p85401048783"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p105402484812"><a name="p105402484812"></a><a name="p105402484812"></a>将已有索引Index从磁盘读入Device。</p>
</td>
</tr>
<tr id="row5540114820810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p85408481819"><a name="p85408481819"></a><a name="p85408481819"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12540154812814"><a name="p12540154812814"></a><a name="p12540154812814"></a><strong id="b1536925918373"><a name="b1536925918373"></a><a name="b1536925918373"></a>const std::string&amp; indexPath</strong>：数据文件路径；</p>
<p id="p25403481482"><a name="p25403481482"></a><a name="p25403481482"></a><strong id="b32032253819"><a name="b32032253819"></a><a name="b32032253819"></a>AscendIndexVStar* indexVStar</strong>：仅在调用<span class="parmvalue" id="parmvalue137410923815"><a name="parmvalue137410923815"></a><a name="parmvalue137410923815"></a>“MultiSearch”</span>接口场景使用，使所有Index共用第一个Index的码本。</p>
</td>
</tr>
<tr id="row5540148780"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9540748082"><a name="p9540748082"></a><a name="p9540748082"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p95401548281"><a name="p95401548281"></a><a name="p95401548281"></a>无</p>
</td>
</tr>
<tr id="row6540948989"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p15540134813819"><a name="p15540134813819"></a><a name="p15540134813819"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row205408480812"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1654010483819"><a name="p1654010483819"></a><a name="p1654010483819"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul42641143101915"></a><a name="ul42641143101915"></a><ul id="ul42641143101915"><li>用户应保证<span class="parmname" id="parmname1263105191815"><a name="parmname1263105191815"></a><a name="parmname1263105191815"></a>“indexPath”</span>文件路径所在的目录存在，且执行用户对目录具有读权限；出于安全加固考虑，目录层级中不能含有软链接。</li><li>indexVStar在<span class="parmname" id="parmname8989319161812"><a name="parmname8989319161812"></a><a name="parmname8989319161812"></a>“MultiSearch”</span>场景下不能为空指针；在单Index场景下必须为空指针，若单Index场景下使用合法Index指针，则原Index码本将被参数Index实例码本替代。</li></ul>
</td>
</tr>
</tbody>
</table>

## WriteIndex接口<a name="ZH-CN_TOPIC_0000002044351681"></a>

<a name="table29774016915"></a>
<table><tbody><tr id="row924112912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1213116916"><a name="p1213116916"></a><a name="p1213116916"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p431711194"><a name="p431711194"></a><a name="p431711194"></a>APP_ERROR WriteIndex(const std::string&amp; indexPath);</p>
</td>
</tr>
<tr id="row103511896"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p193131799"><a name="p193131799"></a><a name="p193131799"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p113131293"><a name="p113131293"></a><a name="p113131293"></a>将索引index写入磁盘。</p>
</td>
</tr>
<tr id="row1432113915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1931212911"><a name="p1931212911"></a><a name="p1931212911"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p5351396"><a name="p5351396"></a><a name="p5351396"></a><strong id="b2635133772118"><a name="b2635133772118"></a><a name="b2635133772118"></a>const std::string&amp; indexPath</strong>：保存的数据文件路径。</p>
</td>
</tr>
<tr id="row43151894"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1331411698"><a name="p1331411698"></a><a name="p1331411698"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p19317113913"><a name="p19317113913"></a><a name="p19317113913"></a>无</p>
</td>
</tr>
<tr id="row231314913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1338118915"><a name="p1338118915"></a><a name="p1338118915"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1535114913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p6319110914"><a name="p6319110914"></a><a name="p6319110914"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul138531628172014"></a><a name="ul138531628172014"></a><ul id="ul138531628172014"><li>用户应保证<span class="parmname" id="parmname943171915206"><a name="parmname943171915206"></a><a name="parmname943171915206"></a>“indexPath”</span>文件路径所在的目录存在，且执行用户对目录具有写权限；出于安全加固考虑，目录层级中不能含有软链接。</li><li>当文件已经存在时，将执行覆盖写，此时程序执行用户应该是该文件的属主。</li></ul>
</td>
</tr>
</tbody>
</table>

## AddCodeBooksByIndex接口<a name="ZH-CN_TOPIC_0000002044510697"></a>

<a name="table81089131197"></a>
<table><tbody><tr id="row5133111315917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p121331113993"><a name="p121331113993"></a><a name="p121331113993"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p17133713892"><a name="p17133713892"></a><a name="p17133713892"></a>APP_ERROR AddCodeBooksByIndex(AscendIndexVStar&amp; indexVStar);</p>
</td>
</tr>
<tr id="row5133013098"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p31341513792"><a name="p31341513792"></a><a name="p31341513792"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p4134151310918"><a name="p4134151310918"></a><a name="p4134151310918"></a>多Index检索场景下，当前Index通过该接口，将传入的参数Index实例的码本载入当前Index。</p>
</td>
</tr>
<tr id="row913418131091"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p513418131293"><a name="p513418131293"></a><a name="p513418131293"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19134191319910"><a name="p19134191319910"></a><a name="p19134191319910"></a><strong id="b624093316215"><a name="b624093316215"></a><a name="b624093316215"></a>AscendIndexVStar&amp; indexVStar</strong>：已填充好码本的Index实例。</p>
</td>
</tr>
<tr id="row313416135910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p10134171311918"><a name="p10134171311918"></a><a name="p10134171311918"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p101347135910"><a name="p101347135910"></a><a name="p101347135910"></a>无</p>
</td>
</tr>
<tr id="row31348131496"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p413419131599"><a name="p413419131599"></a><a name="p413419131599"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row131349131194"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p10134111315919"><a name="p10134111315919"></a><a name="p10134111315919"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p111341913992"><a name="p111341913992"></a><a name="p111341913992"></a>该接口仅在<span class="parmname" id="parmname373513015226"><a name="parmname373513015226"></a><a name="parmname373513015226"></a>“MultiSearch”</span>场景下使用。</p>
</td>
</tr>
</tbody>
</table>

## AddCodeBooksByPath接口<a name="ZH-CN_TOPIC_0000002008390980"></a>

<a name="table1523424814919"></a>
<table><tbody><tr id="row226212481493"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p726234818912"><a name="p726234818912"></a><a name="p726234818912"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p526220483915"><a name="p526220483915"></a><a name="p526220483915"></a>APP_ERROR AddCodeBooksByPath(const std::string&amp; codeBooksPath);</p>
</td>
</tr>
<tr id="row172621484910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p142622488919"><a name="p142622488919"></a><a name="p142622488919"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82625481093"><a name="p82625481093"></a><a name="p82625481093"></a>通过码本路径将码本加载到当前Index。</p>
</td>
</tr>
<tr id="row42621548992"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p102621481499"><a name="p102621481499"></a><a name="p102621481499"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1826294819913"><a name="p1826294819913"></a><a name="p1826294819913"></a><strong id="b1442284619222"><a name="b1442284619222"></a><a name="b1442284619222"></a>const std::string&amp; codeBooksPath</strong>：码本数据文件路径。</p>
</td>
</tr>
<tr id="row122621448591"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p726244817911"><a name="p726244817911"></a><a name="p726244817911"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1726213484919"><a name="p1726213484919"></a><a name="p1726213484919"></a>无</p>
</td>
</tr>
<tr id="row2262194812918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1026316481695"><a name="p1026316481695"></a><a name="p1026316481695"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row102631648898"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1326310482916"><a name="p1326310482916"></a><a name="p1326310482916"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p3263348593"><a name="p3263348593"></a><a name="p3263348593"></a>用户应保证<span class="parmname" id="parmname79933913221"><a name="parmname79933913221"></a><a name="parmname79933913221"></a>“codeBooksPath”</span>文件路径所在的目录存在，且执行用户对目录具有读权限；出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

## Add接口<a name="ZH-CN_TOPIC_0000002008232692"></a>

<a name="table18288921121213"></a>
<table><tbody><tr id="row63171421121210"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.1.1"><p id="p1931702111125"><a name="p1931702111125"></a><a name="p1931702111125"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.1.1 "><p id="p2317721111218"><a name="p2317721111218"></a><a name="p2317721111218"></a>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseData);</p>
</td>
</tr>
<tr id="row031782114123"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.2.1"><p id="p13177211124"><a name="p13177211124"></a><a name="p13177211124"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.2.1 "><p id="p331702181218"><a name="p331702181218"></a><a name="p331702181218"></a>实现AscendIndexVStar建库和向底库中添加新的特征向量的功能。</p>
</td>
</tr>
<tr id="row131742181210"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.3.1"><p id="p1431782121219"><a name="p1431782121219"></a><a name="p1431782121219"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.3.1 "><p id="p12317142117122"><a name="p12317142117122"></a><a name="p12317142117122"></a><strong id="b118938561239"><a name="b118938561239"></a><a name="b118938561239"></a>const std::vector&lt;float&gt;&amp; baseData</strong>：待添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row631713212128"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.4.1"><p id="p4317162110122"><a name="p4317162110122"></a><a name="p4317162110122"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.4.1 "><p id="p1531812171210"><a name="p1531812171210"></a><a name="p1531812171210"></a>无</p>
</td>
</tr>
<tr id="row2318122151218"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.5.1"><p id="p131822151218"><a name="p131822151218"></a><a name="p131822151218"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row73181321111210"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.6.1"><p id="p1231892114123"><a name="p1231892114123"></a><a name="p1231892114123"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.6.1 "><p id="p10318172121220"><a name="p10318172121220"></a><a name="p10318172121220"></a><span class="parmname" id="parmname165671242232"><a name="parmname165671242232"></a><a name="parmname165671242232"></a>“baseData”</span>长度应该为n * dim，n为待添加进底库的向量数量，dim为向量维度。n ∈ [10000, 1e8]。</p>
<p id="p10318821171213"><a name="p10318821171213"></a><a name="p10318821171213"></a>该接口不设置ID，底库默认ID范围为[ntotal, ntotal + n)，其中ntotal为Index已有底库数量，n为待添加进底库的向量数量。</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
>- Add接口不能与AddWithIds接口混用。
>- 使用Add接口后，Search结果的labels可能会重复，如果业务上对label有要求，建议使用[AddWithIds接口](#addwithids接口)。

## AddWithIds接口<a name="ZH-CN_TOPIC_0000002044351685"></a>

<a name="table32483414124"></a>
<table><tbody><tr id="row856113412127"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1156034171214"><a name="p1156034171214"></a><a name="p1156034171214"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p256183451212"><a name="p256183451212"></a><a name="p256183451212"></a>APP_ERROR AddWithIds(const std::vector&lt;float&gt;&amp; baseData, const std::vector&lt;int64_t&gt;&amp; ids);</p>
</td>
</tr>
<tr id="row145683451213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p05683441216"><a name="p05683441216"></a><a name="p05683441216"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p12561734201214"><a name="p12561734201214"></a><a name="p12561734201214"></a>实现AscendIndexVStar建库和向底库中添加新的特征向量的功能。允许用户指定添加底库的ID。</p>
</td>
</tr>
<tr id="row856134161214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1656133491219"><a name="p1656133491219"></a><a name="p1656133491219"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p25616342125"><a name="p25616342125"></a><a name="p25616342125"></a><strong id="b106077431267"><a name="b106077431267"></a><a name="b106077431267"></a>const std::vector&lt;float&gt;&amp; baseData</strong>：待添加进底库的特征向量。</p>
<p id="p0619442133010"><a name="p0619442133010"></a><a name="p0619442133010"></a><strong id="b23161447132610"><a name="b23161447132610"></a><a name="b23161447132610"></a>const std::vector&lt;int64_t&gt;&amp; ids</strong>：待添加底库映射ID的数组。</p>
</td>
</tr>
<tr id="row856133441220"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1056234151218"><a name="p1056234151218"></a><a name="p1056234151218"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14563341129"><a name="p14563341129"></a><a name="p14563341129"></a>无</p>
</td>
</tr>
<tr id="row15561834171214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1156163401219"><a name="p1156163401219"></a><a name="p1156163401219"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row195643471211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1156113441218"><a name="p1156113441218"></a><a name="p1156113441218"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul16435184311252"></a><a name="ul16435184311252"></a><ul id="ul16435184311252"><li><span class="parmname" id="parmname1848013148256"><a name="parmname1848013148256"></a><a name="parmname1848013148256"></a>“baseData”</span>长度应该为n * dim，n为待添加进底库的向量数量，dim为向量维度。</li><li><span class="parmname" id="parmname1942422062518"><a name="parmname1942422062518"></a><a name="parmname1942422062518"></a>“ids”</span>长度必须为n，用户需要根据自己的业务场景，保证<span class="parmname" id="parmname20685131262110"><a name="parmname20685131262110"></a><a name="parmname20685131262110"></a>“ids”</span>的合法性，如底库中存在重复的ID，检索结果中的"label"将无法对应具体的底库向量。</li><li>n∈[10000，1e8]。</li></ul>
</td>
</tr>
</tbody>
</table>

## DeleteByIds接口<a name="ZH-CN_TOPIC_0000002044510701"></a>

<a name="table1284884631210"></a>
<table><tbody><tr id="row18872114601211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p8872846171211"><a name="p8872846171211"></a><a name="p8872846171211"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16872174651217"><a name="p16872174651217"></a><a name="p16872174651217"></a>APP_ERROR DeleteByIds(const std::vector&lt;int64_t&gt;&amp; ids);</p>
</td>
</tr>
<tr id="row0872246171215"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p208721469123"><a name="p208721469123"></a><a name="p208721469123"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p78721846171211"><a name="p78721846171211"></a><a name="p78721846171211"></a>根据参数中id数组删除底库中对应id的向量数据。</p>
</td>
</tr>
<tr id="row68723468124"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14872846121210"><a name="p14872846121210"></a><a name="p14872846121210"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p38721446191220"><a name="p38721446191220"></a><a name="p38721446191220"></a><strong id="b5838162916389"><a name="b5838162916389"></a><a name="b5838162916389"></a>const std::vector&lt;int64_t&gt;&amp; ids</strong>：待删除底库数据的向量ID数组。</p>
</td>
</tr>
<tr id="row1287214641219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p20872184610120"><a name="p20872184610120"></a><a name="p20872184610120"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8872194691219"><a name="p8872194691219"></a><a name="p8872194691219"></a>无</p>
</td>
</tr>
<tr id="row1487284620122"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p14872114671213"><a name="p14872114671213"></a><a name="p14872114671213"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row2087344620128"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p138736461125"><a name="p138736461125"></a><a name="p138736461125"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10873246121219"><a name="p10873246121219"></a><a name="p10873246121219"></a>ids中的ID，应为添加底库接口中的ID。</p>
</td>
</tr>
</tbody>
</table>

## DeleteById接口<a name="ZH-CN_TOPIC_0000002008390984"></a>

<a name="table9845165841212"></a>
<table><tbody><tr id="row6870175812125"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p38700589124"><a name="p38700589124"></a><a name="p38700589124"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p8870155813121"><a name="p8870155813121"></a><a name="p8870155813121"></a>APP_ERROR DeleteById(int64_t id);</p>
</td>
</tr>
<tr id="row10870358171219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p8870135881218"><a name="p8870135881218"></a><a name="p8870135881218"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16870155811216"><a name="p16870155811216"></a><a name="p16870155811216"></a>根据参数ID删除底库中对应ID的向量数据。</p>
</td>
</tr>
<tr id="row108701158141212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p108706584128"><a name="p108706584128"></a><a name="p108706584128"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p2870458181216"><a name="p2870458181216"></a><a name="p2870458181216"></a><strong id="b16531331103917"><a name="b16531331103917"></a><a name="b16531331103917"></a>int64_t id</strong>：待删除的底库向量ID。</p>
</td>
</tr>
<tr id="row38704585124"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p16870105810120"><a name="p16870105810120"></a><a name="p16870105810120"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p787095891216"><a name="p787095891216"></a><a name="p787095891216"></a>无</p>
</td>
</tr>
<tr id="row1987075810126"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p16870155841213"><a name="p16870155841213"></a><a name="p16870155841213"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row887014580125"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19870958191213"><a name="p19870958191213"></a><a name="p19870958191213"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p8870135861218"><a name="p8870135861218"></a><a name="p8870135861218"></a>ID应为添加底库接口中的ID。</p>
</td>
</tr>
</tbody>
</table>

## DeleteByRange接口<a name="ZH-CN_TOPIC_0000002008232696"></a>

<a name="table103969158136"></a>
<table><tbody><tr id="row142117152131"><th class="firstcol" valign="top" width="20.13%" id="mcps1.1.3.1.1"><p id="p84219151131"><a name="p84219151131"></a><a name="p84219151131"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.86999999999999%" headers="mcps1.1.3.1.1 "><p id="p742181519130"><a name="p742181519130"></a><a name="p742181519130"></a>APP_ERROR DeleteByRange(int64_t startId, int64_t endId);</p>
</td>
</tr>
<tr id="row194211155131"><th class="firstcol" valign="top" width="20.13%" id="mcps1.1.3.2.1"><p id="p5421191571314"><a name="p5421191571314"></a><a name="p5421191571314"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.86999999999999%" headers="mcps1.1.3.2.1 "><p id="p124216158137"><a name="p124216158137"></a><a name="p124216158137"></a>根据参数ID范围删除底库中对应ID的向量数据。</p>
</td>
</tr>
<tr id="row242121514132"><th class="firstcol" valign="top" width="20.13%" id="mcps1.1.3.3.1"><p id="p134221415131314"><a name="p134221415131314"></a><a name="p134221415131314"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.86999999999999%" headers="mcps1.1.3.3.1 "><p id="p1042211152135"><a name="p1042211152135"></a><a name="p1042211152135"></a><strong id="b292912460390"><a name="b292912460390"></a><a name="b292912460390"></a>int64_t startId</strong>：待删除底库的起始ID。</p>
<p id="p34221915111315"><a name="p34221915111315"></a><a name="p34221915111315"></a><strong id="b145908510393"><a name="b145908510393"></a><a name="b145908510393"></a>int64_t endId</strong>：待删除底库的结束ID。</p>
</td>
</tr>
<tr id="row6422715111320"><th class="firstcol" valign="top" width="20.13%" id="mcps1.1.3.4.1"><p id="p1442212158131"><a name="p1442212158131"></a><a name="p1442212158131"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.86999999999999%" headers="mcps1.1.3.4.1 "><p id="p342220159130"><a name="p342220159130"></a><a name="p342220159130"></a>无</p>
</td>
</tr>
<tr id="row542201521315"><th class="firstcol" valign="top" width="20.13%" id="mcps1.1.3.5.1"><p id="p104224152136"><a name="p104224152136"></a><a name="p104224152136"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.86999999999999%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row124228153137"><th class="firstcol" valign="top" width="20.13%" id="mcps1.1.3.6.1"><p id="p104226151133"><a name="p104226151133"></a><a name="p104226151133"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.86999999999999%" headers="mcps1.1.3.6.1 "><p id="p3484155772013"><a name="p3484155772013"></a><a name="p3484155772013"></a>待删除ID应为添加底库接口中的ID，ID ∈ [startId, endId]</p>
</td>
</tr>
</tbody>
</table>

## Search接口<a name="ZH-CN_TOPIC_0000002044351689"></a>

<a name="table197566920146"></a>
<table><tbody><tr id="row17839961420"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p678379101413"><a name="p678379101413"></a><a name="p678379101413"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1278316981412"><a name="p1278316981412"></a><a name="p1278316981412"></a>APP_ERROR Search(const AscendIndexSearchParams&amp; params) const;</p>
</td>
</tr>
<tr id="row197832090141"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p57838941417"><a name="p57838941417"></a><a name="p57838941417"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p678311981417"><a name="p678311981417"></a><a name="p678311981417"></a>实现特征向量检索接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname20990164244015"><a name="parmname20990164244015"></a><a name="parmname20990164244015"></a>“topK”</span>条特征的ID。</p>
</td>
</tr>
<tr id="row27831999143"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p157831994143"><a name="p157831994143"></a><a name="p157831994143"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p18254194711350"><a name="p18254194711350"></a><a name="p18254194711350"></a><strong id="b10734716153611"><a name="b10734716153611"></a><a name="b10734716153611"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a>。</p>
<p id="p37831896140"><a name="p37831896140"></a><a name="p37831896140"></a><strong id="b12506130124118"><a name="b12506130124118"></a><a name="b12506130124118"></a>size_t n</strong>：查询的特征向量的条数。</p>
<p id="p1578369181411"><a name="p1578369181411"></a><a name="p1578369181411"></a><strong id="b03401733204119"><a name="b03401733204119"></a><a name="b03401733204119"></a>std::vector&lt;float&gt;&amp; queryData</strong>：特征向量数据。</p>
<p id="p1678311916145"><a name="p1678311916145"></a><a name="p1678311916145"></a><strong id="b6816183594118"><a name="b6816183594118"></a><a name="b6816183594118"></a>int topK</strong>：需要返回的最相似的结果个数。</p>
</td>
</tr>
<tr id="row178369171418"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p078318918148"><a name="p078318918148"></a><a name="p078318918148"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p878314916143"><a name="p878314916143"></a><a name="p878314916143"></a><strong id="b95222038184114"><a name="b95222038184114"></a><a name="b95222038184114"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前<span class="parmname" id="parmname1493635484016"><a name="parmname1493635484016"></a><a name="parmname1493635484016"></a>“topK”</span>个向量间的距离值。</p>
<p id="p1578310913146"><a name="p1578310913146"></a><a name="p1578310913146"></a><strong id="b833274124118"><a name="b833274124118"></a><a name="b833274124118"></a>std::vector&lt;int64_t&gt;&amp; labels</strong>：查询的距离最近的前<span class="parmname" id="parmname198819568403"><a name="parmname198819568403"></a><a name="parmname198819568403"></a>“topK”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row1178317917147"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p778316961413"><a name="p778316961413"></a><a name="p778316961413"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row078310941416"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p10783189131410"><a name="p10783189131410"></a><a name="p10783189131410"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul16285192613404"></a><a name="ul16285192613404"></a><ul id="ul16285192613404"><li>n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。</li><li>queryData：长度应该大于等于n * dim。</li><li>topK∈(0, 4096]。</li><li>dists、labels：长度应该大于等于n * topK。</li></ul>
</td>
</tr>
</tbody>
</table>

## SearchWithMask接口<a name="ZH-CN_TOPIC_0000002044510705"></a>

<a name="table777072291418"></a>
<table><tbody><tr id="row68061223145"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p28061422111410"><a name="p28061422111410"></a><a name="p28061422111410"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p8806202281413"><a name="p8806202281413"></a><a name="p8806202281413"></a>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask) const;</p>
</td>
</tr>
<tr id="row148061722181410"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p208068225148"><a name="p208068225148"></a><a name="p208068225148"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p12806152215149"><a name="p12806152215149"></a><a name="p12806152215149"></a>特征向量查询接口，根据输入的特征向量返回最相似的topK条特征的ID。mask为0、1比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，0表示不参与，1表示参与。</p>
</td>
</tr>
<tr id="row080672231413"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18061822141419"><a name="p18061822141419"></a><a name="p18061822141419"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p18254194711350"><a name="p18254194711350"></a><a name="p18254194711350"></a><strong id="b8266155353513"><a name="b8266155353513"></a><a name="b8266155353513"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a></p>
<p id="p1880615221147"><a name="p1880615221147"></a><a name="p1880615221147"></a><strong id="b880692414428"><a name="b880692414428"></a><a name="b880692414428"></a>size_t n</strong>：查询的特征向量的条数。</p>
<p id="p38066226149"><a name="p38066226149"></a><a name="p38066226149"></a><strong id="b019792715424"><a name="b019792715424"></a><a name="b019792715424"></a>std::vector&lt;float&gt;&amp; queryData</strong>：特征向量数据。</p>
<p id="p1880610221149"><a name="p1880610221149"></a><a name="p1880610221149"></a><strong id="b198552974214"><a name="b198552974214"></a><a name="b198552974214"></a>int topK</strong>：需要返回的最相似的结果个数。</p>
<p id="p9806122220146"><a name="p9806122220146"></a><a name="p9806122220146"></a><strong id="b1835203218428"><a name="b1835203218428"></a><a name="b1835203218428"></a>const std::vector&lt;uint8_t&gt;&amp; mask</strong>：特征底库掩码。</p>
</td>
</tr>
<tr id="row188061622191413"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1080617229144"><a name="p1080617229144"></a><a name="p1080617229144"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p9806622181419"><a name="p9806622181419"></a><a name="p9806622181419"></a><strong id="b119161115164310"><a name="b119161115164310"></a><a name="b119161115164310"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前<span class="parmname" id="parmname15682194914215"><a name="parmname15682194914215"></a><a name="parmname15682194914215"></a>“topK”</span>个向量间的距离值。</p>
<p id="p980618226148"><a name="p980618226148"></a><a name="p980618226148"></a><strong id="b4475183914320"><a name="b4475183914320"></a><a name="b4475183914320"></a>std::vector&lt;int64_t&gt;&amp; labels</strong>：查询的距离最近的前<span class="parmname" id="parmname5605114164313"><a name="parmname5605114164313"></a><a name="parmname5605114164313"></a>“topK”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row1580792218142"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p78071622171415"><a name="p78071622171415"></a><a name="p78071622171415"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row15807422101411"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p208071422141411"><a name="p208071422141411"></a><a name="p208071422141411"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul6664525436"></a><a name="ul6664525436"></a><ul id="ul6664525436"><li>n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。</li><li>queryData：长度应该大于等于n * dim。</li><li>topK∈(0, 4096]。</li><li>dists、labels：长度应该大于等于n * topK。</li><li>mask：长度应该大于等于n * ceil(ntotal/8)，其中ntotal为底库特征数量。</li></ul>
</td>
</tr>
</tbody>
</table>

## MultiSearch接口<a name="ZH-CN_TOPIC_0000002008390988"></a>

<a name="table158666394146"></a>
<table><tbody><tr id="row1689713971419"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p8897163911141"><a name="p8897163911141"></a><a name="p8897163911141"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19897739141412"><a name="p19897739141412"></a><a name="p19897739141412"></a>APP_ERROR MultiSearch(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, bool merge) const;</p>
</td>
</tr>
<tr id="row1897153914142"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p10897143915143"><a name="p10897143915143"></a><a name="p10897143915143"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p489793914146"><a name="p489793914146"></a><a name="p489793914146"></a>实现从多个AscendIndexVStar库执行特征向量查询的接口，根据输入的特征向量返回最相似的topK条特征距离及ID。</p>
</td>
</tr>
<tr id="row11897153911419"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14897113971416"><a name="p14897113971416"></a><a name="p14897113971416"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p48971339131416"><a name="p48971339131416"></a><a name="p48971339131416"></a><strong id="b162299146463"><a name="b162299146463"></a><a name="b162299146463"></a>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</strong>：待执行检索的多个index。</p>
<p id="p18254194711350"><a name="p18254194711350"></a><a name="p18254194711350"></a><strong id="b7965111993515"><a name="b7965111993515"></a><a name="b7965111993515"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a></p>
<p id="p12897133913147"><a name="p12897133913147"></a><a name="p12897133913147"></a><strong id="b1891241644615"><a name="b1891241644615"></a><a name="b1891241644615"></a>size_t n</strong>：查询的特征向量的条数。</p>
<p id="p19897839151414"><a name="p19897839151414"></a><a name="p19897839151414"></a><strong id="b88731618124611"><a name="b88731618124611"></a><a name="b88731618124611"></a>std::vector&lt;float&gt;&amp; queryData</strong>：特征向量数据。</p>
<p id="p16898193912142"><a name="p16898193912142"></a><a name="p16898193912142"></a><strong id="b1359314204469"><a name="b1359314204469"></a><a name="b1359314204469"></a>int topK</strong>：需要返回的最相似的结果个数。</p>
<p id="p10898143919142"><a name="p10898143919142"></a><a name="p10898143919142"></a><strong id="b1302192264616"><a name="b1302192264616"></a><a name="b1302192264616"></a>bool merge</strong>：是否需要合并多个Index上执行检索的结果</p>
</td>
</tr>
<tr id="row1089812398143"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p589811390149"><a name="p589811390149"></a><a name="p589811390149"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1489810392146"><a name="p1489810392146"></a><a name="p1489810392146"></a><strong id="b53223004615"><a name="b53223004615"></a><a name="b53223004615"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前<span class="parmvalue" id="parmvalue12953175064519"><a name="parmvalue12953175064519"></a><a name="parmvalue12953175064519"></a>“topK”</span>个向量间的距离值。</p>
<p id="p188981439191411"><a name="p188981439191411"></a><a name="p188981439191411"></a><strong id="b167033323467"><a name="b167033323467"></a><a name="b167033323467"></a>std::vector&lt;int64_t&gt;&amp; labels</strong>：查询的距离最近的前<span class="parmvalue" id="parmvalue1478935414514"><a name="parmvalue1478935414514"></a><a name="parmvalue1478935414514"></a>“topK”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row78981339171420"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p189823920145"><a name="p189823920145"></a><a name="p189823920145"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row158981939121418"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1898939191417"><a name="p1898939191417"></a><a name="p1898939191417"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul125711434468"></a><a name="ul125711434468"></a><ul id="ul125711434468"><li>n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。</li><li>queryData：长度应该大于等于n * dim。</li><li>topK∈(0, 4096]。</li><li>dists、labels满足：<a name="ul1038283013138"></a><a name="ul1038283013138"></a><ul id="ul1038283013138"><li>当merge = true，长度应该大于等于n * topK。</li><li>当merge = false，长度应该大于等于indexes.size() * n * topK。</li></ul>
</li><li><span class="parmvalue" id="parmvalue1845654104415"><a name="parmvalue1845654104415"></a><a name="parmvalue1845654104415"></a>“indexes”</span>需满足：0 &lt; indexes.size() ≤ 150</li></ul>
</td>
</tr>
</tbody>
</table>

## MultiSearchWithMask接口<a name="ZH-CN_TOPIC_0000002008232700"></a>

<a name="table141672058131413"></a>
<table><tbody><tr id="row3203105801417"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p8203185851416"><a name="p8203185851416"></a><a name="p8203185851416"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p102034583149"><a name="p102034583149"></a><a name="p102034583149"></a>APP_ERROR MultiSearchWithMask(std::vector&lt;AscendIndexVStar*&gt;&amp; indexes, const AscendIndexSearchParams&amp; params, const std::vector&lt;uint8_t&gt;&amp; mask, bool merge);</p>
</td>
</tr>
<tr id="row82033582147"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p13203175821419"><a name="p13203175821419"></a><a name="p13203175821419"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2020313580140"><a name="p2020313580140"></a><a name="p2020313580140"></a>实现从多个AscendIndexVStar库执行特征向量查询的接口，根据输入的特征向量返回最相似的topK条特征距离及ID。提供基于mask掩码决定底库是否参与距离计算的功能。mask为0、1比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<span class="parmvalue" id="parmvalue66918584811"><a name="parmvalue66918584811"></a><a name="parmvalue66918584811"></a>“0”</span>表示不参与，<span class="parmvalue" id="parmvalue9775157184811"><a name="parmvalue9775157184811"></a><a name="parmvalue9775157184811"></a>“1”</span>表示参与。</p>
</td>
</tr>
<tr id="row42036589141"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5204185841415"><a name="p5204185841415"></a><a name="p5204185841415"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p112711315432"><a name="p112711315432"></a><a name="p112711315432"></a><strong id="b63275474819"><a name="b63275474819"></a><a name="b63275474819"></a>std::vector&lt;AscendIndexVStar*&gt;&amp; indexes</strong>：待执行检索的多个index。</p>
<p id="p1112710311432"><a name="p1112710311432"></a><a name="p1112710311432"></a><strong id="b14830144543418"><a name="b14830144543418"></a><a name="b14830144543418"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams</a>。</p>
<p id="p16127831204313"><a name="p16127831204313"></a><a name="p16127831204313"></a><strong id="b15861755124810"><a name="b15861755124810"></a><a name="b15861755124810"></a>size_t n</strong>：查询的特征向量的条数。</p>
<p id="p9128113114439"><a name="p9128113114439"></a><a name="p9128113114439"></a><strong id="b1481465712489"><a name="b1481465712489"></a><a name="b1481465712489"></a>std::vector&lt;float&gt;&amp; queryData</strong>：特征向量数据。</p>
<p id="p1712810313436"><a name="p1712810313436"></a><a name="p1712810313436"></a><strong id="b1972417591486"><a name="b1972417591486"></a><a name="b1972417591486"></a>int topK</strong>：需要返回的最相似的结果个数。</p>
<p id="p1128193117439"><a name="p1128193117439"></a><a name="p1128193117439"></a><strong id="b627242144916"><a name="b627242144916"></a><a name="b627242144916"></a>const std::vector&lt;uint8_t&gt;&amp; mask</strong>：特征底库掩码。</p>
<p id="p6128173124319"><a name="p6128173124319"></a><a name="p6128173124319"></a><strong id="b8460553498"><a name="b8460553498"></a><a name="b8460553498"></a>bool merge</strong>：是否需要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row220475819147"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p22041058121411"><a name="p22041058121411"></a><a name="p22041058121411"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p3204125814143"><a name="p3204125814143"></a><a name="p3204125814143"></a><strong id="b278613105497"><a name="b278613105497"></a><a name="b278613105497"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前<span class="parmvalue" id="parmvalue12953175064519"><a name="parmvalue12953175064519"></a><a name="parmvalue12953175064519"></a>“topK”</span>个向量间的距离值。</p>
<p id="p6204175812144"><a name="p6204175812144"></a><a name="p6204175812144"></a><strong id="b916151319499"><a name="b916151319499"></a><a name="b916151319499"></a>std::vector&lt;int64_t&gt;&amp; labels</strong>：查询的距离最近的前<span class="parmvalue" id="parmvalue65141641114811"><a name="parmvalue65141641114811"></a><a name="parmvalue65141641114811"></a>“topK”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row3204165861414"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p19204758191410"><a name="p19204758191410"></a><a name="p19204758191410"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1420455831420"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p32041758151419"><a name="p32041758151419"></a><a name="p32041758151419"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1548317268494"></a><a name="ul1548317268494"></a><ul id="ul1548317268494"><li>n∈(0，10000]，需保证n * dim * sizeof(float)小于卡的剩余内存，否则可能内存不足导致检索失败。</li><li>queryData：长度应该大于等于n*dim。</li><li>topK∈(0, 4096]。</li><li>dists、labels满足：<a name="ul1038283013138"></a><a name="ul1038283013138"></a><ul id="ul1038283013138"><li>当merge = true，长度应该大于等于n * topK。</li><li>当merge = false，长度应该大于等于indexes.size() * n * topK。</li></ul>
</li><li>mask：长度应该大于等于n * ceil(ntotal_max/8)，其中ntotal_max为底库特征数量，为所有Index中最大的底库数量值。</li><li><span class="parmvalue" id="parmvalue1845654104415"><a name="parmvalue1845654104415"></a><a name="parmvalue1845654104415"></a>“indexes”</span>需满足：0 &lt; indexes.size() ≤ 150</li></ul>
</td>
</tr>
</tbody>
</table>

## SetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044351693"></a>

<a name="table4215111781514"></a>
<table><tbody><tr id="row424541719154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16245917171512"><a name="p16245917171512"></a><a name="p16245917171512"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14245121712155"><a name="p14245121712155"></a><a name="p14245121712155"></a>APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams&amp; params);</p>
</td>
</tr>
<tr id="row2245317151518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p4245201713153"><a name="p4245201713153"></a><a name="p4245201713153"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p924513177152"><a name="p924513177152"></a><a name="p924513177152"></a>设置AscendIndexVstar实例检索时的超参。</p>
</td>
</tr>
<tr id="row12451917191510"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1024561716151"><a name="p1024561716151"></a><a name="p1024561716151"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14891331164012"><a name="p14891331164012"></a><a name="p14891331164012"></a><strong id="b92771420133413"><a name="b92771420133413"></a><a name="b92771420133413"></a>const AscendIndexVstarHyperParams&amp; params</strong>：检索时超参，具体请见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarhyperparams接口">AscendIndexVstarHyperParams</a>。</p>
</td>
</tr>
<tr id="row202451617111514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p3245141761510"><a name="p3245141761510"></a><a name="p3245141761510"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14245181771511"><a name="p14245181771511"></a><a name="p14245181771511"></a>无</p>
</td>
</tr>
<tr id="row1224521711514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2245171717154"><a name="p2245171717154"></a><a name="p2245171717154"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row724551720159"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p124671761512"><a name="p124671761512"></a><a name="p124671761512"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul10599750124914"></a><a name="ul10599750124914"></a><ul id="ul10599750124914"><li>nProbeL1∈(16，nListL1], nProbeL1 % 8 == 0</li><li>nProbeL2∈(16, nProbeL1 * nList2], nProbeL2 % 8 == 0</li><li>l3SegmentNum∈(100,5000], l3SegmentNum % 8 == 0</li></ul>
</td>
</tr>
</tbody>
</table>

## GetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044510709"></a>

<a name="table5860202961515"></a>
<table><tbody><tr id="row8883729101511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1588382911154"><a name="p1588382911154"></a><a name="p1588382911154"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14883202991513"><a name="p14883202991513"></a><a name="p14883202991513"></a>APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams&amp; params) const;</p>
</td>
</tr>
<tr id="row148831295154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p16883102915155"><a name="p16883102915155"></a><a name="p16883102915155"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1088320294158"><a name="p1088320294158"></a><a name="p1088320294158"></a>获取向量检索时的超参。</p>
</td>
</tr>
<tr id="row148831429111512"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17883102911153"><a name="p17883102911153"></a><a name="p17883102911153"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p188831429111515"><a name="p188831429111515"></a><a name="p188831429111515"></a>无</p>
</td>
</tr>
<tr id="row10883102917151"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p10883152941520"><a name="p10883152941520"></a><a name="p10883152941520"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111384864011"><a name="p111384864011"></a><a name="p111384864011"></a><strong id="b5901151112341"><a name="b5901151112341"></a><a name="b5901151112341"></a>AscendIndexVstarHyperParams&amp; params</strong>：检索时超参，具体请见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarhyperparams接口">AscendIndexVstarHyperParams</a>。</p>
</td>
</tr>
<tr id="row4883129161515"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1088312901510"><a name="p1088312901510"></a><a name="p1088312901510"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row108835291150"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1188472961518"><a name="p1188472961518"></a><a name="p1188472961518"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p08843296151"><a name="p08843296151"></a><a name="p08843296151"></a>无</p>
</td>
</tr>
</tbody>
</table>

## GetDim接口<a name="ZH-CN_TOPIC_0000002008390992"></a>

<a name="table6661184351519"></a>
<table><tbody><tr id="row4685124316154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p068584371516"><a name="p068584371516"></a><a name="p068584371516"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p46851543131512"><a name="p46851543131512"></a><a name="p46851543131512"></a>APP_ERROR GetDim(int&amp; dim) const;</p>
</td>
</tr>
<tr id="row13685134318155"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p76851743181516"><a name="p76851743181516"></a><a name="p76851743181516"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p4685184313152"><a name="p4685184313152"></a><a name="p4685184313152"></a>获取初始化索引时的维度。</p>
</td>
</tr>
<tr id="row196850436159"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p068524312151"><a name="p068524312151"></a><a name="p068524312151"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p156857438155"><a name="p156857438155"></a><a name="p156857438155"></a>无</p>
</td>
</tr>
<tr id="row16685134301515"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18685443181515"><a name="p18685443181515"></a><a name="p18685443181515"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2068511434150"><a name="p2068511434150"></a><a name="p2068511434150"></a><strong id="b1354683012163"><a name="b1354683012163"></a><a name="b1354683012163"></a>int&amp; dim</strong>：Index的维度。</p>
</td>
</tr>
<tr id="row106854435154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1668564331513"><a name="p1668564331513"></a><a name="p1668564331513"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row176851343151518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15685443171519"><a name="p15685443171519"></a><a name="p15685443171519"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p7685134319154"><a name="p7685134319154"></a><a name="p7685134319154"></a>无</p>
</td>
</tr>
</tbody>
</table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000002008232704"></a>

<a name="table1919613597154"></a>
<table><tbody><tr id="row152181659101514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p152181359121512"><a name="p152181359121512"></a><a name="p152181359121512"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122181759181514"><a name="p122181759181514"></a><a name="p122181759181514"></a>APP_ERROR GetNTotal(uint64_t&amp; ntotal) const;</p>
</td>
</tr>
<tr id="row1321835981514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1421818591159"><a name="p1421818591159"></a><a name="p1421818591159"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p42183591154"><a name="p42183591154"></a><a name="p42183591154"></a>获取当前索引的底库数量。</p>
</td>
</tr>
<tr id="row1221925941514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p421995913152"><a name="p421995913152"></a><a name="p421995913152"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1221955941514"><a name="p1221955941514"></a><a name="p1221955941514"></a>无</p>
</td>
</tr>
<tr id="row12219135912159"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p202191259111518"><a name="p202191259111518"></a><a name="p202191259111518"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p421995917155"><a name="p421995917155"></a><a name="p421995917155"></a><strong id="b12169202918182"><a name="b12169202918182"></a><a name="b12169202918182"></a>uint64_t&amp; ntotal</strong>：当前Index的底库总向量条数。</p>
</td>
</tr>
<tr id="row62197595151"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1219145911510"><a name="p1219145911510"></a><a name="p1219145911510"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row102191259101511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p8219195910153"><a name="p8219195910153"></a><a name="p8219195910153"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p92192594157"><a name="p92192594157"></a><a name="p92192594157"></a>无</p>
</td>
</tr>
</tbody>
</table>

## Reset接口<a name="ZH-CN_TOPIC_0000002044351697"></a>

<a name="table19794117167"></a>
<table><tbody><tr id="row819122160"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p13118126162"><a name="p13118126162"></a><a name="p13118126162"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p9121231610"><a name="p9121231610"></a><a name="p9121231610"></a>APP_ERROR Reset();</p>
</td>
</tr>
<tr id="row1611212181614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p61712161611"><a name="p61712161611"></a><a name="p61712161611"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p11111126162"><a name="p11111126162"></a><a name="p11111126162"></a>重置索引接口，清除保存的索引数据。</p>
</td>
</tr>
<tr id="row1821612141610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p162191214162"><a name="p162191214162"></a><a name="p162191214162"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p82612141612"><a name="p82612141612"></a><a name="p82612141612"></a>无</p>
</td>
</tr>
<tr id="row17211211166"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p142111291620"><a name="p142111291620"></a><a name="p142111291620"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p152191281615"><a name="p152191281615"></a><a name="p152191281615"></a>无</p>
</td>
</tr>
<tr id="row1021412171617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1821312151613"><a name="p1821312151613"></a><a name="p1821312151613"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1241231617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p132151221620"><a name="p132151221620"></a><a name="p132151221620"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182712191614"><a name="p182712191614"></a><a name="p182712191614"></a>重置索引后，会保留用户初始化索引时输入的参数。</p>
</td>
</tr>
</tbody>
</table>

## operator= 接口<a name="ZH-CN_TOPIC_0000002008390996"></a>

<a name="table3792193711620"></a>
<table><tbody><tr id="row1681723717164"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p38176375169"><a name="p38176375169"></a><a name="p38176375169"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p98178379165"><a name="p98178379165"></a><a name="p98178379165"></a>AscendIndexVStar&amp; operator=(const AscendIndexVStar&amp;) = delete;</p>
</td>
</tr>
<tr id="row981720372161"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p13817337101618"><a name="p13817337101618"></a><a name="p13817337101618"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8817103711167"><a name="p8817103711167"></a><a name="p8817103711167"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row481715372169"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p208179376160"><a name="p208179376160"></a><a name="p208179376160"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8817103714162"><a name="p8817103714162"></a><a name="p8817103714162"></a><strong id="b0300123915910"><a name="b0300123915910"></a><a name="b0300123915910"></a>const AscendIndexVStar&amp;</strong>：AscendIndexVStar对象。</p>
</td>
</tr>
<tr id="row881763741614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p481715374162"><a name="p481715374162"></a><a name="p481715374162"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p10817143761611"><a name="p10817143761611"></a><a name="p10817143761611"></a>无</p>
</td>
</tr>
<tr id="row581716374163"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p68171637131610"><a name="p68171637131610"></a><a name="p68171637131610"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p18817113771617"><a name="p18817113771617"></a><a name="p18817113771617"></a>无</p>
</td>
</tr>
<tr id="row1881763716166"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p98171437151619"><a name="p98171437151619"></a><a name="p98171437151619"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p18817237131611"><a name="p18817237131611"></a><a name="p18817237131611"></a>无</p>
</td>
</tr>
</tbody>
</table>
