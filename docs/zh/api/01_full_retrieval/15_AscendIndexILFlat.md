# AscendIndexILFlat<a name="ZH-CN_TOPIC_0000002514896041"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002482656058"></a>

AscendIndexILFlat为ILFlat标准态场景，需要使用Init指定对应资源的初始化，初始化完成之后会申请一段完整空间用于存储底库。在使用完成之后，需要调用Finalize接口对资源进行释放。

AscendIndexILFlat仅支持使用<term>Atlas 推理系列产品</term>，在标准态部署方式下的向量内积距离类型。AscendIndexILFlat在使用时依赖Flat和AICPU算子，具体请参见[Flat](../../05_user_guide.md#flat)和[AICPU](../../05_user_guide.md#aicpu)。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AddFeatures接口<a name="ZH-CN_TOPIC_0000002514776041"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR AddFeatures(int n, const float *features);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>向特征库追加“n”个特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b645815911297"><a name="b645815911297"></a><a name="b645815911297"></a>int n</strong>：插入特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b1053019911245"><a name="b1053019911245"></a><a name="b1053019911245"></a>const float *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1835741212302"><a name="b1835741212302"></a><a name="b1835741212302"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul10674191294110"></a><a name="ul10674191294110"></a><ul id="ul10674191294110"><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname3816191310019"><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a>“features”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table392463914228"></a>
<table><tbody><tr id="row17924183911228"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p159241739182219"><a name="p159241739182219"></a><a name="p159241739182219"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19924039112212"><a name="p19924039112212"></a><a name="p19924039112212"></a>APP_ERROR AddFeatures(int n, const float16_t *features);</p>
</td>
</tr>
<tr id="row13924439172216"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p792417393229"><a name="p792417393229"></a><a name="p792417393229"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1792413962211"><a name="p1792413962211"></a><a name="p1792413962211"></a>向特征库追加“n”个特征向量。</p>
</td>
</tr>
<tr id="row792418398229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1692410397224"><a name="p1692410397224"></a><a name="p1692410397224"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1192416399222"><a name="p1192416399222"></a><a name="p1192416399222"></a><strong id="b1992493913229"><a name="b1992493913229"></a><a name="b1992493913229"></a>int n</strong>：插入特征向量数目。</p>
<p id="p292416399222"><a name="p292416399222"></a><a name="p292416399222"></a><strong id="b1453142822318"><a name="b1453142822318"></a><a name="b1453142822318"></a>const float16_t *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row5924163962213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p0924439162217"><a name="p0924439162217"></a><a name="p0924439162217"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p19924103932211"><a name="p19924103932211"></a><a name="p19924103932211"></a>无</p>
</td>
</tr>
<tr id="row14924163932212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p18924143910228"><a name="p18924143910228"></a><a name="p18924143910228"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p179241397228"><a name="p179241397228"></a><a name="p179241397228"></a><strong id="b892483952211"><a name="b892483952211"></a><a name="b892483952211"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row159242391222"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2924153992215"><a name="p2924153992215"></a><a name="p2924153992215"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul16924339172210"></a><a name="ul16924339172210"></a><ul id="ul16924339172210"><li><strong id="b1592433919225"><a name="b1592433919225"></a><a name="b1592433919225"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname109242395227"><a name="varname109242395227"></a><a name="varname109242395227"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname10924153952217"><a name="parmname10924153952217"></a><a name="parmname10924153952217"></a>“features”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## AscendIndexILFlat接口<a name="ZH-CN_TOPIC_0000002516511133"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexILFlat();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexILFlat的构造函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table161511529133912"></a>
<table><tbody><tr id="row1615110293394"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2151429113910"><a name="p2151429113910"></a><a name="p2151429113910"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16794134614447"><a name="p16794134614447"></a><a name="p16794134614447"></a>AscendIndexILFlat(const AscendIndexILFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row51517295398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p21514294391"><a name="p21514294391"></a><a name="p21514294391"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015122918399"><a name="p2015122918399"></a><a name="p2015122918399"></a>声明此Index拷贝函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row815120292398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7151122933917"><a name="p7151122933917"></a><a name="p7151122933917"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b2450181274519"><a name="b2450181274519"></a><a name="b2450181274519"></a>const AscendIndexILFlat&amp;</strong>：AscendIndexILFlat对象。</p>
</td>
</tr>
<tr id="row18151172918399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p615182993916"><a name="p615182993916"></a><a name="p615182993916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8151329143914"><a name="p8151329143914"></a><a name="p8151329143914"></a>无</p>
</td>
</tr>
<tr id="row171511295399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p17151192917392"><a name="p17151192917392"></a><a name="p17151192917392"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16151122914394"><a name="p16151122914394"></a><a name="p16151122914394"></a>无</p>
</td>
</tr>
<tr id="row12151829153910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p615192973914"><a name="p615192973914"></a><a name="p615192973914"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15151929163918"><a name="p15151929163918"></a><a name="p15151929163918"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table62621513124018"></a>
<table><tbody><tr id="row726218134408"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1726212134400"><a name="p1726212134400"></a><a name="p1726212134400"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexILFlat();</p>
</td>
</tr>
<tr id="row1926221314401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1926218134408"><a name="p1926218134408"></a><a name="p1926218134408"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82621213184020"><a name="p82621213184020"></a><a name="p82621213184020"></a>AscendIndexILFlat的析构函数。</p>
</td>
</tr>
<tr id="row15262213104015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p826221314402"><a name="p826221314402"></a><a name="p826221314402"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row1726271324017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p5262213154014"><a name="p5262213154014"></a><a name="p5262213154014"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16262131311400"><a name="p16262131311400"></a><a name="p16262131311400"></a>无</p>
</td>
</tr>
<tr id="row0262121324020"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p8262191319409"><a name="p8262191319409"></a><a name="p8262191319409"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p726201319407"><a name="p726201319407"></a><a name="p726201319407"></a>无</p>
</td>
</tr>
<tr id="row526241324016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p526201310404"><a name="p526201310404"></a><a name="p526201310404"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15262111319403"><a name="p15262111319403"></a><a name="p15262111319403"></a>无</p>
</td>
</tr>
</tbody>
</table>

## ComputeDistance接口<a name="ZH-CN_TOPIC_0000002482736032"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19771939144811"><a name="p19771939144811"></a><a name="p19771939144811"></a>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname11281111175619"><a name="parmname11281111175619"></a><a name="parmname11281111175619"></a>“n”</span>条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出经过映射后的距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b75131329183314"><a name="b75131329183314"></a><a name="b75131329183314"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b6862131183312"><a name="b6862131183312"></a><a name="b6862131183312"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b1041123433313"><a name="b1041123433313"></a><a name="b1041123433313"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue354895917522"><a name="parmvalue354895917522"></a><a name="parmvalue354895917522"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b376183643316"><a name="b376183643316"></a><a name="b376183643316"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname834710403338"><a name="parmname834710403338"></a><a name="parmname834710403338"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue16873151054312"><a name="parmvalue16873151054312"></a><a name="parmvalue16873151054312"></a>“48”</span>，即<span class="parmname" id="parmname135806266430"><a name="parmname135806266430"></a><a name="parmname135806266430"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1750033215518"><a name="p1750033215518"></a><a name="p1750033215518"></a><strong id="b668194911337"><a name="b668194911337"></a><a name="b668194911337"></a>float *distances</strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（<span class="parmname" id="parmname10121121717561"><a name="parmname10121121717561"></a><a name="parmname10121121717561"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname13631434155615"><a name="varname13631434155615"></a><a name="varname13631434155615"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname0810103815562"><a name="parmname0810103815562"></a><a name="parmname0810103815562"></a>“ntotal”</span>对16补齐）。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul167968714447"></a><a name="ul167968714447"></a><ul id="ul167968714447"><li><strong id="b1141410121444"><a name="b1141410121444"></a><a name="b1141410121444"></a>n</strong>：合理的n值应该在(0, <i><span class="varname" id="varname1835520162442"><a name="varname1835520162442"></a><a name="varname1835520162442"></a>capacity</span></i>]之间。</li><li><strong id="b429253634917"><a name="b429253634917"></a><a name="b429253634917"></a>distances</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname1598134614491"><a name="parmname1598134614491"></a><a name="parmname1598134614491"></a>“ntotalPad”</span>为(<i><span class="varname" id="varname1654664914915"><a name="varname1654664914915"></a><a name="varname1654664914915"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname145712539496"><a name="parmname145712539496"></a><a name="parmname145712539496"></a>“ntotal”</span>对16补齐的结果，每个query的有效比对距离存储在前<span class="parmname" id="parmname15322121501"><a name="parmname15322121501"></a><a name="parmname15322121501"></a>“ntotal”</span>的空间，补齐部分数据没有实际意义）。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table17574555124816"></a>
<table><tbody><tr id="row757435594819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p155742557480"><a name="p155742557480"></a><a name="p155742557480"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p0574255104813"><a name="p0574255104813"></a><a name="p0574255104813"></a>APP_ERROR ComputeDistance(int n, const float *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row14574135514811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p205741955124812"><a name="p205741955124812"></a><a name="p205741955124812"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p45741955174814"><a name="p45741955174814"></a><a name="p45741955174814"></a>查询<span class="parmname" id="parmname18574205574813"><a name="parmname18574205574813"></a><a name="parmname18574205574813"></a>“n”</span>条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出经过映射后的距离。</p>
</td>
</tr>
<tr id="row85751555194813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p35754551488"><a name="p35754551488"></a><a name="p35754551488"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1757535519484"><a name="p1757535519484"></a><a name="p1757535519484"></a><strong id="b257555544811"><a name="b257555544811"></a><a name="b257555544811"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p1575115520489"><a name="p1575115520489"></a><a name="p1575115520489"></a><strong id="b158293141686"><a name="b158293141686"></a><a name="b158293141686"></a>const float *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p1957512553485"><a name="p1957512553485"></a><a name="p1957512553485"></a><strong id="b7575155164810"><a name="b7575155164810"></a><a name="b7575155164810"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue15575255144814"><a name="parmvalue15575255144814"></a><a name="parmvalue15575255144814"></a>“10000”</span>。</p>
<p id="p20575185520489"><a name="p20575185520489"></a><a name="p20575185520489"></a><strong id="b1657585510484"><a name="b1657585510484"></a><a name="b1657585510484"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname257575517485"><a name="parmname257575517485"></a><a name="parmname257575517485"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue19575115554814"><a name="parmvalue19575115554814"></a><a name="parmvalue19575115554814"></a>“48”</span>，即<span class="parmname" id="parmname55751655194811"><a name="parmname55751655194811"></a><a name="parmname55751655194811"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row1557595510487"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1257545518481"><a name="p1257545518481"></a><a name="p1257545518481"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p115751555184810"><a name="p115751555184810"></a><a name="p115751555184810"></a><strong id="b13575195584812"><a name="b13575195584812"></a><a name="b13575195584812"></a>float *distances</strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（<span class="parmname" id="parmname1557575514485"><a name="parmname1557575514485"></a><a name="parmname1557575514485"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname15575125515488"><a name="varname15575125515488"></a><a name="varname15575125515488"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname657555514488"><a name="parmname657555514488"></a><a name="parmname657555514488"></a>“ntotal”</span>对16补齐）。</p>
</td>
</tr>
<tr id="row7575175554817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1757575514814"><a name="p1757575514814"></a><a name="p1757575514814"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p125758557485"><a name="p125758557485"></a><a name="p125758557485"></a><strong id="b19575165519489"><a name="b19575165519489"></a><a name="b19575165519489"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row9575755204810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12575185574818"><a name="p12575185574818"></a><a name="p12575185574818"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7575125514817"></a><a name="ul7575125514817"></a><ul id="ul7575125514817"><li><strong id="b18575155510484"><a name="b18575155510484"></a><a name="b18575155510484"></a>n</strong>：合理的n值应该在(0, <i><span class="varname" id="varname1557595554819"><a name="varname1557595554819"></a><a name="varname1557595554819"></a>capacity</span></i>]之间。</li><li><strong id="b75751155114811"><a name="b75751155114811"></a><a name="b75751155114811"></a>distances</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname857575524810"><a name="parmname857575524810"></a><a name="parmname857575524810"></a>“ntotalPad”</span>为(<i><span class="varname" id="varname11575355194813"><a name="varname11575355194813"></a><a name="varname11575355194813"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname3575185514489"><a name="parmname3575185514489"></a><a name="parmname3575185514489"></a>“ntotal”</span>对16补齐的结果，每个query的有效比对距离存储在前<span class="parmname" id="parmname12575175514482"><a name="parmname12575175514482"></a><a name="parmname12575175514482"></a>“ntotal”</span>的空间，补齐部分数据没有实际意义）。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="parmname457605554818"><a name="parmname457605554818"></a><a name="parmname457605554818"></a>“queries”</span>和<span class="parmname" id="parmname185761655104810"><a name="parmname185761655104810"></a><a name="parmname185761655104810"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000002514896043"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1196718166412"><a name="p1196718166412"></a><a name="p1196718166412"></a>APP_ERROR ComputeDistanceByIdx(int n, const float *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>ComputeDistance计算待查询向量与所有底库向量的距离，而ComputeDistanceByIdx接口只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1178514265435"><a name="b1178514265435"></a><a name="b1178514265435"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b811819197314"><a name="b811819197314"></a><a name="b811819197314"></a>const float *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname1441759144217"><a name="parmname1441759144217"></a><a name="parmname1441759144217"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a>const int *num</strong>： 给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b79815523517"><a name="b79815523517"></a><a name="b79815523517"></a>const idx_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname2711154912437"><a name="parmname2711154912437"></a><a name="parmname2711154912437"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname742124364316"><a name="parmname742124364316"></a><a name="parmname742124364316"></a>“indices”</span>长度为n * max(num)。输入在host，indices为host指针；输入在device，indices为device指针。</p>
<p id="p13553919567"><a name="p13553919567"></a><a name="p13553919567"></a><strong id="b184841525865"><a name="b184841525865"></a><a name="b184841525865"></a>MEMORY_TYPE memoryType</strong>：输入输出存放位置策略，默认为MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，可选策略如下：</p>
<a name="ul125365550127"></a><a name="ul125365550127"></a><ul id="ul125365550127"><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST：输入在host，输出在host。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE：输入在device，输出在device。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST：输入在device，输出在host。</li><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE：输入在host，输出在device。</li></ul>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b156251035184313"><a name="b156251035184313"></a><a name="b156251035184313"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b25863377438"><a name="b25863377438"></a><a name="b25863377438"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname136035014443"><a name="parmname136035014443"></a><a name="parmname136035014443"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue2069717180212"><a name="parmvalue2069717180212"></a><a name="parmvalue2069717180212"></a>“48”</span>，即<span class="parmname" id="parmname1997224217"><a name="parmname1997224217"></a><a name="parmname1997224217"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p057182814222"><a name="p057182814222"></a><a name="p057182814222"></a><strong id="b0194557446"><a name="b0194557446"></a><a name="b0194557446"></a>float *distances</strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大<span class="parmname" id="parmname658971354417"><a name="parmname658971354417"></a><a name="parmname658971354417"></a>“num”</span>补齐空间占用，空间长度为n * max(num)。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b140412864414"><a name="b140412864414"></a><a name="b140412864414"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname82723561324"><a name="varname82723561324"></a><a name="varname82723561324"></a>capacity</span></i>]之间。</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0, ntotal]之间。</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname343119418149"><a name="parmname343119418149"></a><a name="parmname343119418149"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li><li>选择memoryType存放策略时，<span class="parmname" id="parmname17419164171417"><a name="parmname17419164171417"></a><a name="parmname17419164171417"></a>“queries”</span>、<span class="parmname" id="parmname13419174191415"><a name="parmname13419174191415"></a><a name="parmname13419174191415"></a>“distances”</span>需要为对应位置指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table93703718308"></a>
<table><tbody><tr id="row20370173302"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p143716720307"><a name="p143716720307"></a><a name="p143716720307"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1037115763020"><a name="p1037115763020"></a><a name="p1037115763020"></a>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row103719723013"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p0371207153017"><a name="p0371207153017"></a><a name="p0371207153017"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p53715711306"><a name="p53715711306"></a><a name="p53715711306"></a>ComputeDistance计算待查询向量与所有底库向量的距离，而ComputeDistanceByIdx接口只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row123716710302"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p16371147193013"><a name="p16371147193013"></a><a name="p16371147193013"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p23711377308"><a name="p23711377308"></a><a name="p23711377308"></a><strong id="b83711372301"><a name="b83711372301"></a><a name="b83711372301"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p173714714309"><a name="p173714714309"></a><a name="p173714714309"></a><strong id="b1926533153016"><a name="b1926533153016"></a><a name="b1926533153016"></a>const float16_t *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname8371207113014"><a name="parmname8371207113014"></a><a name="parmname8371207113014"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1237110743020"><a name="p1237110743020"></a><a name="p1237110743020"></a><strong id="b133711723012"><a name="b133711723012"></a><a name="b133711723012"></a>const int *num</strong>： 给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p037167153019"><a name="p037167153019"></a><a name="p037167153019"></a><strong id="b12371167113014"><a name="b12371167113014"></a><a name="b12371167113014"></a>const idx_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname33718710304"><a name="parmname33718710304"></a><a name="parmname33718710304"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname113713793017"><a name="parmname113713793017"></a><a name="parmname113713793017"></a>“indices”</span>长度为n * max(num)。输入在host，indices为host指针；输入在device，indices为device指针。</p>
<p id="p123717711303"><a name="p123717711303"></a><a name="p123717711303"></a><strong id="b1137116717301"><a name="b1137116717301"></a><a name="b1137116717301"></a>MEMORY_TYPE memoryType</strong>：输入输出存放位置策略，默认为MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，可选策略如下：</p>
<a name="ul183711373302"></a><a name="ul183711373302"></a><ul id="ul183711373302"><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST：输入在host，输出在host。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE：输入在device，输出在device。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST：输入在device，输出在host。</li><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE：输入在host，输出在device。</li></ul>
<p id="p173715717301"><a name="p173715717301"></a><a name="p173715717301"></a><strong id="b18371874303"><a name="b18371874303"></a><a name="b18371874303"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue2371137173014"><a name="parmvalue2371137173014"></a><a name="parmvalue2371137173014"></a>“10000”</span>。</p>
<p id="p1637117793019"><a name="p1637117793019"></a><a name="p1637117793019"></a><strong id="b153711719302"><a name="b153711719302"></a><a name="b153711719302"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname123719713013"><a name="parmname123719713013"></a><a name="parmname123719713013"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue637117153014"><a name="parmvalue637117153014"></a><a name="parmvalue637117153014"></a>“48”</span>，即<span class="parmname" id="parmname193714743018"><a name="parmname193714743018"></a><a name="parmname193714743018"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row837117183012"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p1737119715302"><a name="p1737119715302"></a><a name="p1737119715302"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p4371378300"><a name="p4371378300"></a><a name="p4371378300"></a><strong id="b137117716306"><a name="b137117716306"></a><a name="b137117716306"></a>float *distances</strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大<span class="parmname" id="parmname73715718306"><a name="parmname73715718306"></a><a name="parmname73715718306"></a>“num”</span>补齐空间占用，空间长度为n * max(num)。</p>
</td>
</tr>
<tr id="row037177153010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p183711274309"><a name="p183711274309"></a><a name="p183711274309"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p63711776304"><a name="p63711776304"></a><a name="p63711776304"></a><strong id="b1337118763012"><a name="b1337118763012"></a><a name="b1337118763012"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row193711676307"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p137114753010"><a name="p137114753010"></a><a name="p137114753010"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul13371476304"></a><a name="ul13371476304"></a><ul id="ul13371476304"><li><strong id="b737127163015"><a name="b737127163015"></a><a name="b737127163015"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname737137183015"><a name="varname737137183015"></a><a name="varname737137183015"></a>capacity</span></i>]之间。</li><li><strong id="b103719712303"><a name="b103719712303"></a><a name="b103719712303"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0, ntotal]之间。</li><li><strong id="b33718723018"><a name="b33718723018"></a><a name="b33718723018"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname6371147113013"><a name="varname6371147113013"></a><a name="varname6371147113013"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul10372137123013"></a><a name="ul10372137123013"></a><ul id="ul10372137123013"><li><span class="parmname" id="parmname17372776306"><a name="parmname17372776306"></a><a name="parmname17372776306"></a>“indices”</span>、<span class="parmname" id="parmname437213773019"><a name="parmname437213773019"></a><a name="parmname437213773019"></a>“queries”</span>、<span class="parmname" id="parmname2037220753020"><a name="parmname2037220753020"></a><a name="parmname2037220753020"></a>“distances”</span>和<span class="parmname" id="parmname33721178301"><a name="parmname33721178301"></a><a name="parmname33721178301"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## Finalize接口<a name="ZH-CN_TOPIC_0000002482656060"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>void Finalize();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>释放特征库管理资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1570793612442"><a name="b1570793612442"></a><a name="b1570793612442"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>

## GetFeatures接口<a name="ZH-CN_TOPIC_0000002484074790"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeatures(int n, float *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname9635334135520"><a name="parmname9635334135520"></a><a name="parmname9635334135520"></a>“n”</span>条指定下标索引的特征向量。输出在host。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b38573713437"><a name="b38573713437"></a><a name="b38573713437"></a>float *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table018415716495"></a>
<table><tbody><tr id="row51841657124915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3184757144914"><a name="p3184757144914"></a><a name="p3184757144914"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16184957174915"><a name="p16184957174915"></a><a name="p16184957174915"></a>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row121844578499"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1184205717496"><a name="p1184205717496"></a><a name="p1184205717496"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6184457154911"><a name="p6184457154911"></a><a name="p6184457154911"></a>查询<span class="parmname" id="parmname518485711495"><a name="parmname518485711495"></a><a name="parmname518485711495"></a>“n”</span>条指定下标索引的特征向量。输出在host。</p>
</td>
</tr>
<tr id="row20184257154913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18184657114919"><a name="p18184657114919"></a><a name="p18184657114919"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1418413578493"><a name="p1418413578493"></a><a name="p1418413578493"></a><strong id="b16184115711494"><a name="b16184115711494"></a><a name="b16184115711494"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p11184757204918"><a name="p11184757204918"></a><a name="p11184757204918"></a><strong id="b718445714916"><a name="b718445714916"></a><a name="b718445714916"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row19184195714498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p3184175711496"><a name="p3184175711496"></a><a name="p3184175711496"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p12184657154920"><a name="p12184657154920"></a><a name="p12184657154920"></a><strong id="b12804115914544"><a name="b12804115914544"></a><a name="b12804115914544"></a>float16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row1918411573494"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p19184357194919"><a name="p19184357194919"></a><a name="p19184357194919"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p14184757184914"><a name="p14184757184914"></a><a name="p14184757184914"></a><strong id="b191841957184915"><a name="b191841957184915"></a><a name="b191841957184915"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row16184195774920"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p181841657104916"><a name="p181841657104916"></a><a name="p181841657104916"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul181842571494"></a><a name="ul181842571494"></a><ul id="ul181842571494"><li><strong id="b6184105764918"><a name="b6184105764918"></a><a name="b6184105764918"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname618445754917"><a name="varname618445754917"></a><a name="varname618445754917"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b111847574493"><a name="b111847574493"></a><a name="b111847574493"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname1918410571491"><a name="varname1918410571491"></a><a name="varname1918410571491"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname2018413570492"><a name="parmname2018413570492"></a><a name="parmname2018413570492"></a>“features”</span>和<span class="parmname" id="parmname818495794917"><a name="parmname818495794917"></a><a name="parmname818495794917"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## GetFeaturesOnDevice<a name="ZH-CN_TOPIC_0000002516516843"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeaturesOnDevice (int n, float16_t *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询“n”条指定下标索引的特征向量。输出在Device。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b885718291130"><a name="b885718291130"></a><a name="b885718291130"></a>float16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。Device侧指针。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table15312115612410"></a>
<table><tbody><tr id="row1831211561843"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2031217561042"><a name="p2031217561042"></a><a name="p2031217561042"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10312185617418"><a name="p10312185617418"></a><a name="p10312185617418"></a>APP_ERROR GetFeaturesOnDevice (int n, float *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row123121356046"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p531245612416"><a name="p531245612416"></a><a name="p531245612416"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p431217567418"><a name="p431217567418"></a><a name="p431217567418"></a>查询“n”条指定下标索引的特征向量。输出在Device。</p>
</td>
</tr>
<tr id="row531213561245"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p431225615416"><a name="p431225615416"></a><a name="p431225615416"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p63123561342"><a name="p63123561342"></a><a name="p63123561342"></a><strong id="b13312356848"><a name="b13312356848"></a><a name="b13312356848"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1731215620414"><a name="p1731215620414"></a><a name="p1731215620414"></a><strong id="b143129561748"><a name="b143129561748"></a><a name="b143129561748"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row53126562043"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1431212564417"><a name="p1431212564417"></a><a name="p1431212564417"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p193121656244"><a name="p193121656244"></a><a name="p193121656244"></a><strong id="b1679419391057"><a name="b1679419391057"></a><a name="b1679419391057"></a>float *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。Device侧指针。</p>
</td>
</tr>
<tr id="row10312056346"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p73124561444"><a name="p73124561444"></a><a name="p73124561444"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p931217561412"><a name="p931217561412"></a><a name="p931217561412"></a><strong id="b1931275616410"><a name="b1931275616410"></a><a name="b1931275616410"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row123127565418"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1931214561141"><a name="p1931214561141"></a><a name="p1931214561141"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4312115612410"></a><a name="ul4312115612410"></a><ul id="ul4312115612410"><li><strong id="b331255610416"><a name="b331255610416"></a><a name="b331255610416"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname1631245618414"><a name="varname1631245618414"></a><a name="varname1631245618414"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b2312185619410"><a name="b2312185619410"></a><a name="b2312185619410"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname1831295612411"><a name="varname1831295612411"></a><a name="varname1831295612411"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname83125560419"><a name="parmname83125560419"></a><a name="parmname83125560419"></a>“features”</span>和<span class="parmname" id="parmname18312125620410"><a name="parmname18312125620410"></a><a name="parmname18312125620410"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000002514776043"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p0336129171210"><a name="p0336129171210"></a><a name="p0336129171210"></a>int GetNTotal() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询当前特征库特征向量数目的理论最大值。如果插入特征向量indices连续，则ntotal等于特征向量数目。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a><strong id="b445021732816"><a name="b445021732816"></a><a name="b445021732816"></a>int ntotal</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a>int</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>

## Init接口<a name="ZH-CN_TOPIC_0000002482736034"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>AscendIndexILFlat的初始化函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1517311219268"><a name="b1517311219268"></a><a name="b1517311219268"></a>int dim</strong>：AscendIndexILFlat管理的特征向量的维度。</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8628752620"><a name="b8628752620"></a><a name="b8628752620"></a>int capacity</strong>：底库最大容量，接口会根据<span class="parmname" id="parmname16513113011414"><a name="parmname16513113011414"></a><a name="parmname16513113011414"></a>“capacity”</span>值申请capacity * dim * sizeof(fp16) 字节内存数据。</p>
<p id="p1450765311416"><a name="p1450765311416"></a><a name="p1450765311416"></a><strong id="b5404134231712"><a name="b5404134231712"></a><a name="b5404134231712"></a>faiss::MetricType metricType</strong>： 特征距离类别（向量内积、欧氏距离、余弦相似度）。</p>
<p id="p1291682015184"><a name="p1291682015184"></a><a name="p1291682015184"></a><strong id="b6916192019187"><a name="b6916192019187"></a><a name="b6916192019187"></a>const std::vector&lt;int&gt; &amp;deviceList</strong>：Device侧资源配置。</p>
<p id="p1411722401512"><a name="p1411722401512"></a><a name="p1411722401512"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为“-1”，表示设置为“128MB”。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b13793144414268"><a name="b13793144414268"></a><a name="b13793144414268"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1768605017262"></a><a name="ul1768605017262"></a><ul id="ul1768605017262"><li>dim ∈ {32, 64, 128, 256, 384, 512}</li><li>metricType：AscendIndexILFlat目前只实现了向量内积距离，即只支持“faiss::MetricType::METRIC_INNER_PRODUCT”。</li><li>capacity：接口允许为底库申请的内存上限设为12,288,000,000Byte，同时“capacity”的值域约束为[0, 12000000]。<a name="ul138816512117"></a><a name="ul138816512117"></a><ul id="ul138816512117"><li>以512维、FP16类型的底库向量为例，最大支持的<span class="parmname" id="parmname1593195143016"><a name="parmname1593195143016"></a><a name="parmname1593195143016"></a>“capacity”</span>为1200万(12288000000 / (512 * sizeof(fp_16)) )。</li><li>对于256维、FP16类型的底库向量，尽管内存约束支持更大的<span class="parmname" id="parmname15389183615368"><a name="parmname15389183615368"></a><a name="parmname15389183615368"></a>“capacity”</span>，<span class="parmname" id="parmname169642468367"><a name="parmname169642468367"></a><a name="parmname169642468367"></a>“capacity”</span>最大也只能设为1200万。</li></ul>
</li><li>仅支持配置单卡，暂不支持配置多卡，需满足<strong id="b16270210371"><a name="b16270210371"></a><a name="b16270210371"></a>deviceList.size() == 1</strong>。</li><li>resourceSize：可以配置为-1或[134217728，4294967296]之间的值，相当于[128MB，4096MB]。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置为“1024MB”。</li></ul>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000002482794858"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4759192852812"><a name="p4759192852812"></a><a name="p4759192852812"></a>AscendIndexILFlat&amp; operator=(const AscendIndexILFlat &amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5787143522812"><a name="p5787143522812"></a><a name="p5787143522812"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p175601140172811"><a name="p175601140172811"></a><a name="p175601140172811"></a><strong id="b1023216252912"><a name="b1023216252912"></a><a name="b1023216252912"></a>const AscendIndexILFlat &amp;</strong>：AscendIndexILFlat对象。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## RemoveFeatures接口<a name="ZH-CN_TOPIC_0000002482917750"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p411713313214"><a name="p411713313214"></a><a name="p411713313214"></a>APP_ERROR RemoveFeatures(int n, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>删除向量库中<span class="parmname" id="parmname15331161155517"><a name="parmname15331161155517"></a><a name="parmname15331161155517"></a>“n”</span>个指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b10676733203011"><a name="b10676733203011"></a><a name="b10676733203011"></a>int n</strong>：删除特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b1248654013016"><a name="b1248654013016"></a><a name="b1248654013016"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b727535618302"><a name="b727535618302"></a><a name="b727535618302"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85131151122111"><a name="varname85131151122111"></a><a name="varname85131151122111"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## Search接口<a name="ZH-CN_TOPIC_0000002514896045"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p95721657104813"><a name="p95721657104813"></a><a name="p95721657104813"></a>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询与query向量距离最近的<span class="parmname" id="parmname66101618135916"><a name="parmname66101618135916"></a><a name="parmname66101618135916"></a>“topk”</span>个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出映射后的距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b2797917153611"><a name="b2797917153611"></a><a name="b2797917153611"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1445642010368"><a name="b1445642010368"></a><a name="b1445642010368"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1177612213614"><a name="b1177612213614"></a><a name="b1177612213614"></a>int topk</strong>：查询向量和底库的比对距离进行排序，返回<span class="parmname" id="parmname19303182555918"><a name="parmname19303182555918"></a><a name="parmname19303182555918"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b18837162415364"><a name="b18837162415364"></a><a name="b18837162415364"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b1174653015367"><a name="b1174653015367"></a><a name="b1174653015367"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname173331533163614"><a name="parmname173331533163614"></a><a name="parmname173331533163614"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue1525151355319"><a name="parmvalue1525151355319"></a><a name="parmvalue1525151355319"></a>“48”</span>，即<span class="parmname" id="parmname78791212533"><a name="parmname78791212533"></a><a name="parmname78791212533"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b187841356153711"><a name="b187841356153711"></a><a name="b187841356153711"></a>float *distances</strong>：外部内存，与query相似度最高的<strong id="b8665314543"><a name="b8665314543"></a><a name="b8665314543"></a>topk </strong>* <strong id="b1408103115411"><a name="b1408103115411"></a><a name="b1408103115411"></a>n</strong>个底库特征向量所对应的余弦距离，长度为n * topk。</p>
<p id="p1154614405264"><a name="p1154614405264"></a><a name="p1154614405264"></a><strong id="b12933115811373"><a name="b12933115811373"></a><a name="b12933115811373"></a>idx_t *indices</strong>：外部内存，返回与query相似度最高的<span class="parmname" id="parmname1346104111594"><a name="parmname1346104111594"></a><a name="parmname1346104111594"></a>“topk”</span>个底库向量对应的下标索引，长度为n * topk。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b111527319385"><a name="b111527319385"></a><a name="b111527319385"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1346615102548"></a><a name="ul1346615102548"></a><ul id="ul1346615102548"><li><strong id="b5538111715545"><a name="b5538111715545"></a><a name="b5538111715545"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname17384527155412"><a name="varname17384527155412"></a><a name="varname17384527155412"></a>capacity</span></i>]之间。</li><li><strong id="b18681518105411"><a name="b18681518105411"></a><a name="b18681518105411"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table838713119461"></a>
<table><tbody><tr id="row33871117462"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p83873114461"><a name="p83873114461"></a><a name="p83873114461"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p33878119466"><a name="p33878119466"></a><a name="p33878119466"></a>APP_ERROR Search(int n, const float *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row4388161184611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p143889110469"><a name="p143889110469"></a><a name="p143889110469"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1388121164616"><a name="p1388121164616"></a><a name="p1388121164616"></a>查询与query向量距离最近的<span class="parmname" id="parmname138821154613"><a name="parmname138821154613"></a><a name="parmname138821154613"></a>“topk”</span>个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出映射后的距离。</p>
</td>
</tr>
<tr id="row1038821104617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p9388121124613"><a name="p9388121124613"></a><a name="p9388121124613"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p113881518469"><a name="p113881518469"></a><a name="p113881518469"></a><strong id="b123881015468"><a name="b123881015468"></a><a name="b123881015468"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p6388111204615"><a name="p6388111204615"></a><a name="p6388111204615"></a><strong id="b4285489460"><a name="b4285489460"></a><a name="b4285489460"></a>const float *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p1738816112464"><a name="p1738816112464"></a><a name="p1738816112464"></a><strong id="b838811114614"><a name="b838811114614"></a><a name="b838811114614"></a>int topk</strong>：查询向量和底库的比对距离进行排序，返回<span class="parmname" id="parmname173881511463"><a name="parmname173881511463"></a><a name="parmname173881511463"></a>“topk”</span>条结果。</p>
<p id="p1338831154612"><a name="p1338831154612"></a><a name="p1338831154612"></a><strong id="b1438814115461"><a name="b1438814115461"></a><a name="b1438814115461"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1238861104619"><a name="parmvalue1238861104619"></a><a name="parmvalue1238861104619"></a>“10000”</span>。</p>
<p id="p183888134612"><a name="p183888134612"></a><a name="p183888134612"></a><strong id="b438816119468"><a name="b438816119468"></a><a name="b438816119468"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname133881219461"><a name="parmname133881219461"></a><a name="parmname133881219461"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue83881018461"><a name="parmvalue83881018461"></a><a name="parmvalue83881018461"></a>“48”</span>，即<span class="parmname" id="parmname1938812174616"><a name="parmname1938812174616"></a><a name="parmname1938812174616"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row1938815124610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p13388151134618"><a name="p13388151134618"></a><a name="p13388151134618"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p538810111463"><a name="p538810111463"></a><a name="p538810111463"></a><strong id="b1338861114611"><a name="b1338861114611"></a><a name="b1338861114611"></a>float *distances</strong>：外部内存，与query相似度最高的<strong id="b10388171164619"><a name="b10388171164619"></a><a name="b10388171164619"></a>topk </strong>* <strong id="b183884194620"><a name="b183884194620"></a><a name="b183884194620"></a>n</strong>个底库特征向量所对应的余弦距离，长度为n * topk。</p>
<p id="p1638841124616"><a name="p1638841124616"></a><a name="p1638841124616"></a><strong id="b9388191154616"><a name="b9388191154616"></a><a name="b9388191154616"></a>idx_t *indices</strong>：外部内存，返回与query相似度最高的<span class="parmname" id="parmname18388131164610"><a name="parmname18388131164610"></a><a name="parmname18388131164610"></a>“topk”</span>个底库向量对应的下标索引，长度为n * topk。</p>
</td>
</tr>
<tr id="row1938811154610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1838821174615"><a name="p1838821174615"></a><a name="p1838821174615"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p338814194614"><a name="p338814194614"></a><a name="p338814194614"></a><strong id="b1138813119469"><a name="b1138813119469"></a><a name="b1138813119469"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row8388618462"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1338815119460"><a name="p1338815119460"></a><a name="p1338815119460"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1138817154617"></a><a name="ul1138817154617"></a><ul id="ul1138817154617"><li><strong id="b3388181144610"><a name="b3388181144610"></a><a name="b3388181144610"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname838861194611"><a name="varname838861194611"></a><a name="varname838861194611"></a>capacity</span></i>]之间。</li><li><strong id="b13881714467"><a name="b13881714467"></a><a name="b13881714467"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul193898112469"></a><a name="ul193898112469"></a><ul id="ul193898112469"><li><span class="parmname" id="parmname338981164615"><a name="parmname338981164615"></a><a name="parmname338981164615"></a>“indices”</span>、<span class="parmname" id="parmname1938916174613"><a name="parmname1938916174613"></a><a name="parmname1938916174613"></a>“queries”</span>和<span class="parmname" id="parmname143891517460"><a name="parmname143891517460"></a><a name="parmname143891517460"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## SearchByThreshold接口<a name="ZH-CN_TOPIC_0000002482656062"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p111481632134920"><a name="p111481632134920"></a><a name="p111481632134920"></a>APP_ERROR SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1893114655310"><a name="p1893114655310"></a><a name="p1893114655310"></a>在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1378124514396"><a name="b1378124514396"></a><a name="b1378124514396"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1840232010507"><a name="b1840232010507"></a><a name="b1840232010507"></a>const float *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p12923104514555"><a name="p12923104514555"></a><a name="p12923104514555"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname166164371714"><a name="parmname166164371714"></a><a name="parmname166164371714"></a>“threshold”</span>进行过滤。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1578817211311"><a name="parmname1578817211311"></a><a name="parmname1578817211311"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname279351319407"><a name="parmname279351319407"></a><a name="parmname279351319407"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue376417339011"><a name="parmvalue376417339011"></a><a name="parmvalue376417339011"></a>“48”</span>，即<span class="parmname" id="parmname19896039903"><a name="parmname19896039903"></a><a name="parmname19896039903"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b662915439408"><a name="b662915439408"></a><a name="b662915439408"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b15701310524"><a name="b15701310524"></a><a name="b15701310524"></a>idx_t * indices</strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname1633953516114"><a name="parmname1633953516114"></a><a name="parmname1633953516114"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname1829610421613"><a name="parmname1829610421613"></a><a name="parmname1829610421613"></a>“indices”</span>总长度为n * topk。</p>
<p id="p03841120175217"><a name="p03841120175217"></a><a name="p03841120175217"></a><strong id="b1581125094017"><a name="b1581125094017"></a><a name="b1581125094017"></a>float *distances</strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与<span class="parmname" id="parmname681785019417"><a name="parmname681785019417"></a><a name="parmname681785019417"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b8300175210408"><a name="b8300175210408"></a><a name="b8300175210408"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname616535816"><a name="varname616535816"></a><a name="varname616535816"></a>capacity</span></i>]之间。</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname1431611816133"><a name="parmname1431611816133"></a><a name="parmname1431611816133"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table910711421721"></a>
<table><tbody><tr id="row13108642623"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1610817426217"><a name="p1610817426217"></a><a name="p1610817426217"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1013445511210"><a name="p1013445511210"></a><a name="p1013445511210"></a>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row141085421821"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1510864219218"><a name="p1510864219218"></a><a name="p1510864219218"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1810818421423"><a name="p1810818421423"></a><a name="p1810818421423"></a>在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row12108942725"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p131081342022"><a name="p131081342022"></a><a name="p131081342022"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21081142426"><a name="p21081142426"></a><a name="p21081142426"></a><strong id="b810811421722"><a name="b810811421722"></a><a name="b810811421722"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p16108194212219"><a name="p16108194212219"></a><a name="p16108194212219"></a><strong id="b026811345317"><a name="b026811345317"></a><a name="b026811345317"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p410824217215"><a name="p410824217215"></a><a name="p410824217215"></a><strong id="b1510814425212"><a name="b1510814425212"></a><a name="b1510814425212"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname7108442521"><a name="parmname7108442521"></a><a name="parmname7108442521"></a>“threshold”</span>进行过滤。</p>
<p id="p121087424218"><a name="p121087424218"></a><a name="p121087424218"></a><strong id="b31086422216"><a name="b31086422216"></a><a name="b31086422216"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1010817428219"><a name="parmname1010817428219"></a><a name="parmname1010817428219"></a>“topk”</span>条结果。</p>
<p id="p21081342623"><a name="p21081342623"></a><a name="p21081342623"></a><strong id="b20108134216211"><a name="b20108134216211"></a><a name="b20108134216211"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue181081542727"><a name="parmvalue181081542727"></a><a name="parmvalue181081542727"></a>“10000”</span>。</p>
<p id="p51081421027"><a name="p51081421027"></a><a name="p51081421027"></a><strong id="b101081421825"><a name="b101081421825"></a><a name="b101081421825"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname810816423210"><a name="parmname810816423210"></a><a name="parmname810816423210"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue01080422217"><a name="parmvalue01080422217"></a><a name="parmvalue01080422217"></a>“48”</span>，即<span class="parmname" id="parmname111089425219"><a name="parmname111089425219"></a><a name="parmname111089425219"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row1010816424210"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p81081742228"><a name="p81081742228"></a><a name="p81081742228"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p121088421922"><a name="p121088421922"></a><a name="p121088421922"></a><strong id="b11108142324"><a name="b11108142324"></a><a name="b11108142324"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p1310817428215"><a name="p1310817428215"></a><a name="p1310817428215"></a><strong id="b12433151942"><a name="b12433151942"></a><a name="b12433151942"></a>idx_t* indices</strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname5108174214220"><a name="parmname5108174214220"></a><a name="parmname5108174214220"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname201086421622"><a name="parmname201086421622"></a><a name="parmname201086421622"></a>“indices”</span>总长度为n * topk。</p>
<p id="p6108194217212"><a name="p6108194217212"></a><a name="p6108194217212"></a><strong id="b81081429211"><a name="b81081429211"></a><a name="b81081429211"></a>float *distances</strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与<span class="parmname" id="parmname1310824214216"><a name="parmname1310824214216"></a><a name="parmname1310824214216"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row1810854219219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p310810423210"><a name="p310810423210"></a><a name="p310810423210"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p810819421524"><a name="p810819421524"></a><a name="p810819421524"></a><strong id="b710815421022"><a name="b710815421022"></a><a name="b710815421022"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1110811421218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p191086421521"><a name="p191086421521"></a><a name="p191086421521"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul2010818425217"></a><a name="ul2010818425217"></a><ul id="ul2010818425217"><li><strong id="b1810834211215"><a name="b1810834211215"></a><a name="b1810834211215"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname8108164219220"><a name="varname8108164219220"></a><a name="varname8108164219220"></a>capacity</span></i>]之间。</li><li><strong id="b610974212213"><a name="b610974212213"></a><a name="b610974212213"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul01091542821"></a><a name="ul01091542821"></a><ul id="ul01091542821"><li><span class="parmname" id="parmname3109184216214"><a name="parmname3109184216214"></a><a name="parmname3109184216214"></a>“indices”</span>、<span class="parmname" id="parmname1410910421821"><a name="parmname1410910421821"></a><a name="parmname1410910421821"></a>“queries”</span>、<span class="parmname" id="parmname310910421129"><a name="parmname310910421129"></a><a name="parmname310910421129"></a>“distances”</span>和<span class="parmname" id="parmname19109104217218"><a name="parmname19109104217218"></a><a name="parmname19109104217218"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## SetNTotal接口<a name="ZH-CN_TOPIC_0000002514776045"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR SetNTotal(int n);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p7313759183119"><a name="p7313759183119"></a><a name="p7313759183119"></a>为外部提供调整<span class="parmname" id="parmname159164443116"><a name="parmname159164443116"></a><a name="parmname159164443116"></a>“ntotal”</span>计数。</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>每次增加底库向量后，Index内部尽管会根据最大插入下标更新<span class="parmname" id="parmname114061192322"><a name="parmname114061192322"></a><a name="parmname114061192322"></a>“ntotal”</span>值，但并没有记录[0, <i><span class="varname" id="varname1917151317325"><a name="varname1917151317325"></a><a name="varname1917151317325"></a>ntotal</span></i>]范围内哪些区域是无效的空间，因此<strong id="b144482818157"><a name="b144482818157"></a><a name="b144482818157"></a>RemoveFeatures</strong>操作没有改变<span class="parmname" id="parmname1274441121512"><a name="parmname1274441121512"></a><a name="parmname1274441121512"></a>“ntotal”</span>的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置<span class="parmname" id="parmname159521818143214"><a name="parmname159521818143214"></a><a name="parmname159521818143214"></a>“ntotal”</span>，这样可以在可控范围内减少算子的计算量，以提高接口性能。</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>例如：当前插入100条向量，底库索引为0~99 时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部<span class="parmname" id="parmname974517165332"><a name="parmname974517165332"></a><a name="parmname974517165332"></a>“ntotal”</span>保持不变，只能设为[<i><span class="varname" id="varname169891216331"><a name="varname169891216331"></a><a name="varname169891216331"></a>ntotal</span></i>, <i><span class="varname" id="varname91661324163313"><a name="varname91661324163313"></a><a name="varname91661324163313"></a>capacity</span></i>]之间的值，再次执行删除索引为90~99的底库，此时可以手动把<span class="parmname" id="parmname18801143812373"><a name="parmname18801143812373"></a><a name="parmname18801143812373"></a>“ntotal”</span>设置为[80, <i><span class="varname" id="varname737175673612"><a name="varname737175673612"></a><a name="varname737175673612"></a>capacity</span></i>]之间的值，设置为<span class="parmvalue" id="parmvalue8748356153711"><a name="parmvalue8748356153711"></a><a name="parmvalue8748356153711"></a>“80”</span>时，可以使参与比对的底库数据量有效减少20条。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1215142783714"><a name="b1215142783714"></a><a name="b1215142783714"></a>int n</strong>：由用户在业务面管理的最大底库的索引加1。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</p>
</td>
</tr>
</tbody>
</table>

## UpdateFeatures接口<a name="ZH-CN_TOPIC_0000002516314733"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p119217478565"><a name="p119217478565"></a><a name="p119217478565"></a>APP_ERROR UpdateFeatures (int n, const float16_t *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>向特征库更新“n”个指定下标索引的特征向量，如果在下标处不存在特征向量，则添加；如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：插入特征向量数目。</p>
<p id="p042220329586"><a name="p042220329586"></a><a name="p042220329586"></a><strong id="b17419938175818"><a name="b17419938175818"></a><a name="b17419938175818"></a>const float16_t *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
<p id="p18422153235817"><a name="p18422153235817"></a><a name="p18422153235817"></a><strong id="b5921164425815"><a name="b5921164425815"></a><a name="b5921164425815"></a>const idx_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table19567183517113"></a>
<table><tbody><tr id="row145678353110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1056713351120"><a name="p1056713351120"></a><a name="p1056713351120"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13567835018"><a name="p13567835018"></a><a name="p13567835018"></a>APP_ERROR UpdateFeatures(int n, const float *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row1256719351818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p175675351511"><a name="p175675351511"></a><a name="p175675351511"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p20567123520115"><a name="p20567123520115"></a><a name="p20567123520115"></a>向特征库更新“n”个指定下标索引的特征向量，如果在下标处不存在特征向量，则添加；如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row756713352110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1456793520110"><a name="p1456793520110"></a><a name="p1456793520110"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p2567133518120"><a name="p2567133518120"></a><a name="p2567133518120"></a><strong id="b65671335119"><a name="b65671335119"></a><a name="b65671335119"></a>int n</strong>：插入特征向量数目。</p>
<p id="p165673357114"><a name="p165673357114"></a><a name="p165673357114"></a><strong id="b3277341929"><a name="b3277341929"></a><a name="b3277341929"></a>const float *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
<p id="p175671135216"><a name="p175671135216"></a><a name="p175671135216"></a><strong id="b2567173515117"><a name="b2567173515117"></a><a name="b2567173515117"></a>const idx_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为n。</p>
</td>
</tr>
<tr id="row456710351212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1556715352111"><a name="p1556715352111"></a><a name="p1556715352111"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1956716351718"><a name="p1956716351718"></a><a name="p1956716351718"></a>无</p>
</td>
</tr>
<tr id="row155678359112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p115677351519"><a name="p115677351519"></a><a name="p115677351519"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1056773515119"><a name="p1056773515119"></a><a name="p1056773515119"></a><strong id="b1456793518119"><a name="b1456793518119"></a><a name="b1456793518119"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row65673351912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9567135715"><a name="p9567135715"></a><a name="p9567135715"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1567203512115"></a><a name="ul1567203512115"></a><ul id="ul1567203512115"><li><strong id="b556783510117"><a name="b556783510117"></a><a name="b556783510117"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname185673351619"><a name="varname185673351619"></a><a name="varname185673351619"></a>ntotal</span></i>)之间。</li><li><strong id="b1456713512117"><a name="b1456713512117"></a><a name="b1456713512117"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname1756723512112"><a name="varname1756723512112"></a><a name="varname1756723512112"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname175671351116"><a name="parmname175671351116"></a><a name="parmname175671351116"></a>“features”</span>和<span class="parmname" id="parmname19567935111"><a name="parmname19567935111"></a><a name="parmname19567935111"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>
