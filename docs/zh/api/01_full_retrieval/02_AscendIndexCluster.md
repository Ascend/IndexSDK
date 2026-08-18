# AscendIndexCluster<a id="ZH-CN_TOPIC_0000001614744825"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001564586790"></a>

AscendIndexCluster需要使用[Init](#init接口)指定对应资源的初始化，初始化完之后会申请一段完整空间用于存储底库。在使用完之后，需要调用[Finalize](#finalize接口)接口对资源进行释放。

AscendIndexCluster仅支持使用<term>Atlas 推理系列产品</term>，在标准态部署方式下的向量内积距离类型。AscendIndexCluster在使用时依赖Flat和AICPU算子，具体请参见[Flat](../../05_user_guide.md#flat)和[AICPU](../../05_user_guide.md#aicpu)。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AddFeatures接口<a name="ZH-CN_TOPIC_0000001614746533"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122877269522"><a name="p122877269522"></a><a name="p122877269522"></a>APP_ERROR AddFeatures(int n, const float *features, const uint32_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16287112617527"><a name="p16287112617527"></a><a name="p16287112617527"></a>向特征库插入<span class="parmname" id="parmname51041610175311"><a name="parmname51041610175311"></a><a name="parmname51041610175311"></a>“n”</span>个指定下标索引的特征向量，如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1728872675217"><a name="p1728872675217"></a><a name="p1728872675217"></a><strong id="b428813268527"><a name="b428813268527"></a><a name="b428813268527"></a>int n</strong>：插入特征向量数目。</p>
<p id="p172889261527"><a name="p172889261527"></a><a name="p172889261527"></a><strong id="b92884267526"><a name="b92884267526"></a><a name="b92884267526"></a>const float *features</strong>：待插入的特征向量，长度为<strong id="b317413192537"><a name="b317413192537"></a><a name="b317413192537"></a>n</strong> * 向量维度dim。</p>
<p id="p6288192619521"><a name="p6288192619521"></a><a name="p6288192619521"></a><strong id="b1228832615215"><a name="b1228832615215"></a><a name="b1228832615215"></a>const uint32_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为<strong id="b429616208533"><a name="b429616208533"></a><a name="b429616208533"></a>n</strong>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1628882619521"><a name="p1628882619521"></a><a name="p1628882619521"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p528872613525"><a name="p528872613525"></a><a name="p528872613525"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li><strong id="b6288132610521"><a name="b6288132610521"></a><a name="b6288132610521"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname3195938155216"><a name="varname3195938155216"></a><a name="varname3195938155216"></a>capacity</span></i>)之间，indices要求是连续的。</li><li><strong id="b628892695220"><a name="b628892695220"></a><a name="b628892695220"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname12559104095211"><a name="varname12559104095211"></a><a name="varname12559104095211"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname3816191310019"><a name="parmname3816191310019"></a><a name="parmname3816191310019"></a>“features”</span>和<span class="parmname" id="parmname6551185632610"><a name="parmname6551185632610"></a><a name="parmname6551185632610"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table772538154310"></a>
<table><tbody><tr id="row97256854317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p372568194310"><a name="p372568194310"></a><a name="p372568194310"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19725148124318"><a name="p19725148124318"></a><a name="p19725148124318"></a>APP_ERROR AddFeatures(int n, const uint16_t *features, const int64_t *indices);</p>
</td>
</tr>
<tr id="row9725983433"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1372519814431"><a name="p1372519814431"></a><a name="p1372519814431"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p47259810431"><a name="p47259810431"></a><a name="p47259810431"></a>向特征库插入<span class="parmname" id="parmname872512814432"><a name="parmname872512814432"></a><a name="parmname872512814432"></a>“n”</span>个指定下标索引的特征向量，如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row1272528104315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p167251682439"><a name="p167251682439"></a><a name="p167251682439"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1872513819432"><a name="p1872513819432"></a><a name="p1872513819432"></a><strong id="b1772528184310"><a name="b1772528184310"></a><a name="b1772528184310"></a>int n</strong>：插入特征向量数目。</p>
<p id="p87251087438"><a name="p87251087438"></a><a name="p87251087438"></a><strong id="b16758135110442"><a name="b16758135110442"></a><a name="b16758135110442"></a>const uint16_t *features</strong>：待插入的特征向量，长度为<strong id="b1072519824314"><a name="b1072519824314"></a><a name="b1072519824314"></a>n</strong> * 向量维度dim。</p>
<p id="p672518884310"><a name="p672518884310"></a><a name="p672518884310"></a><strong id="b13589161410453"><a name="b13589161410453"></a><a name="b13589161410453"></a>const int64_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为<strong id="b1672510864314"><a name="b1672510864314"></a><a name="b1672510864314"></a>n</strong>。</p>
</td>
</tr>
<tr id="row187251389432"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p20725788435"><a name="p20725788435"></a><a name="p20725788435"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p3725198194318"><a name="p3725198194318"></a><a name="p3725198194318"></a>无</p>
</td>
</tr>
<tr id="row672517820435"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p57254812434"><a name="p57254812434"></a><a name="p57254812434"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p107256824313"><a name="p107256824313"></a><a name="p107256824313"></a><strong id="b97252810435"><a name="b97252810435"></a><a name="b97252810435"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1972548114318"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9725582431"><a name="p9725582431"></a><a name="p9725582431"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul87251988433"></a><a name="ul87251988433"></a><ul id="ul87251988433"><li><strong id="b1872558124320"><a name="b1872558124320"></a><a name="b1872558124320"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname1272528114317"><a name="varname1272528114317"></a><a name="varname1272528114317"></a>capacity</span></i>)之间。</li><li><strong id="b67251181438"><a name="b67251181438"></a><a name="b67251181438"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname117251088432"><a name="varname117251088432"></a><a name="varname117251088432"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname872512894317"><a name="parmname872512894317"></a><a name="parmname872512894317"></a>“features”</span>和<span class="parmname" id="parmname1172514824315"><a name="parmname1172514824315"></a><a name="parmname1172514824315"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## AscendIndexCluster接口<a name="ZH-CN_TOPIC_0000001564746410"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p9608143314716"><a name="p9608143314716"></a><a name="p9608143314716"></a>AscendIndexCluster();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1760814333474"><a name="p1760814333474"></a><a name="p1760814333474"></a>AscendIndexCluster的构造函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p204291821488"><a name="p204291821488"></a><a name="p204291821488"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p194301211486"><a name="p194301211486"></a><a name="p194301211486"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1743014244819"><a name="p1743014244819"></a><a name="p1743014244819"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20430172184812"><a name="p20430172184812"></a><a name="p20430172184812"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table15621560282"></a>
<table><tbody><tr id="row1256265642816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p20562165614286"><a name="p20562165614286"></a><a name="p20562165614286"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p29285682519"><a name="p29285682519"></a><a name="p29285682519"></a>AscendIndexCluster(const AscendIndexCluster&amp;) = delete;</p>
</td>
</tr>
<tr id="row756235619282"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p95621656152818"><a name="p95621656152818"></a><a name="p95621656152818"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p538993116244"><a name="p538993116244"></a><a name="p538993116244"></a>声明此Index拷贝函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row356225619283"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1956245682817"><a name="p1956245682817"></a><a name="p1956245682817"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p998472882614"><a name="p998472882614"></a><a name="p998472882614"></a><strong id="b188011316264"><a name="b188011316264"></a><a name="b188011316264"></a>const AscendIndexCluster&amp;</strong>：AscendIndexCluster对象。</p>
</td>
</tr>
<tr id="row55621556102815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p135627560287"><a name="p135627560287"></a><a name="p135627560287"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row0562256142813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1656225622819"><a name="p1656225622819"></a><a name="p1656225622819"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row145621856162814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19562105616284"><a name="p19562105616284"></a><a name="p19562105616284"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## \~AscendIndexCluster接口<a name="ZH-CN_TOPIC_0000002399598393"></a>

<a name="table179216322487"></a>
<table><tbody><tr id="row2092173214484"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3929320480"><a name="p3929320480"></a><a name="p3929320480"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p118119814816"><a name="p118119814816"></a><a name="p118119814816"></a>virtual ~AscendIndexCluster() = default;</p>
</td>
</tr>
<tr id="row092163217481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p493532184819"><a name="p493532184819"></a><a name="p493532184819"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2811384485"><a name="p2811384485"></a><a name="p2811384485"></a>AscendIndexCluster的析构函数。</p>
</td>
</tr>
<tr id="row1193163244813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p19316328485"><a name="p19316328485"></a><a name="p19316328485"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1581178114813"><a name="p1581178114813"></a><a name="p1581178114813"></a>无</p>
</td>
</tr>
<tr id="row9931932104818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p793173212484"><a name="p793173212484"></a><a name="p793173212484"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p148111782488"><a name="p148111782488"></a><a name="p148111782488"></a>无</p>
</td>
</tr>
<tr id="row89333214481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p89316323489"><a name="p89316323489"></a><a name="p89316323489"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p58111481480"><a name="p58111481480"></a><a name="p58111481480"></a>无</p>
</td>
</tr>
<tr id="row49311328486"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p993113234818"><a name="p993113234818"></a><a name="p993113234818"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4811148104812"><a name="p4811148104812"></a><a name="p4811148104812"></a>无</p>
</td>
</tr>
</tbody>
</table>

## ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000002446061685"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p68995910575"><a name="p68995910575"></a><a name="p68995910575"></a>APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num, const uint32_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
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
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b276887014"><a name="b276887014"></a><a name="b276887014"></a>const uint16_t *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname1441759144217"><a name="parmname1441759144217"></a><a name="parmname1441759144217"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a>const int *num</strong>：给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b921610409216"><a name="b921610409216"></a><a name="b921610409216"></a>const uint32_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname2711154912437"><a name="parmname2711154912437"></a><a name="parmname2711154912437"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname742124364316"><a name="parmname742124364316"></a><a name="parmname742124364316"></a>“indices”</span>长度为n * max(num)。</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname82723561324"><a name="varname82723561324"></a><a name="varname82723561324"></a>capacity</span></i>]之间。</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0， ntotal]之间。</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
</td>
</tr>
</tbody>
</table>

## ComputeDistanceByThreshold接口<a name="ZH-CN_TOPIC_0000001615066169"></a>

> [!NOTE]
>当前接口需配合[AddFeatures\(int n, const float \*features, const uint32\_t \*indices\);](#addfeatures接口)使用。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.1.1 "><p id="p15352133820537"><a name="p15352133820537"></a><a name="p15352133820537"></a>APP_ERROR ComputeDistanceByThreshold(const std::vector&lt;uint32_t&gt; &amp;queryIdxArr, uint32_t codeStartIdx,  uint32_t codeNum, float threshold, bool aboveFilter, std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr, std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.2.1 "><p id="p1935283855318"><a name="p1935283855318"></a><a name="p1935283855318"></a>查询指定条数在底库中的特征向量与指定的底库特征向量的距离，并根据阈值筛选，返回满足条件的距离和其label。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.3.1 "><p id="p1535283885320"><a name="p1535283885320"></a><a name="p1535283885320"></a><strong id="b976392112545"><a name="b976392112545"></a><a name="b976392112545"></a>const std::vector&lt;uint32_t&gt; &amp;queryIdxArr</strong>：要查询的向量在底库中的序号。</p>
<p id="p1835213385530"><a name="p1835213385530"></a><a name="p1835213385530"></a><strong id="b57881123165418"><a name="b57881123165418"></a><a name="b57881123165418"></a>uint32_t codeStartIdx</strong>：要计算距离的底库的起始序号。</p>
<p id="p23526383537"><a name="p23526383537"></a><a name="p23526383537"></a><strong id="b16334152510547"><a name="b16334152510547"></a><a name="b16334152510547"></a>uint32_t codeNum</strong>：要计算距离的底库向量的数量。</p>
<p id="p203521138195313"><a name="p203521138195313"></a><a name="p203521138195313"></a><strong id="b112831830115415"><a name="b112831830115415"></a><a name="b112831830115415"></a>float threshold</strong>：用于过滤的阈值，过滤掉比阈值小的距离。</p>
<p id="p1435223815538"><a name="p1435223815538"></a><a name="p1435223815538"></a><strong id="b2932133114541"><a name="b2932133114541"></a><a name="b2932133114541"></a>bool aboveFilter</strong>：预留参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.4.1 "><p id="p13522382534"><a name="p13522382534"></a><a name="p13522382534"></a><strong id="b0375144918549"><a name="b0375144918549"></a><a name="b0375144918549"></a>std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr</strong>：返回的二维数组，每个要查询的向量与其满足阈值条件的底库向量的距离。</p>
<p id="p83524381530"><a name="p83524381530"></a><a name="p83524381530"></a><strong id="b166640512549"><a name="b166640512549"></a><a name="b166640512549"></a>std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr</strong>：返回的二维数组，每个要查询的向量与其满足阈值条件的底库向量的label。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.5.1 "><p id="p735215385536"><a name="p735215385536"></a><a name="p735215385536"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.6.1 "><a name="ul1624216519526"></a><a name="ul1624216519526"></a><ul id="ul1624216519526"><li><span class="parmname" id="parmname138712196568"><a name="parmname138712196568"></a><a name="parmname138712196568"></a>“queryIdxArr”</span>与<span class="parmname" id="parmname181661023145619"><a name="parmname181661023145619"></a><a name="parmname181661023145619"></a>“resDistArr”</span>和<span class="parmname" id="parmname1785414248565"><a name="parmname1785414248565"></a><a name="parmname1785414248565"></a>“resIdxArr”</span>长度要一致，即<strong id="b591716125616"><a name="b591716125616"></a><a name="b591716125616"></a>queryIdxArr.size() == resDistArr.size()</strong>。</li><li><span class="parmname" id="parmname1560612576523"><a name="parmname1560612576523"></a><a name="parmname1560612576523"></a>“queryIdxArr.size()”</span>需大于<span class="parmvalue" id="parmvalue71274012531"><a name="parmvalue71274012531"></a><a name="parmvalue71274012531"></a>“0”</span>并且小于等于<span class="parmname" id="parmname1723074011015"><a name="parmname1723074011015"></a><a name="parmname1723074011015"></a>“ntotal”</span>。</li><li><span class="parmname" id="parmname7798135125316"><a name="parmname7798135125316"></a><a name="parmname7798135125316"></a>“codeNum”</span>需大于<span class="parmvalue" id="parmvalue11855877538"><a name="parmvalue11855877538"></a><a name="parmvalue11855877538"></a>“0”</span>并且小于等于<span class="parmname" id="parmname18896901916"><a name="parmname18896901916"></a><a name="parmname18896901916"></a>“ntotal”</span>。</li><li><span class="parmname" id="parmname19911327165616"><a name="parmname19911327165616"></a><a name="parmname19911327165616"></a>“codeStartIdx”</span> + <span class="parmname" id="parmname10710830185612"><a name="parmname10710830185612"></a><a name="parmname10710830185612"></a>“codeNum”</span>不大于<span class="parmname" id="parmname163571213569"><a name="parmname163571213569"></a><a name="parmname163571213569"></a>“ntotal”</span>（底库大小）。</li><li><span class="parmname" id="parmname6690927114312"><a name="parmname6690927114312"></a><a name="parmname6690927114312"></a>“codeStartIdx”</span>需大于等于<span class="parmvalue" id="parmvalue136896102437"><a name="parmvalue136896102437"></a><a name="parmvalue136896102437"></a>“0”</span>并且小于等于<span class="parmname" id="parmname3409151774318"><a name="parmname3409151774318"></a><a name="parmname3409151774318"></a>“ntotal”</span></li></ul>
</td>
</tr>
</tbody>
</table>

## Finalize接口<a name="ZH-CN_TOPIC_0000001614906601"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p377414915513"><a name="p377414915513"></a><a name="p377414915513"></a>void Finalize();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>释放特征库管理资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p877484919511"><a name="p877484919511"></a><a name="p877484919511"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14774134917519"><a name="p14774134917519"></a><a name="p14774134917519"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1277416496515"><a name="p1277416496515"></a><a name="p1277416496515"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16774184918516"><a name="p16774184918516"></a><a name="p16774184918516"></a>无</p>
</td>
</tr>
</tbody>
</table>

## GetFeatures接口<a name="ZH-CN_TOPIC_0000002412742482"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeatures(int n, uint16_t *features, const int64_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname9635334135520"><a name="parmname9635334135520"></a><a name="parmname9635334135520"></a>“n”</span>条指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b13667438181211"><a name="b13667438181211"></a><a name="b13667438181211"></a>const int64_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b2033375501314"><a name="b2033375501314"></a><a name="b2033375501314"></a>uint16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
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

## GetNTotal接口<a name="ZH-CN_TOPIC_0000002412582646"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p61958153167"><a name="p61958153167"></a><a name="p61958153167"></a>int GetNTotal() const;</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b171461107533"><a name="b171461107533"></a><a name="b171461107533"></a>int</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>

## Init接口<a name="ZH-CN_TOPIC_0000001614866169"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p463919713918"><a name="p463919713918"></a><a name="p463919713918"></a>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p4783164014487"><a name="p4783164014487"></a><a name="p4783164014487"></a>AscendIndexCluster的初始化函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14783940124815"><a name="p14783940124815"></a><a name="p14783940124815"></a><strong id="b10219125824811"><a name="b10219125824811"></a><a name="b10219125824811"></a>int dim</strong>：AscendIndexCluster管理的特征向量的维度。</p>
<p id="p178314407484"><a name="p178314407484"></a><a name="p178314407484"></a><strong id="b391943494"><a name="b391943494"></a><a name="b391943494"></a>int capacity</strong>：底库最大容量，接口会根据<span class="parmname" id="parmname15611164915567"><a name="parmname15611164915567"></a><a name="parmname15611164915567"></a>“capacity”</span>值申请capacity * dim * sizeof(fp16) 字节内存数据。</p>
<p id="p1478354004816"><a name="p1478354004816"></a><a name="p1478354004816"></a><strong id="b05501363492"><a name="b05501363492"></a><a name="b05501363492"></a>faiss::MetricType metricType</strong>：特征距离类别（向量内积、欧氏距离、余弦相似度）。</p>
<p id="p978314010482"><a name="p978314010482"></a><a name="p978314010482"></a><strong id="b12591941184914"><a name="b12591941184914"></a><a name="b12591941184914"></a>const std::vector&lt;int&gt; &amp;deviceList</strong>：Device侧资源配置。</p>
<p id="p278364014481"><a name="p278364014481"></a><a name="p278364014481"></a><strong id="b19288144414914"><a name="b19288144414914"></a><a name="b19288144414914"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为<span class="parmvalue" id="parmvalue116601371297"><a name="parmvalue116601371297"></a><a name="parmvalue116601371297"></a>“-1”</span>，表示设置为<span class="parmvalue" id="parmvalue16975511132911"><a name="parmvalue16975511132911"></a><a name="parmvalue16975511132911"></a>“128MB”</span>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1378314016485"><a name="p1378314016485"></a><a name="p1378314016485"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p4783174017482"><a name="p4783174017482"></a><a name="p4783174017482"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3783154018480"></a><a name="ul3783154018480"></a><ul id="ul3783154018480"><li>dim ∈ {32, 64, 128, 256, 384, 512}。</li><li>metricType：AscendIndexCluster目前只实现了向量内积距离，即只支持<span class="parmvalue" id="parmvalue153881199571"><a name="parmvalue153881199571"></a><a name="parmvalue153881199571"></a>“faiss::MetricType::METRIC_INNER_PRODUCT。”</span></li><li>接口允许为底库申请的内存上限设为12,288,000,000Byte，同时<span class="parmname" id="parmname1951055020570"><a name="parmname1951055020570"></a><a name="parmname1951055020570"></a>“capacity”</span>的值域约束为[0, 12000000]。</li><li>以512维、FP16类型的底库向量为例，最大支持的<span class="parmname" id="parmname10415122055714"><a name="parmname10415122055714"></a><a name="parmname10415122055714"></a>“capacity”</span>为1200万( 12288000000 / (512 * sizeof(fp_16)) )。</li><li>对于256维、FP16类型的底库向量，尽管内存约束支持更大的<span class="parmname" id="parmname15756192115712"><a name="parmname15756192115712"></a><a name="parmname15756192115712"></a>“capacity”</span>，<span class="parmname" id="parmname3788102218578"><a name="parmname3788102218578"></a><a name="parmname3788102218578"></a>“capacity”</span>最大也只能设为1200万。</li><li>仅支持配置单卡，暂不支持配置多卡，需满足<strong id="b1129012151811"><a name="b1129012151811"></a><a name="b1129012151811"></a>deviceList.size() == 1</strong>。</li><li><span class="parmname" id="parmname2017164055710"><a name="parmname2017164055710"></a><a name="parmname2017164055710"></a>“resourceSize”</span>：可以配置为-1或[134217728，4294967296]之间的值，相当于[128MB，4096MB]。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置为<span class="parmvalue" id="parmvalue15293111673918"><a name="parmvalue15293111673918"></a><a name="parmvalue15293111673918"></a>“1024MB”</span>。</li></ul>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001897100377"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19940753122510"><a name="p19940753122510"></a><a name="p19940753122510"></a>AscendIndexCluster&amp; operator=(const AscendIndexCluster&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3198107264"><a name="p3198107264"></a><a name="p3198107264"></a>声明此Index赋值构造函数为空，即不可拷贝类型</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6538114402614"><a name="p6538114402614"></a><a name="p6538114402614"></a><strong id="b18689747142614"><a name="b18689747142614"></a><a name="b18689747142614"></a>const AscendIndexCluster&amp;</strong>：AscendIndexCluster对象。</p>
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

## RemoveFeatures接口<a name="ZH-CN_TOPIC_0000002446181741"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p457211594914"><a name="p457211594914"></a><a name="p457211594914"></a>APP_ERROR RemoveFeatures(int n, const int64_t *indices);</p>
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
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b16204233201013"><a name="b16204233201013"></a><a name="b16204233201013"></a>const int64_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
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

## SearchByThreshold接口<a name="ZH-CN_TOPIC_0000002446061689"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p126941862232"><a name="p126941862232"></a><a name="p126941862232"></a>APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num, int64_t * indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p138361625164412"><a name="p138361625164412"></a><a name="p138361625164412"></a>在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1378124514396"><a name="b1378124514396"></a><a name="b1378124514396"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1148123874520"><a name="b1148123874520"></a><a name="b1148123874520"></a>const uint16_t *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p11518191412248"><a name="p11518191412248"></a><a name="p11518191412248"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname166164371714"><a name="parmname166164371714"></a><a name="parmname166164371714"></a>“threshold”</span>进行过滤。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1578817211311"><a name="parmname1578817211311"></a><a name="parmname1578817211311"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname279351319407"><a name="parmname279351319407"></a><a name="parmname279351319407"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue376417339011"><a name="parmvalue376417339011"></a><a name="parmvalue376417339011"></a>“48”</span>，即<span class="parmname" id="parmname19896039903"><a name="parmname19896039903"></a><a name="parmname19896039903"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18962112242616"><a name="p18962112242616"></a><a name="p18962112242616"></a><strong id="b39757334268"><a name="b39757334268"></a><a name="b39757334268"></a>int *num：</strong>每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p1796272252611"><a name="p1796272252611"></a><a name="p1796272252611"></a><strong id="b647125110265"><a name="b647125110265"></a><a name="b647125110265"></a>int64_t *indices：</strong>满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname10888162862715"><a name="parmname10888162862715"></a><a name="parmname10888162862715"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname2399203922710"><a name="parmname2399203922710"></a><a name="parmname2399203922710"></a>“indices”</span>总长度为n * topk。</p>
<p id="p296222222618"><a name="p296222222618"></a><a name="p296222222618"></a><strong id="b167278112714"><a name="b167278112714"></a><a name="b167278112714"></a>float *distances：</strong>满足阈值条件的底库向量与待查询向量距离，记录方式和长度与<span class="parmname" id="parmname2406996287"><a name="parmname2406996287"></a><a name="parmname2406996287"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b8300175210408"><a name="b8300175210408"></a><a name="b8300175210408"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname616535816"><a name="varname616535816"></a><a name="varname616535816"></a>capacity</span></i>]之间。</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a>topk</strong>：k取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname1431611816133"><a name="parmname1431611816133"></a><a name="parmname1431611816133"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## SetNTotal<a name="ZH-CN_TOPIC_0000002412742486"></a>

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
