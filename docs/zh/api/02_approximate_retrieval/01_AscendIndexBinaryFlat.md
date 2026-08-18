# AscendIndexBinaryFlat<a name="ZH-CN_TOPIC_0000001506334701"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456694988"></a>

AscendIndexBinaryFlat类继承自Faiss的IndexBinary类，用于二值化特征检索。

仅支持<term>Atlas 推理系列产品</term>。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## add接口<a name="ZH-CN_TOPIC_0000001456854896"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p4703164217446"><a name="p4703164217446"></a><a name="p4703164217446"></a>void add(idx_t n, const uint8_t *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>向底库中添加特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b812832571217"><a name="b812832571217"></a><a name="b812832571217"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b261412289122"><a name="b261412289122"></a><a name="b261412289122"></a>const uint8_t *x</strong>：添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p13415621173615"><a name="p13415621173615"></a><a name="p13415621173615"></a>指针<span class="parmname" id="parmname197639415139"><a name="parmname197639415139"></a><a name="parmname197639415139"></a>“x”</span>的长度应该为dims/8 * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写的错误或程序崩溃。</p>
<p id="p1727814248218"><a name="p1727814248218"></a><a name="p1727814248218"></a>n &gt; 0，add操作需要保证最终底库大小ntotal取<i><span class="varname" id="varname63161016193917"><a name="varname63161016193917"></a><a name="varname63161016193917"></a>芯片内存实际容量</span></i>与<span class="parmvalue" id="parmvalue1831613164393"><a name="parmvalue1831613164393"></a><a name="parmvalue1831613164393"></a>“1e9”</span>之间的较小值。</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。

## add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001506414809"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p15923147164612"><a name="p15923147164612"></a><a name="p15923147164612"></a>void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p359315567338"><a name="p359315567338"></a><a name="p359315567338"></a>向底库中添加特征向量并指定对应的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b21773411615"><a name="b21773411615"></a><a name="b21773411615"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1733711363162"><a name="b1733711363162"></a><a name="b1733711363162"></a>const uint8_t *x</strong>：添加进底库的特征向量。</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b990063701613"><a name="b990063701613"></a><a name="b990063701613"></a>const idx_t *xids</strong>：添加进底库的特征向量对应的ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p019510419535"><a name="p019510419535"></a><a name="p019510419535"></a>0 &lt; n，add操作需要保证最终底库大小n取<i><span class="varname" id="varname63161016193917"><a name="varname63161016193917"></a><a name="varname63161016193917"></a>芯片内存实际容量</span></i>与“1e9”之间的较小值。</p>
<p id="p819694119533"><a name="p819694119533"></a><a name="p819694119533"></a>指针<span class="parmname" id="parmname176054330012"><a name="parmname176054330012"></a><a name="parmname176054330012"></a>“x”</span>的长度应该为dims/8 * n，指针<span class="parmname" id="parmname10355123815013"><a name="parmname10355123815013"></a><a name="parmname10355123815013"></a>“xids”</span>的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。用户需要根据自己的业务场景，保证xids的合法性，如底库中存在重复的ID，search结果中的label将无法对应具体的底库向量。</p>
</td>
</tr>
</tbody>
</table>

## AscendIndexBinaryFlat接口<a name="ZH-CN_TOPIC_0000001456535056"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p33682318435"><a name="p33682318435"></a><a name="p33682318435"></a>AscendIndexBinaryFlat(int dims, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexBinaryFlat的构造函数，生成维度为dims的AscendIndexBinaryFlat，根据<span class="parmname" id="parmname18664330662"><a name="parmname18664330662"></a><a name="parmname18664330662"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b48571551268"><a name="b48571551268"></a><a name="b48571551268"></a>int dims</strong>：AscendIndexBinaryFlat管理的一组特征向量的维度。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b167641631570"><a name="b167641631570"></a><a name="b167641631570"></a>AscendIndexBinaryFlatConfig config</strong>：Device侧资源配置。</p>
<p id="p1280191519597"><a name="p1280191519597"></a><a name="p1280191519597"></a><strong id="b769792117303"><a name="b769792117303"></a><a name="b769792117303"></a>bool usedFloat</strong>：用于入库为二进制、检索特征为float类型的检索方式（<a href="#ZH-CN_TOPIC_0000001456375288">search接口</a>）的性能提升，默认为<span class="parmvalue" id="parmvalue10171681088"><a name="parmvalue10171681088"></a><a name="parmvalue10171681088"></a>“false”</span>；设置为<span class="parmvalue" id="parmvalue4679216884"><a name="parmvalue4679216884"></a><a name="parmvalue4679216884"></a>“true”</span>时表示进行性能提升。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname95619186911"><a name="parmname95619186911"></a><a name="parmname95619186911"></a>“dims”</span>∈ { 256, 512, 1024 }</p>
</td>
</tr>
</tbody>
</table>

<a name="table191641015539"></a>
<table><tbody><tr id="row8164101513314"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p14164141519314"><a name="p14164141519314"></a><a name="p14164141519314"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p1092313612565"><a name="p1092313612565"></a><a name="p1092313612565"></a>AscendIndexBinaryFlat(const faiss::IndexBinaryFlat *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</p>
</td>
</tr>
<tr id="row171644151312"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p101644157311"><a name="p101644157311"></a><a name="p101644157311"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p216571511319"><a name="p216571511319"></a><a name="p216571511319"></a>AscendIndexBinaryFlat的构造函数，基于一个已有的<span class="parmname" id="parmname186437475368"><a name="parmname186437475368"></a><a name="parmname186437475368"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row816511155319"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p141652152034"><a name="p141652152034"></a><a name="p141652152034"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p1216513152314"><a name="p1216513152314"></a><a name="p1216513152314"></a><strong id="b19571543583"><a name="b19571543583"></a><a name="b19571543583"></a>const faiss::IndexBinaryFlat *index</strong>：CPU侧index资源。</p>
<p id="p141651115738"><a name="p141651115738"></a><a name="p141651115738"></a><strong id="b171651315732"><a name="b171651315732"></a><a name="b171651315732"></a>AscendIndexBinaryFlatConfig config</strong>：Device侧资源配置。</p>
<p id="p1516531514317"><a name="p1516531514317"></a><a name="p1516531514317"></a><strong id="b156172041183020"><a name="b156172041183020"></a><a name="b156172041183020"></a>bool usedFloat</strong>：用于入库为二进制、检索特征为float类型的检索方式（<a href="#ZH-CN_TOPIC_0000001456375288">search接口</a>）的性能提升，默认为<span class="parmvalue" id="parmvalue12861143143016"><a name="parmvalue12861143143016"></a><a name="parmvalue12861143143016"></a>“false”</span>；设置为<span class="parmvalue" id="parmvalue78611731193019"><a name="parmvalue78611731193019"></a><a name="parmvalue78611731193019"></a>“true”</span>时表示进行性能提升。</p>
</td>
</tr>
<tr id="row5165515438"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p4165515138"><a name="p4165515138"></a><a name="p4165515138"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p5165161518319"><a name="p5165161518319"></a><a name="p5165161518319"></a>无</p>
</td>
</tr>
<tr id="row1165141515316"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p41650151032"><a name="p41650151032"></a><a name="p41650151032"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p12165315634"><a name="p12165315634"></a><a name="p12165315634"></a>无</p>
</td>
</tr>
<tr id="row2165101519312"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p7165201516316"><a name="p7165201516316"></a><a name="p7165201516316"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><p id="p616501515317"><a name="p616501515317"></a><a name="p616501515317"></a><span class="parmname" id="parmname721815874613"><a name="parmname721815874613"></a><a name="parmname721815874613"></a>“index”</span>需要为合法有效的CPU index指针，index-&gt;d ∈ {256, 512, 1024}，index-&gt;ntotal取<i><span class="varname" id="varname1935761793413"><a name="varname1935761793413"></a><a name="varname1935761793413"></a>芯片内存实际容量</span></i>与<span class="parmvalue" id="parmvalue176360203341"><a name="parmvalue176360203341"></a><a name="parmvalue176360203341"></a>“1e9”</span>之间的较小值。</p>
</td>
</tr>
</tbody>
</table>

<a name="table142022518319"></a>
<table><tbody><tr id="row720152517313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p17201257311"><a name="p17201257311"></a><a name="p17201257311"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p96251456568"><a name="p96251456568"></a><a name="p96251456568"></a>AscendIndexBinaryFlat(const faiss::IndexBinaryIDMap *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);</p>
</td>
</tr>
<tr id="row42092517313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p220152511318"><a name="p220152511318"></a><a name="p220152511318"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p22002516311"><a name="p22002516311"></a><a name="p22002516311"></a>AscendIndexBinaryFlat的构造函数，基于一个已有的<span class="parmname" id="parmname0209251139"><a name="parmname0209251139"></a><a name="parmname0209251139"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row1520625935"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p920725739"><a name="p920725739"></a><a name="p920725739"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p620132517318"><a name="p620132517318"></a><a name="p620132517318"></a><strong id="b12011251937"><a name="b12011251937"></a><a name="b12011251937"></a>const faiss::IndexBinaryIDMap *index</strong>：CPU侧index资源。</p>
<p id="p152011259319"><a name="p152011259319"></a><a name="p152011259319"></a><strong id="b12013257310"><a name="b12013257310"></a><a name="b12013257310"></a>AscendIndexBinaryFlatConfig config</strong>：Device侧资源配置。</p>
<p id="p5201425933"><a name="p5201425933"></a><a name="p5201425933"></a><strong id="b2019215542303"><a name="b2019215542303"></a><a name="b2019215542303"></a>bool usedFloat</strong>：用于入库为二进制、检索特征为float类型的检索方式（<a href="#ZH-CN_TOPIC_0000001456375288">search接口</a>）的性能提升，默认为<span class="parmvalue" id="parmvalue162018253313"><a name="parmvalue162018253313"></a><a name="parmvalue162018253313"></a>“false”</span>；设置为<span class="parmvalue" id="parmvalue9201251534"><a name="parmvalue9201251534"></a><a name="parmvalue9201251534"></a>“true”</span>时表示进行性能提升。</p>
</td>
</tr>
<tr id="row1120122517310"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p02017253319"><a name="p02017253319"></a><a name="p02017253319"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p52142514310"><a name="p52142514310"></a><a name="p52142514310"></a>无</p>
</td>
</tr>
<tr id="row8211825339"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3217255318"><a name="p3217255318"></a><a name="p3217255318"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p02116255316"><a name="p02116255316"></a><a name="p02116255316"></a>无</p>
</td>
</tr>
<tr id="row6216254312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p22182519314"><a name="p22182519314"></a><a name="p22182519314"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p152291327101513"><a name="p152291327101513"></a><a name="p152291327101513"></a><span class="parmname" id="parmname16174215239"><a name="parmname16174215239"></a><a name="parmname16174215239"></a>“index”</span>需要为合法有效的faiss::IndexBinaryIDMap指针，index-&gt;index为合法有效的IndexBinaryFlat指针，index-&gt;index-&gt;d ∈ {256, 512, 1024}，index-&gt;index-&gt;ntotal取<i><span class="varname" id="varname9229152721512"><a name="varname9229152721512"></a><a name="varname9229152721512"></a>芯片内存实际容量</span></i>与<span class="parmvalue" id="parmvalue1422922712157"><a name="parmvalue1422922712157"></a><a name="parmvalue1422922712157"></a>“1e9”</span>之间的较小值。</p>
</td>
</tr>
</tbody>
</table>

<a name="table145324411437"></a>
<table><tbody><tr id="row75329411438"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p145326412034"><a name="p145326412034"></a><a name="p145326412034"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6828175217434"><a name="p6828175217434"></a><a name="p6828175217434"></a>AscendIndexBinaryFlat(const AscendIndexBinaryFlat &amp;) = delete;</p>
</td>
</tr>
<tr id="row0532841735"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1532154111318"><a name="p1532154111318"></a><a name="p1532154111318"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p165321741735"><a name="p165321741735"></a><a name="p165321741735"></a>声明AscendIndexBinaryFlat拷贝构造函数为空，即AscendIndexBinaryFlat为不可拷贝类型。</p>
</td>
</tr>
<tr id="row55324411131"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17532641633"><a name="p17532641633"></a><a name="p17532641633"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b162111517184312"><a name="b162111517184312"></a><a name="b162111517184312"></a>const AscendIndexBinaryFlat &amp;</strong>：常量AscendIndexBinaryFlat。</p>
</td>
</tr>
<tr id="row19532144117319"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p353214114316"><a name="p353214114316"></a><a name="p353214114316"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p953217411937"><a name="p953217411937"></a><a name="p953217411937"></a>无</p>
</td>
</tr>
<tr id="row2532164118313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p653210411131"><a name="p653210411131"></a><a name="p653210411131"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p105324416311"><a name="p105324416311"></a><a name="p105324416311"></a>无</p>
</td>
</tr>
<tr id="row16532041331"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1653294114316"><a name="p1653294114316"></a><a name="p1653294114316"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1253284115310"><a name="p1253284115310"></a><a name="p1253284115310"></a>无</p>
</td>
</tr>
</tbody>
</table>

## \~AscendIndexBinaryFlat接口<a name="ZH-CN_TOPIC_0000001506495917"></a>

<a name="table13115573310"></a>
<table><tbody><tr id="row133117571634"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p93116571312"><a name="p93116571312"></a><a name="p93116571312"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1223817411653"><a name="p1223817411653"></a><a name="p1223817411653"></a>virtual ~AscendIndexBinaryFlat() = default;</p>
</td>
</tr>
<tr id="row131111571314"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p19311857938"><a name="p19311857938"></a><a name="p19311857938"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p20311175712315"><a name="p20311175712315"></a><a name="p20311175712315"></a>AscendIndexBinaryFlat的析构函数，销毁AscendIndexBinaryFlat对象，释放资源。</p>
</td>
</tr>
<tr id="row1631185720315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p331111571035"><a name="p331111571035"></a><a name="p331111571035"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p823816417516"><a name="p823816417516"></a><a name="p823816417516"></a>无</p>
</td>
</tr>
<tr id="row131110576311"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p123112571132"><a name="p123112571132"></a><a name="p123112571132"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p113111757932"><a name="p113111757932"></a><a name="p113111757932"></a>无</p>
</td>
</tr>
<tr id="row2031118575316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2311757432"><a name="p2311757432"></a><a name="p2311757432"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p133117571934"><a name="p133117571934"></a><a name="p133117571934"></a>无</p>
</td>
</tr>
<tr id="row0311205718311"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p03118571131"><a name="p03118571131"></a><a name="p03118571131"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1231195718316"><a name="p1231195718316"></a><a name="p1231195718316"></a>无</p>
</td>
</tr>
</tbody>
</table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001506414941"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyFrom(const faiss::IndexBinaryFlat *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>基于一个已有的Index拷贝到AscendIndexBinaryFlat，清空当前的AscendIndexBinaryFlat底库，并保持原有的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b166777494441"><a name="b166777494441"></a><a name="b166777494441"></a>const faiss::IndexBinaryFlat *index</strong>：faiss::IndexBinaryFlat指针。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname16174215239"><a name="parmname16174215239"></a><a name="parmname16174215239"></a>“index”</span>需要为合法有效的IndexBinaryFlat指针，index-&gt;d ∈ {256, 512, 1024}，index-&gt;ntotal取<i><span class="varname" id="varname1935761793413"><a name="varname1935761793413"></a><a name="varname1935761793413"></a>芯片内存实际容量</span></i>与<span class="parmvalue" id="parmvalue176360203341"><a name="parmvalue176360203341"></a><a name="parmvalue176360203341"></a>“1e9”</span>之间的较小值。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1570816514419"></a>
<table><tbody><tr id="row87089510415"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2070816511342"><a name="p2070816511342"></a><a name="p2070816511342"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p370818511140"><a name="p370818511140"></a><a name="p370818511140"></a>void copyFrom(const faiss::IndexBinaryIDMap *index);</p>
</td>
</tr>
<tr id="row1970816519416"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p157086511048"><a name="p157086511048"></a><a name="p157086511048"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p17086511346"><a name="p17086511346"></a><a name="p17086511346"></a>基于一个已有的<span class="parmname" id="parmname063811221710"><a name="parmname063811221710"></a><a name="parmname063811221710"></a>“index”</span>拷贝到AscendIndexBinaryFlat，清空当前的AscendIndexBinaryFlat底库，并保持原有的Device侧资源配置。</p>
</td>
</tr>
<tr id="row67081551148"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p37082511045"><a name="p37082511045"></a><a name="p37082511045"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1670865117413"><a name="p1670865117413"></a><a name="p1670865117413"></a><strong id="b95691227174516"><a name="b95691227174516"></a><a name="b95691227174516"></a>const faiss::IndexBinaryIDMap *index</strong>：faiss::IndexBinaryIDMap指针。</p>
</td>
</tr>
<tr id="row117082511940"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p197081551340"><a name="p197081551340"></a><a name="p197081551340"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1070811513418"><a name="p1070811513418"></a><a name="p1070811513418"></a>无</p>
</td>
</tr>
<tr id="row1170805111412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p370805118413"><a name="p370805118413"></a><a name="p370805118413"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p117085512418"><a name="p117085512418"></a><a name="p117085512418"></a>无</p>
</td>
</tr>
<tr id="row1370895113414"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p16708951942"><a name="p16708951942"></a><a name="p16708951942"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p77081951843"><a name="p77081951843"></a><a name="p77081951843"></a><span class="parmname" id="parmname3708751241"><a name="parmname3708751241"></a><a name="parmname3708751241"></a>“index”</span>需要为合法有效的faiss::IndexBinaryIDMap指针，index-&gt;index为合法有效的IndexBinaryFlat指针，index-&gt;index-&gt;d ∈ {256, 512, 1024}，index-&gt;index-&gt;ntotal取<i><span class="varname" id="varname670818511944"><a name="varname670818511944"></a><a name="varname670818511944"></a>芯片内存实际容量</span></i>与<span class="parmvalue" id="parmvalue1570819519418"><a name="parmvalue1570819519418"></a><a name="parmvalue1570819519418"></a>“1e9”</span>之间的较小值。</p>
</td>
</tr>
</tbody>
</table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001456855048"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p425342585420"><a name="p425342585420"></a><a name="p425342585420"></a>void copyTo(faiss::IndexBinaryFlat *index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>基于一个已有的AscendIndexBinaryFlat拷贝到faiss::IndexBinaryFlat index, index原有资源被清空。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b194718510463"><a name="b194718510463"></a><a name="b194718510463"></a>faiss::IndexBinaryFlat *index</strong>：faiss::IndexBinaryFlat指针。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname16174215239"><a name="parmname16174215239"></a><a name="parmname16174215239"></a>“index”</span>需要为合法有效的IndexBinaryFlat指针，拷贝后的<span class="parmname" id="parmname160612019123"><a name="parmname160612019123"></a><a name="parmname160612019123"></a>“index”</span>资源由用户释放。</p>
</td>
</tr>
</tbody>
</table>

<a name="table19831553111512"></a>
<table><tbody><tr id="row1183118539158"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p10831135301517"><a name="p10831135301517"></a><a name="p10831135301517"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexBinaryIDMap *index) const;</p>
</td>
</tr>
<tr id="row1831153151517"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p158311553101511"><a name="p158311553101511"></a><a name="p158311553101511"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p68311653201512"><a name="p68311653201512"></a><a name="p68311653201512"></a>基于一个已有的AscendIndexBinaryFlat拷贝到faiss::IndexBinaryIDMap index, index原有资源被清空。</p>
</td>
</tr>
<tr id="row8831125312154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p10831145314156"><a name="p10831145314156"></a><a name="p10831145314156"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p108311153141519"><a name="p108311153141519"></a><a name="p108311153141519"></a><strong id="b172171419114612"><a name="b172171419114612"></a><a name="b172171419114612"></a>faiss::IndexBinaryIDMap *index</strong>：faiss::IndexBinaryIDMap指针。</p>
</td>
</tr>
<tr id="row11831195315154"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1683110535159"><a name="p1683110535159"></a><a name="p1683110535159"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p58323537152"><a name="p58323537152"></a><a name="p58323537152"></a>无</p>
</td>
</tr>
<tr id="row1083225391518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1683225314159"><a name="p1683225314159"></a><a name="p1683225314159"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1683235310157"><a name="p1683235310157"></a><a name="p1683235310157"></a>无</p>
</td>
</tr>
<tr id="row1983215318157"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11832175317159"><a name="p11832175317159"></a><a name="p11832175317159"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4832115316152"><a name="p4832115316152"></a><a name="p4832115316152"></a><span class="parmname" id="parmname13154121312411"><a name="parmname13154121312411"></a><a name="parmname13154121312411"></a>“index”</span>需要为合法有效的IndexBinaryIDMap指针，拷贝后的Index资源由用户释放。</p>
</td>
</tr>
</tbody>
</table>

## operator= 接口<a name="ZH-CN_TOPIC_0000001456535072"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16196201484419"><a name="p16196201484419"></a><a name="p16196201484419"></a>AscendIndexBinaryFlat &amp;operator = (const AscendIndexBinaryFlat &amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明AscendIndexBinaryFlat赋值构造函数为空，即AscendIndexBinaryFlat为不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b105275248101"><a name="b105275248101"></a><a name="b105275248101"></a>const AscendIndexBinaryFlat &amp;</strong>：常量AscendIndexBinaryFlat。</p>
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

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001506495769"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p723211434816"><a name="p723211434816"></a><a name="p723211434816"></a>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>删除底库中指定的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b94414427185"><a name="b94414427185"></a><a name="b94414427185"></a>const faiss::IDSelector &amp;sel</strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>返回成功删除（忽略非法ID）的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## reset接口<a name="ZH-CN_TOPIC_0000001456855028"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void reset() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>清空该AscendIndexBinaryFlat的底库向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a>无</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1911619471633"><a name="p1911619471633"></a><a name="p1911619471633"></a>无</p>
</td>
</tr>
</tbody>
</table>

## search接口<a id="ZH-CN_TOPIC_0000001456375288"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p18443123012413"><a name="p18443123012413"></a><a name="p18443123012413"></a>void search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels, const SearchParameters *params) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname9885102471911"><a name="parmname9885102471911"></a><a name="parmname9885102471911"></a>“k”</span>条特征的ID和对应距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><strong id="b6402181191915"><a name="b6402181191915"></a><a name="b6402181191915"></a>idx_t n</strong>：查询向量个数。</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><strong id="b8615513161912"><a name="b8615513161912"></a><a name="b8615513161912"></a>const uint8_t *x</strong>：查询向量。</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><strong id="b82719159198"><a name="b82719159198"></a><a name="b82719159198"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p191711443182415"><a name="p191711443182415"></a><a name="p191711443182415"></a><strong id="b10561743122410"><a name="b10561743122410"></a><a name="b10561743122410"></a>const SearchParameters *params：</strong>Faiss的可选参数，默认为<span class="parmvalue" id="parmvalue6623754182414"><a name="parmvalue6623754182414"></a><a name="parmvalue6623754182414"></a>“nullptr”</span>，暂不支持该参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b74651934121914"><a name="b74651934121914"></a><a name="b74651934121914"></a>int32_t *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname8247205815587"><a name="parmname8247205815587"></a><a name="parmname8247205815587"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b11761113620191"><a name="b11761113620191"></a><a name="b11761113620191"></a>idx_t *labels</strong>：<span class="parmname" id="parmname1016310116599"><a name="parmname1016310116599"></a><a name="parmname1016310116599"></a>“k”</span>个最近向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul38481656216"></a><a name="ul38481656216"></a><ul id="ul38481656216"><li>查询的特征向量数据<span class="parmname" id="parmname1860145416011"><a name="parmname1860145416011"></a><a name="parmname1860145416011"></a>“x”</span>的长度应该为dims/8 * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，<span class="parmname" id="parmname1180921315209"><a name="parmname1180921315209"></a><a name="parmname1180921315209"></a>“distances”</span>以及<span class="parmname" id="parmname6637916162017"><a name="parmname6637916162017"></a><a name="parmname6637916162017"></a>“labels”</span>的长度应该为<strong id="b7409174322613"><a name="b7409174322613"></a><a name="b7409174322613"></a>k</strong> * <strong id="b17392135042613"><a name="b17392135042613"></a><a name="b17392135042613"></a>n</strong>，否则可能会出现越界读写的情况，引起程序的崩溃。</li><li>0 &lt; n ≤ 1e9，0 &lt; k ≤1e5（n ≤ 1e9的限制远超过实际可用资源，请用户根据业务场景选择合适的查询向量个数）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1659211341612"></a>
<table><tbody><tr id="row1259231351612"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.1.1"><p id="p11592161301616"><a name="p11592161301616"></a><a name="p11592161301616"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.1.1 "><p id="p17215919132819"><a name="p17215919132819"></a><a name="p17215919132819"></a>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;</p>
</td>
</tr>
<tr id="row8592513191612"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.2.1"><p id="p859216134169"><a name="p859216134169"></a><a name="p859216134169"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.2.1 "><p id="p1617317122911"><a name="p1617317122911"></a><a name="p1617317122911"></a>特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID和对应距离。用于入库特征为二进制特征，检索特征为float类型的检索方式。</p>
</td>
</tr>
<tr id="row8592121311162"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.3.1"><p id="p5592151320162"><a name="p5592151320162"></a><a name="p5592151320162"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.3.1 "><p id="p142581642102914"><a name="p142581642102914"></a><a name="p142581642102914"></a><strong id="b36001143196"><a name="b36001143196"></a><a name="b36001143196"></a>idx_t n</strong>：查询向量个数。</p>
<p id="p1025813425296"><a name="p1025813425296"></a><a name="p1025813425296"></a><strong id="b525218718197"><a name="b525218718197"></a><a name="b525218718197"></a>const float *x</strong>：查询向量。</p>
<p id="p1825814422294"><a name="p1825814422294"></a><a name="p1825814422294"></a><strong id="b12147951917"><a name="b12147951917"></a><a name="b12147951917"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
</td>
</tr>
<tr id="row19592181320167"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.4.1"><p id="p9592111318169"><a name="p9592111318169"></a><a name="p9592111318169"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.4.1 "><p id="p66920493291"><a name="p66920493291"></a><a name="p66920493291"></a><strong id="b1785151311912"><a name="b1785151311912"></a><a name="b1785151311912"></a>float *distances</strong>：查询向量与距离最近的前“k”个向量间的距离值。</p>
<p id="p14692144913291"><a name="p14692144913291"></a><a name="p14692144913291"></a><strong id="b133591761916"><a name="b133591761916"></a><a name="b133591761916"></a>idx_t *labels</strong>：“k”个最近向量的ID。</p>
</td>
</tr>
<tr id="row6592171319163"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.5.1"><p id="p45921313161611"><a name="p45921313161611"></a><a name="p45921313161611"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.5.1 "><p id="p220841917285"><a name="p220841917285"></a><a name="p220841917285"></a>无</p>
</td>
</tr>
<tr id="row19593513111612"><th class="firstcol" valign="top" width="19.55%" id="mcps1.1.3.6.1"><p id="p459391311616"><a name="p459391311616"></a><a name="p459391311616"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.45%" headers="mcps1.1.3.6.1 "><a name="ul127218309196"></a><a name="ul127218309196"></a><ul id="ul127218309196"><li>查询的特征向量数据“x”的长度应该为dims * n，“distances”以及“labels”的长度应该为k * n，否则可能会出现越界读写的情况，引起程序的崩溃。</li><li>0 &lt; n ≤ 1e9，0 &lt; k ≤1e5（n ≤ 1e9的限制远超过实际可用资源，请用户根据业务场景选择合适的查询向量个数）。</li></ul>
</td>
</tr>
</tbody>
</table>

## setRemoveFast接口<a name="ZH-CN_TOPIC_0000002024780673"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>static void setRemoveFast(bool removeFast);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p347711919317"><a name="p347711919317"></a><a name="p347711919317"></a>设置是否快速删除底库中的向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b943925813513"><a name="b943925813513"></a><a name="b943925813513"></a>bool removeFast</strong>：设置为<span class="parmvalue" id="parmvalue3715773388"><a name="parmvalue3715773388"></a><a name="parmvalue3715773388"></a>“true”</span>表示使用快速删除；<span class="parmvalue" id="parmvalue1474241143813"><a name="parmvalue1474241143813"></a><a name="parmvalue1474241143813"></a>“false”</span>表示不使用。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>快速删除会提高删除底库的性能，但是会稍微降低添加底库的性能。不调用该接口时默认不使用快速删除。该接口只能调用一次，且需要在构造index对象前使用。</p>
</td>
</tr>
</tbody>
</table>
