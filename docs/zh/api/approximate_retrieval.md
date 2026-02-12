
## 近似检索<a name="ZH-CN_TOPIC_0000001482524834"></a>

### AscendIndexBinaryFlat<a name="ZH-CN_TOPIC_0000001506334701"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456694988"></a>

AscendIndexBinaryFlat类继承自Faiss的IndexBinary类，用于二值化特征检索。

仅支持Atlas 推理系列产品。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

#### add接口<a name="ZH-CN_TOPIC_0000001456854896"></a>

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

> [!NOTE] 说明 
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。

#### add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001506414809"></a>

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

#### AscendIndexBinaryFlat接口<a name="ZH-CN_TOPIC_0000001456535056"></a>

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

#### \~AscendIndexBinaryFlat接口<a name="ZH-CN_TOPIC_0000001506495917"></a>

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

#### copyFrom接口<a name="ZH-CN_TOPIC_0000001506414941"></a>

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

#### copyTo接口<a name="ZH-CN_TOPIC_0000001456855048"></a>

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

#### operator= 接口<a name="ZH-CN_TOPIC_0000001456535072"></a>

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

#### remove\_ids接口<a name="ZH-CN_TOPIC_0000001506495769"></a>

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

#### reset接口<a name="ZH-CN_TOPIC_0000001456855028"></a>

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

#### search接口<a id="ZH-CN_TOPIC_0000001456375288"></a>

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

#### setRemoveFast接口<a name="ZH-CN_TOPIC_0000002024780673"></a>

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

### AscendIndexBinaryFlatConfig<a name="ZH-CN_TOPIC_0000001506495777"></a>

AscendIndexBinaryFlat需要使用对应的AscendIndexBinaryFlatConfig执行对应资源的初始化，配置执行检索过程中的硬件资源“devices”和预置的内存池大小“resources”。

- AscendIndexBinaryFlat仅支持单个昇腾AI处理器的Atlas 推理系列产品，依赖AICPU算子和BinaryFlat算子，请参考[自定义算子介绍](../user_guide.md#自定义算子介绍)生成对应算子。
- AscendIndexBinaryFlat仅支持标准态部署方式。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|deviceList|std::vector\<int>|Device侧设备ID。AscendIndexBinaryFlat类仅支持单个Atlas 推理系列产品的加速卡。|
|resourceSize|int64_t|Device侧内存池大小，单位为字节，默认参数值为1024MB，合法范围为[1024*1024*1024, 32*1024*1024*1024]，10million底库推荐申请5GB。|

**接口说明<a name="section108610580175"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1551916503464"><a name="p1551916503464"></a><a name="p1551916503464"></a>AscendIndexBinaryFlatConfig() = default;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>默认构造函数，默认devices为{ 0 }，使用第0个<span id="ph79732210444"><a name="ph79732210444"></a><a name="ph79732210444"></a>昇腾AI处理器</span>进行计算，默认resources为1024MB。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>AscendIndexBinaryFlat仅支持单个<span id="ph13714132410186"><a name="ph13714132410186"></a><a name="ph13714132410186"></a>昇腾AI处理器</span>的<span id="ph87140243189"><a name="ph87140243189"></a><a name="ph87140243189"></a>Atlas 推理系列产品</span>，如果第0个<span id="ph112871028144417"><a name="ph112871028144417"></a><a name="ph112871028144417"></a>昇腾AI处理器</span>不可用则无法使用默认构造。</p>
</td>
</tr>
</tbody>
</table>

<a name="table092314378186"></a>
<table><tbody><tr id="row6923173719182"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18923337131816"><a name="p18923337131816"></a><a name="p18923337131816"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3686859135116"><a name="p3686859135116"></a><a name="p3686859135116"></a>AscendIndexBinaryFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row1692315371180"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p19923153751814"><a name="p19923153751814"></a><a name="p19923153751814"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p14661924131015"><a name="p14661924131015"></a><a name="p14661924131015"></a><span class="parmname" id="parmname367823310315"><a name="parmname367823310315"></a><a name="parmname367823310315"></a>“devices”</span>使用initializer_list的构造函数。</p>
</td>
</tr>
<tr id="row092353751820"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1292333771814"><a name="p1292333771814"></a><a name="p1292333771814"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p192393741812"><a name="p192393741812"></a><a name="p192393741812"></a><strong id="b7745577018"><a name="b7745577018"></a><a name="b7745577018"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID，对于该类，仅支持单Device，即<span class="parmname" id="parmname69151954163010"><a name="parmname69151954163010"></a><a name="parmname69151954163010"></a>“devices”</span>长度为<span class="parmvalue" id="parmvalue5243157143011"><a name="parmvalue5243157143011"></a><a name="parmvalue5243157143011"></a>“1”</span>。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b120518131001"><a name="b120518131001"></a><a name="b120518131001"></a>int64_t resources：</strong>预置的内存池大小，默认值为1024MB。</p>
</td>
</tr>
<tr id="row4923163715183"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1692313375186"><a name="p1692313375186"></a><a name="p1692313375186"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15923173719188"><a name="p15923173719188"></a><a name="p15923173719188"></a>无</p>
</td>
</tr>
<tr id="row392317375180"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p119239370187"><a name="p119239370187"></a><a name="p119239370187"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p109231372187"><a name="p109231372187"></a><a name="p109231372187"></a>无</p>
</td>
</tr>
<tr id="row119241937191814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11924193715185"><a name="p11924193715185"></a><a name="p11924193715185"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5609290011"></a><a name="ul5609290011"></a><ul id="ul5609290011"><li><span class="parmname" id="parmname869315371603"><a name="parmname869315371603"></a><a name="parmname869315371603"></a>“devices”</span>需要为合法有效不重复的设备ID，长度为1。</li><li><span class="parmname" id="parmname95620401905"><a name="parmname95620401905"></a><a name="parmname95620401905"></a>“resources”</span>合法范围为[1024*1024*1024, 32*1024*1024*1024]，10million底库推荐申请5GB。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1743710521181"></a>
<table><tbody><tr id="row18437752161818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2437175281813"><a name="p2437175281813"></a><a name="p2437175281813"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19298152493214"><a name="p19298152493214"></a><a name="p19298152493214"></a>AscendIndexBinaryFlatConfig(std::vector&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row1243755211815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p15437452131811"><a name="p15437452131811"></a><a name="p15437452131811"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10437125271814"><a name="p10437125271814"></a><a name="p10437125271814"></a><span class="parmname" id="parmname043714527182"><a name="parmname043714527182"></a><a name="parmname043714527182"></a>“devices”</span>使用vector的构造函数。</p>
</td>
</tr>
<tr id="row843735251817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p8437152171817"><a name="p8437152171817"></a><a name="p8437152171817"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p74371152181812"><a name="p74371152181812"></a><a name="p74371152181812"></a><strong id="b76053103337"><a name="b76053103337"></a><a name="b76053103337"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID，对于该类，仅支持单Device，即<span class="parmname" id="parmname943712527186"><a name="parmname943712527186"></a><a name="parmname943712527186"></a>“devices”</span>长度为<span class="parmvalue" id="parmvalue1343713529189"><a name="parmvalue1343713529189"></a><a name="parmvalue1343713529189"></a>“1”</span>。</p>
<p id="p843710522187"><a name="p843710522187"></a><a name="p843710522187"></a><strong id="b175341016133314"><a name="b175341016133314"></a><a name="b175341016133314"></a>int64_t resources</strong>：预置的内存池大小，默认值为1024MB。</p>
</td>
</tr>
<tr id="row243775261813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9437452171817"><a name="p9437452171817"></a><a name="p9437452171817"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p4437352101816"><a name="p4437352101816"></a><a name="p4437352101816"></a>无</p>
</td>
</tr>
<tr id="row17437195218187"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p743785215182"><a name="p743785215182"></a><a name="p743785215182"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p15437852171815"><a name="p15437852171815"></a><a name="p15437852171815"></a>无</p>
</td>
</tr>
<tr id="row143717524181"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1743735241812"><a name="p1743735241812"></a><a name="p1743735241812"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul144378526181"></a><a name="ul144378526181"></a><ul id="ul144378526181"><li><span class="parmname" id="parmname1643775216187"><a name="parmname1643775216187"></a><a name="parmname1643775216187"></a>“devices”</span>需要为合法有效不重复的设备ID，长度为1。</li><li><span class="parmname" id="parmname1437115251816"><a name="parmname1437115251816"></a><a name="parmname1437115251816"></a>“resources”</span>合法范围为[1024*1024*1024, 32*1024*1024*1024]，10million底库推荐申请5GB。</li></ul>
</td>
</tr>
</tbody>
</table>

### AscendIndexIVF<a name="ZH-CN_TOPIC_0000001456375220"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001506334721"></a>

AscendIndexIVF作为特征检索组件中的采用IVF的Index的基类，为特征检索中的其他的IVF的Index定义接口。

对于IVF系列算法，在Atlas 300I Duo 推理卡上的线性增长取决于距离计算的运算量在整个search过程的占比。相较于其他计算类型，只有距离计算的运算量可以均分到多个运算单元，所以在大batch和nprobe较大的场景下，线性增长度更好，而小batch和nprobe较小的场景下线性增长度则较差。

> [!NOTE] 说明 
> IVF系列算法，应遵循nlist \* 2MB +  **resourceSize**  < NPU侧内存的规则，避免程序运行时申请内存失败，例如：npu卡上内存为64GB，则nlist应小于32768，32768 \* 2MB = 64GB，程序运行可能超出NPU内存大小。造成该限制的原因是目前检索业务申请内存的规则为大页内存优先，大页内存申请粒度为2MB。当nlist个桶内都有数据时，向硬件申请内存时，硬件分配的内存按照2MB的粒度对齐。（其中**resourceSize**是AscendIndexIVFConfig中用户指定的共享内存大小，默认128MB）

#### AscendIndexIVF接口<a name="ZH-CN_TOPIC_0000001506414821"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p43030218474"><a name="p43030218474"></a><a name="p43030218474"></a>AscendIndexIVF(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFConfig config = AscendIndexIVFConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVF的构造函数，生成AscendIndexIVF，此时根据<span class="parmname" id="parmname445472621618"><a name="parmname445472621618"></a><a name="parmname445472621618"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b1351634291614"><a name="b1351634291614"></a><a name="b1351634291614"></a>int dims</strong>：AscendIndexIVF管理的一组特征向量的维度。</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b83901946121615"><a name="b83901946121615"></a><a name="b83901946121615"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型，当前支持<span class="parmvalue" id="parmvalue1987818211718"><a name="parmvalue1987818211718"></a><a name="parmvalue1987818211718"></a>“faiss::MetricType::METRIC_L2”</span>以及<span class="parmvalue" id="parmvalue189126111720"><a name="parmvalue189126111720"></a><a name="parmvalue189126111720"></a>“faiss::MetricType::METRIC_INNER_PRODUCT”</span>。</p>
<p id="p1465818245133"><a name="p1465818245133"></a><a name="p1465818245133"></a><strong id="b14852184813167"><a name="b14852184813167"></a><a name="b14852184813167"></a>int nlist</strong>：聚类中心的个数，与算子生成脚本中的<span class="parmname" id="parmname6984639131616"><a name="parmname6984639131616"></a><a name="parmname6984639131616"></a>“coarse_centroid_num”</span>参数对应。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b12540950121613"><a name="b12540950121613"></a><a name="b12540950121613"></a>AscendIndexIVFConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p526919571274"><a name="p526919571274"></a><a name="p526919571274"></a>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}</p>
</td>
</tr>
</tbody>
</table>

<a name="table9624174810199"></a>
<table><tbody><tr id="row16624948141916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2624848121918"><a name="p2624848121918"></a><a name="p2624848121918"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p8445440165114"><a name="p8445440165114"></a><a name="p8445440165114"></a>AscendIndexIVF(const AscendIndexIVF&amp;) = delete;</p>
</td>
</tr>
<tr id="row5624348101916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p76243485193"><a name="p76243485193"></a><a name="p76243485193"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1862414482192"><a name="p1862414482192"></a><a name="p1862414482192"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row562484861911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p86244488196"><a name="p86244488196"></a><a name="p86244488196"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b684232810173"><a name="b684232810173"></a><a name="b684232810173"></a>const AscendIndexIVF&amp;</strong>：常量AscendIndexIVF。</p>
</td>
</tr>
<tr id="row6624174811193"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p12624114811194"><a name="p12624114811194"></a><a name="p12624114811194"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p86248482195"><a name="p86248482195"></a><a name="p86248482195"></a>无</p>
</td>
</tr>
<tr id="row1962474811911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p14624184831911"><a name="p14624184831911"></a><a name="p14624184831911"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p196241548171912"><a name="p196241548171912"></a><a name="p196241548171912"></a>无</p>
</td>
</tr>
<tr id="row19624194810195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15624134819195"><a name="p15624134819195"></a><a name="p15624134819195"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### \~AscendIndexIVF接口<a name="ZH-CN_TOPIC_0000001506334765"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexIVF();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVF的析构函数，销毁AscendIndexIVF对象，释放资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
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

#### copyFrom接口<a name="ZH-CN_TOPIC_0000001506334601"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1215384082314"><a name="p1215384082314"></a><a name="p1215384082314"></a>void copyFrom(const faiss::IndexIVF* index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVF基于一个已有的<span class="parmname" id="parmname19326113201714"><a name="parmname19326113201714"></a><a name="parmname19326113201714"></a>“index”</span>拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b762812176182"><a name="b762812176182"></a><a name="b762812176182"></a>const faiss::IndexIVF* index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1256192919201"></a><a name="ul1256192919201"></a><ul id="ul1256192919201"><li><span class="parmname" id="parmname08665451200"><a name="parmname08665451200"></a><a name="parmname08665451200"></a>“index”</span>需要为合法有效的CPU Index指针。</li><li>该<span class="parmname" id="parmname2511188163815"><a name="parmname2511188163815"></a><a name="parmname2511188163815"></a>“index”</span>的probe必须大于0且小于或等于nlist。</li></ul>
</td>
</tr>
</tbody>
</table>

#### copyTo接口<a name="ZH-CN_TOPIC_0000001506615113"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19222052125319"><a name="p19222052125319"></a><a name="p19222052125319"></a>void copyTo(faiss::IndexIVF* index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>将AscendIndexIVF的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b2873102510208"><a name="b2873102510208"></a><a name="b2873102510208"></a>faiss::IndexIVF* index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname3171340162014"><a name="parmname3171340162014"></a><a name="parmname3171340162014"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

#### getNumLists接口<a name="ZH-CN_TOPIC_0000001506614893"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>int getNumLists() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVF的nlist数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>AscendIndexIVF的nlist数。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### getNumProbes接口<a name="ZH-CN_TOPIC_0000001456534948"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>int getNumProbes() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVF的nprobe数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>AscendIndexIVF的nprobe数。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### getListCodesAndIds接口<a name="ZH-CN_TOPIC_0000001456854940"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVF的nlist中的特定nlistId上的特征向量和对应ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b234205219283"><a name="b234205219283"></a><a name="b234205219283"></a>int listId</strong>：AscendIndexIVF的nlist中的特定nlistId。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p812472610226"><a name="p812472610226"></a><a name="p812472610226"></a><strong id="b8752144372820"><a name="b8752144372820"></a><a name="b8752144372820"></a>std::vector&lt;uint8_t&gt;&amp; codes</strong>：AscendIndexIVF的nlist中的特定nlistId上的特征向量。</p>
<p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b198817462287"><a name="b198817462287"></a><a name="b198817462287"></a>std::vector&lt;ascend_idx_t&gt;&amp; ids</strong>：AscendIndexIVF的nlist中的特定nlistId上的特征向量ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>0 ≤ listId＜nlist</p>
</td>
</tr>
</tbody>
</table>

#### getListLength接口<a name="ZH-CN_TOPIC_0000001506614973"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual uint32_t getListLength(int listId) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVF的nlist中的特定nlistId上的长度。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b121461446192713"><a name="b121461446192713"></a><a name="b121461446192713"></a>int listId</strong>：AscendIndexIVF的nlist中的特定nlistId。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>AscendIndexIVF的nlist中的特定nlistId上的长度。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>0 ≤ listId＜nlist</p>
</td>
</tr>
</tbody>
</table>

#### operator = 接口<a name="ZH-CN_TOPIC_0000001506495837"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10442427125319"><a name="p10442427125319"></a><a name="p10442427125319"></a>AscendIndexIVF&amp; operator=(const AscendIndexIVF&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b6551145181714"><a name="b6551145181714"></a><a name="b6551145181714"></a>const AscendIndexIVF&amp;</strong>：常量AscendIndexIVF。</p>
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

#### reclaimMemory接口<a name="ZH-CN_TOPIC_0000001506615049"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>size_t reclaimMemory() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>在保证底库数量不变的情况下，缩减底库占用的内存，继承AscendIndex中的接口并提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>缩减的内存大小，单位为Byte。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### reserveMemory接口<a name="ZH-CN_TOPIC_0000001506334617"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>void reserveMemory(size_t numVecs) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p169016017519"><a name="p169016017519"></a><a name="p169016017519"></a>在建立底库前为底库申请预留内存的抽象接口，继承AscendIndex中的接口并提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a><strong id="b551710192252"><a name="b551710192252"></a><a name="b551710192252"></a>size_t numVecs</strong>：申请预留内存的底库数量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1760910169348"><a name="p1760910169348"></a><a name="p1760910169348"></a>单卡环境时：0&lt;<span class="parmname" id="parmname9414456173417"><a name="parmname9414456173417"></a><a name="parmname9414456173417"></a>“numVecs”</span>≤ <span class="parmvalue" id="parmvalue15320144218357"><a name="parmvalue15320144218357"></a><a name="parmvalue15320144218357"></a>“2e8”</span>；多卡环境时：0 &lt; <span class="parmname" id="parmname19902911113511"><a name="parmname19902911113511"></a><a name="parmname19902911113511"></a>“numVecs”</span>≤ <span class="parmvalue" id="parmvalue1524262823812"><a name="parmvalue1524262823812"></a><a name="parmvalue1524262823812"></a>“1e9”</span>(<span class="parmname" id="parmname4328121903513"><a name="parmname4328121903513"></a><a name="parmname4328121903513"></a>“numVecs”</span> &divide; 卡的数量需小于<span class="parmvalue" id="parmvalue2075815584359"><a name="parmvalue2075815584359"></a><a name="parmvalue2075815584359"></a>“2e8”</span>)；超出限制会抛出异常，停止运行。</p>
</td>
</tr>
</tbody>
</table>

#### reset接口<a name="ZH-CN_TOPIC_0000001506414685"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void reset() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>清空该AscendIndexIVF的底库向量。</p>
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

#### setNumProbes接口<a name="ZH-CN_TOPIC_0000001506614937"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual void setNumProbes(int nprobes);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>设置当前的AscendIndexIVF的nprobe数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b16217144619214"><a name="b16217144619214"></a><a name="b16217144619214"></a>int nprobes</strong>：AscendIndexIVF的nprobe数。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>0 &lt; nprobe ≤ nlist</p>
</td>
</tr>
</tbody>
</table>

### AscendIndexIVFConfig<a name="ZH-CN_TOPIC_0000001456535024"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456695128"></a>

AscendIndexIVF需要使用对应的AscendIndexIVFConfig执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|flatConfig|AscendIndexConfig|参数配置对象。|
|useKmeansPP|bool|是否使用NPU加速IVF聚类过程。|
|cp|ClusteringParameters|聚类相关参数，具体可以参见Faiss相关接口说明。不建议修改此参数，其中训练迭代次数参数默认为16。迭代次数设置过大，会显著增加训练时长。|

> [!NOTE] 说明
> 
> AscendIndexIVFSQConfig继承于[AscendIndexConfig](./full_retrieval.md#ascendindexconfig)。

#### AscendIndexIVFConfig接口<a name="ZH-CN_TOPIC_0000001506334629"></a>

<a name="table1319620316150"></a>
<table><tbody><tr id="row19196173161512"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p8196736151"><a name="p8196736151"></a><a name="p8196736151"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p91961331157"><a name="p91961331157"></a><a name="p91961331157"></a>inline AscendIndexIVFConfig();</p>
</td>
</tr>
<tr id="row519612310152"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p131967381517"><a name="p131967381517"></a><a name="p131967381517"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p319616311155"><a name="p319616311155"></a><a name="p319616311155"></a>默认构造函数，默认devices为{0}，使用第0个<span id="ph5196035156"><a name="ph5196035156"></a><a name="ph5196035156"></a>昇腾AI处理器</span>进行计算，默认resources为128MB，默认useKmeansPP为<span class="parmvalue" id="parmvalue319613191512"><a name="parmvalue319613191512"></a><a name="parmvalue319613191512"></a>“false”</span>。</p>
</td>
</tr>
<tr id="row191967381510"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1719683121520"><a name="p1719683121520"></a><a name="p1719683121520"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p31961435151"><a name="p31961435151"></a><a name="p31961435151"></a>无</p>
</td>
</tr>
<tr id="row191966331518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1719613141519"><a name="p1719613141519"></a><a name="p1719613141519"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1319615331515"><a name="p1319615331515"></a><a name="p1319615331515"></a>无</p>
</td>
</tr>
<tr id="row1019673161519"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p14196173191518"><a name="p14196173191518"></a><a name="p14196173191518"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p719603201510"><a name="p719603201510"></a><a name="p719603201510"></a>无</p>
</td>
</tr>
<tr id="row519633171513"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2197163171510"><a name="p2197163171510"></a><a name="p2197163171510"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4197143151510"><a name="p4197143151510"></a><a name="p4197143151510"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table3725347611"></a>
<table><tbody><tr id="row137251141265"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1372544561"><a name="p1372544561"></a><a name="p1372544561"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4706533988"><a name="p4706533988"></a><a name="p4706533988"></a>inline AscendIndexIVFConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row0725941369"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p87251143611"><a name="p87251143611"></a><a name="p87251143611"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFConfig的构造函数，生成AscendIndexIVFConfig，此时根据<span class="parmname" id="parmname18510024575"><a name="parmname18510024575"></a><a name="parmname18510024575"></a>“devices”</span>中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源，配置资源池大小并设置默认迭代数。</p>
</td>
</tr>
<tr id="row872516411614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p157251441762"><a name="p157251441762"></a><a name="p157251441762"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1172515411612"><a name="p1172515411612"></a><a name="p1172515411612"></a><strong id="b74801235171213"><a name="b74801235171213"></a><a name="b74801235171213"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b851894117126"><a name="b851894117126"></a><a name="b851894117126"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname15799101092219"><a name="parmname15799101092219"></a><a name="parmname15799101092219"></a>“IVF_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row13725184068"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p0725844620"><a name="p0725844620"></a><a name="p0725844620"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p37251747615"><a name="p37251747615"></a><a name="p37251747615"></a>无</p>
</td>
</tr>
<tr id="row19725104260"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p9725446613"><a name="p9725446613"></a><a name="p9725446613"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p77251043619"><a name="p77251043619"></a><a name="p77251043619"></a>无</p>
</td>
</tr>
<tr id="row7725641869"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p272634161"><a name="p272634161"></a><a name="p272634161"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5591115071213"></a><a name="ul5591115071213"></a><ul id="ul5591115071213"><li><span class="parmname" id="parmname14872125571216"><a name="parmname14872125571216"></a><a name="parmname14872125571216"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname15293195861219"><a name="parmname15293195861219"></a><a name="parmname15293195861219"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue51241072364"><a name="parmvalue51241072364"></a><a name="parmvalue51241072364"></a>“-1”</span>时，Device侧<span id="ph8157732103"><a name="ph8157732103"></a><a name="ph8157732103"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table745471811619"></a>
<table><tbody><tr id="row445418187618"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p145417181561"><a name="p145417181561"></a><a name="p145417181561"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172151146898"><a name="p172151146898"></a><a name="p172151146898"></a>inline AscendIndexIVFConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row845519181169"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p164551418362"><a name="p164551418362"></a><a name="p164551418362"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1445513182614"><a name="p1445513182614"></a><a name="p1445513182614"></a>AscendIndexIVFConfig的构造函数，生成AscendIndexIVFConfig，此时根据<span class="parmname" id="parmname15821822101518"><a name="parmname15821822101518"></a><a name="parmname15821822101518"></a>“devices”</span>中配置的值设置Device侧<span id="ph663911476576"><a name="ph663911476576"></a><a name="ph663911476576"></a>昇腾AI处理器</span>资源，配置资源池大小并设置默认迭代数。</p>
</td>
</tr>
<tr id="row845512181667"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14551718264"><a name="p14551718264"></a><a name="p14551718264"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1945571813613"><a name="p1945571813613"></a><a name="p1945571813613"></a><strong id="b9403131414155"><a name="b9403131414155"></a><a name="b9403131414155"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p11455318966"><a name="p11455318966"></a><a name="p11455318966"></a><strong id="b132471122150"><a name="b132471122150"></a><a name="b132471122150"></a>int resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname20104122922215"><a name="parmname20104122922215"></a><a name="parmname20104122922215"></a>“IVF_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row12455718267"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p645513184613"><a name="p645513184613"></a><a name="p645513184613"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1545511816618"><a name="p1545511816618"></a><a name="p1545511816618"></a>无</p>
</td>
</tr>
<tr id="row11455318162"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p845511186617"><a name="p845511186617"></a><a name="p845511186617"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p104556181962"><a name="p104556181962"></a><a name="p104556181962"></a>无</p>
</td>
</tr>
<tr id="row17455118361"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p64551618167"><a name="p64551618167"></a><a name="p64551618167"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9168241111516"></a><a name="ul9168241111516"></a><ul id="ul9168241111516"><li><span class="parmname" id="parmname975454381516"><a name="parmname975454381516"></a><a name="parmname975454381516"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname8790174512156"><a name="parmname8790174512156"></a><a name="parmname8790174512156"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue8480143663618"><a name="parmvalue8480143663618"></a><a name="parmvalue8480143663618"></a>“-1”</span>时，Device侧<span id="ph245516181062"><a name="ph245516181062"></a><a name="ph245516181062"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>

#### SetDefaultClusteringConfig接口<a name="ZH-CN_TOPIC_0000001506495669"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172151146898"><a name="p172151146898"></a><a name="p172151146898"></a>inline void SetDefaultClusteringConfig();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7535131221216"><a name="p7535131221216"></a><a name="p7535131221216"></a>设置此时的AscendIndexIVF的迭代次数为默认值<span class="parmvalue" id="parmvalue694616391543"><a name="parmvalue694616391543"></a><a name="parmvalue694616391543"></a>“10”</span>。</p>
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

### AscendIndexIVFSP<a name="ZH-CN_TOPIC_0000001635576081"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001635815481"></a>

昇腾原生IVFSP检索算法，使用自研矩阵近似策略，压缩特征向量后存底库，并使用自研倒排链策略选取出最可能包含Ground Truth（真实）的底库，最后使用自研检索策略在倒排链过滤后的底库进行检索得到Top K向量结果。

AscendIndexIVFSP只支持标准态场景，且只支持Atlas 推理系列产品。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

#### add接口<a name="ZH-CN_TOPIC_0000001585895568"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void add(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>向底库中添加特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1794683517262"><a name="p1794683517262"></a><a name="p1794683517262"></a><strong id="b016584171514"><a name="b016584171514"></a><a name="b016584171514"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p1594616353266"><a name="p1594616353266"></a><a name="p1594616353266"></a><strong id="b12572953101710"><a name="b12572953101710"></a><a name="b12572953101710"></a>const float *x</strong>：添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1572065617233"></a><a name="ul1572065617233"></a><ul id="ul1572065617233"><li>指针<span class="parmname" id="parmname241211111910"><a name="parmname241211111910"></a><a name="parmname241211111910"></a>“x”</span>的长度应该为dims * <strong id="b99141183199"><a name="b99141183199"></a><a name="b99141183199"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>底库向量总数<span class="parmname" id="parmname913981814284"><a name="parmname913981814284"></a><a name="parmname913981814284"></a>“n”</span>通常大于0且小于1e9。</li><li>一次性add的数据量应该小于等于特征底库数据大小。</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE] 说明 
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。
>- add接口在小batch添加场景进行了性能优化，此场景根据数据集不同，精度会有所降低，建议在已有底库场景下用小batch添加。

#### add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001586055512"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>向底库中添加特征向量并指定对应的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p133861411291"><a name="p133861411291"></a><a name="p133861411291"></a><strong id="b1195072611184"><a name="b1195072611184"></a><a name="b1195072611184"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p338634162915"><a name="p338634162915"></a><a name="p338634162915"></a><strong id="b551412342183"><a name="b551412342183"></a><a name="b551412342183"></a>const float *x</strong>：添加进底库的特征向量。</p>
<p id="p17386184182913"><a name="p17386184182913"></a><a name="p17386184182913"></a><strong id="b42592374181"><a name="b42592374181"></a><a name="b42592374181"></a>const idx_t *ids</strong>：添加进底库的特征向量对应的ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p152321022122915"><a name="p152321022122915"></a><a name="p152321022122915"></a>指针<span class="parmname" id="parmname241211111910"><a name="parmname241211111910"></a><a name="parmname241211111910"></a>“x”</span>的长度应该为dims * <strong id="b99141183199"><a name="b99141183199"></a><a name="b99141183199"></a>n</strong>，指针<span class="parmname" id="parmname186814213323"><a name="parmname186814213323"></a><a name="parmname186814213323"></a>“ids”</span>的长度应为<span class="parmname" id="parmname492873714325"><a name="parmname492873714325"></a><a name="parmname492873714325"></a>“n”</span>，否则可能出现越界读写错误并引起程序崩溃。用户需要根据自己的业务场景，保证<span class="parmname" id="parmname39625451329"><a name="parmname39625451329"></a><a name="parmname39625451329"></a>“ids”</span>的合法性，如底库中存在重复的ID，检索结果中的<span class="parmname" id="parmname1770501133315"><a name="parmname1770501133315"></a><a name="parmname1770501133315"></a>“label”</span>将无法对应具体的底库向量。</p>
<p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a><span class="parmname" id="parmname31332191105"><a name="parmname31332191105"></a><a name="parmname31332191105"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE] 说明 
> add\_with\_ids接口在小batch添加场景进行了性能优化，此场景根据数据集不同，精度会有所降低，建议在已有底库场景下用小batch添加。

#### AscendIndexIVFSP接口<a name="ZH-CN_TOPIC_0000001585736168"></a>

> [!NOTE] 说明 
>将参数“config”传递给函数前，请根据实际情况先设置conf.handleBatch、conf.nprobe、conf.searchListSize的值（字段描述参考[公共参数](#ZH-CN_TOPIC_0000001635696057)）。
>其中conf.handleBatch、conf.searchListSize值需与[IVFSP](../user_guide.md#ivfsp)业务算子模型文件生成中的<nprobe handle batch\>、<search list size\>保持一致。
>conf.filterable（继承自[AscendIndexConfig](./full_retrieval.md#ascendindexconfig) ）默认为“false”，如果要使用search\_with\_filter\(\)接口，需设置**conf.filterable = true**。“conf.filterable”设置为“true”将在NPU卡上存储额外的信息，消耗更多的NPU卡上内存。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>AscendIndexIVFSP的构造函数，根据<span class="parmname" id="parmname18661204514127"><a name="parmname18661204514127"></a><a name="parmname18661204514127"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1561263817143"><a name="p1561263817143"></a><a name="p1561263817143"></a><strong id="b19612183881419"><a name="b19612183881419"></a><a name="b19612183881419"></a>int dims</strong>：AscendIndexIVFSP管理的一组特征向量的维度。</p>
<p id="p1661213891412"><a name="p1661213891412"></a><a name="p1661213891412"></a><strong id="b48633716167"><a name="b48633716167"></a><a name="b48633716167"></a>int nonzeroNum</strong>：特征向量压缩降维后非零维度个数。</p>
<p id="p17612133810143"><a name="p17612133810143"></a><a name="p17612133810143"></a><strong id="b761216381145"><a name="b761216381145"></a><a name="b761216381145"></a>int nlist</strong>：聚类中心的个数，与<a href="../user_guide.md#ivfsp">IVFSP业务算子模型文件生成</a>中的&lt;centroid num&gt;参数值对应。</p>
<p id="p166121738111411"><a name="p166121738111411"></a><a name="p166121738111411"></a><strong id="b249718368317"><a name="b249718368317"></a><a name="b249718368317"></a>const char *codeBookPath</strong>：IVFSP使用的码本文件路径。</p>
<p id="p1061210384146"><a name="p1061210384146"></a><a name="p1061210384146"></a><strong id="b451311429018"><a name="b451311429018"></a><a name="b451311429018"></a>faiss::ScalarQuantizer::QuantizerType qType</strong>：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
<p id="p2617038181412"><a name="p2617038181412"></a><a name="p2617038181412"></a><strong id="b1861763817147"><a name="b1861763817147"></a><a name="b1861763817147"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。当前<span class="parmname" id="parmname858915211735"><a name="parmname858915211735"></a><a name="parmname858915211735"></a>“faiss::MetricType metric”</span>仅支持<span class="parmvalue" id="parmvalue1483205813413"><a name="parmvalue1483205813413"></a><a name="parmvalue1483205813413"></a>“METRIC_L2”</span>。</p>
<p id="p1161723812145"><a name="p1161723812145"></a><a name="p1161723812145"></a><strong id="b11550924103117"><a name="b11550924103117"></a><a name="b11550924103117"></a>AscendIndexIVFSPConfig</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul719511175195"></a><a name="ul719511175195"></a><ul id="ul719511175195"><li>训练生成码本时的&lt;dim&gt;、&lt;nonzero num&gt;、&lt;centroid num&gt; 值应该与此函数的参数<span class="parmname" id="parmname838317156357"><a name="parmname838317156357"></a><a name="parmname838317156357"></a>“dims”</span>、<span class="parmname" id="parmname15618111743514"><a name="parmname15618111743514"></a><a name="parmname15618111743514"></a>“nonzeroNum”</span>、<span class="parmname" id="parmname10133020143510"><a name="parmname10133020143510"></a><a name="parmname10133020143510"></a>“nlist”</span>对应。</li><li><span class="parmname" id="parmname35157306285"><a name="parmname35157306285"></a><a name="parmname35157306285"></a>“codeBookPath”</span>加载的码本应该与此函数的参数<span class="parmname" id="parmname13115348162817"><a name="parmname13115348162817"></a><a name="parmname13115348162817"></a>“dims”</span>、<span class="parmname" id="parmname164983568286"><a name="parmname164983568286"></a><a name="parmname164983568286"></a>“nonzeroNum”</span>、<span class="parmname" id="parmname58341316295"><a name="parmname58341316295"></a><a name="parmname58341316295"></a>“nlist”</span>对应，且程序的执行用户是码本文件的属主；且码本文件不能为软链接。</li><li>当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。</li><li><span class="parmname" id="parmname178225132016"><a name="parmname178225132016"></a><a name="parmname178225132016"></a>“nonzeroNum”</span>需为16的倍数且小于等于min(128, dims)。</li><li>metric ∈ {faiss::MetricType::METRIC_L2}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table49022324218"></a>
<table><tbody><tr id="row199021732102118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p79020325216"><a name="p79020325216"></a><a name="p79020325216"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7378142174712"><a name="p7378142174712"></a><a name="p7378142174712"></a>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &amp;codeBookSharedIdx, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</p>
</td>
</tr>
<tr id="row190216323214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p13902153202111"><a name="p13902153202111"></a><a name="p13902153202111"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1390393214211"><a name="p1390393214211"></a><a name="p1390393214211"></a>AscendIndexIVFSP的构造函数，根据<span class="parmname" id="parmname19903532122113"><a name="parmname19903532122113"></a><a name="parmname19903532122113"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row3903113252110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p89039324212"><a name="p89039324212"></a><a name="p89039324212"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p790310324213"><a name="p790310324213"></a><a name="p790310324213"></a><strong id="b209032032182118"><a name="b209032032182118"></a><a name="b209032032182118"></a>int dims</strong>：AscendIndexIVFSP管理的一组特征向量的维度。</p>
<p id="p1590310322217"><a name="p1590310322217"></a><a name="p1590310322217"></a><strong id="b99031324210"><a name="b99031324210"></a><a name="b99031324210"></a>int nonzeroNum</strong>：特征向量压缩降维后非零维度个数。</p>
<p id="p490373216218"><a name="p490373216218"></a><a name="p490373216218"></a><strong id="b390383218212"><a name="b390383218212"></a><a name="b390383218212"></a>int nlist</strong>：聚类中心的个数，与<a href="../user_guide.md#ivfsp">IVFSP业务算子模型文件生成</a>中的&lt;centroid num&gt;参数值对应。</p>
<p id="p390313219218"><a name="p390313219218"></a><a name="p390313219218"></a><strong id="b116451015104820"><a name="b116451015104820"></a><a name="b116451015104820"></a>const AscendIndexIVFSP &amp;codeBookSharedIdx</strong>：共享码本的AscendIndexIVFSP对象。</p>
<p id="p1990343252111"><a name="p1990343252111"></a><a name="p1990343252111"></a><strong id="b49034325211"><a name="b49034325211"></a><a name="b49034325211"></a>faiss::ScalarQuantizer::QuantizerType qType</strong>：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
<p id="p14903132162110"><a name="p14903132162110"></a><a name="p14903132162110"></a><strong id="b119031132182114"><a name="b119031132182114"></a><a name="b119031132182114"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。当前<span class="parmname" id="parmname7903143217211"><a name="parmname7903143217211"></a><a name="parmname7903143217211"></a>“faiss::MetricType metric”</span>仅支持<span class="parmvalue" id="parmvalue1190333211216"><a name="parmvalue1190333211216"></a><a name="parmvalue1190333211216"></a>“METRIC_L2”</span>。</p>
<p id="p20903173272114"><a name="p20903173272114"></a><a name="p20903173272114"></a><strong id="b1490310325216"><a name="b1490310325216"></a><a name="b1490310325216"></a>AscendIndexIVFSPConfig</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row890313323211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1090314321219"><a name="p1090314321219"></a><a name="p1090314321219"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6903123222115"><a name="p6903123222115"></a><a name="p6903123222115"></a>无</p>
</td>
</tr>
<tr id="row190393211210"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2903532142118"><a name="p2903532142118"></a><a name="p2903532142118"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p18903163218211"><a name="p18903163218211"></a><a name="p18903163218211"></a>无</p>
</td>
</tr>
<tr id="row4903123214219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p39031232162110"><a name="p39031232162110"></a><a name="p39031232162110"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul790383216215"></a><a name="ul790383216215"></a><ul id="ul790383216215"><li>训练生成码本时的&lt;dim&gt;、&lt;nonzero num&gt;、&lt;centroid num&gt; 值应该与此函数的参数<span class="parmname" id="parmname159038326214"><a name="parmname159038326214"></a><a name="parmname159038326214"></a>“dims”</span>、<span class="parmname" id="parmname390318322215"><a name="parmname390318322215"></a><a name="parmname390318322215"></a>“nonzeroNum”</span>、<span class="parmname" id="parmname69031632182114"><a name="parmname69031632182114"></a><a name="parmname69031632182114"></a>“nlist”</span>对应。</li><li>codeBookSharedIdx共享码本的码本配置要与当前Index的码本配置相同，且配置相同的Device资源。</li><li>当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。</li><li><span class="parmname" id="parmname13753112012420"><a name="parmname13753112012420"></a><a name="parmname13753112012420"></a>“nonzeroNum”</span>需为16的倍数且小于等于min(128, dims)。</li><li>metric ∈ {faiss::MetricType::METRIC_L2}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table8581162710235"></a>
<table><tbody><tr id="row258119270238"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p758152711231"><a name="p758152711231"></a><a name="p758152711231"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p158117272235"><a name="p158117272235"></a><a name="p158117272235"></a>AscendIndexIVFSP (const AscendIndexIVFSP&amp;) = delete;</p>
</td>
</tr>
<tr id="row6581192742313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p358110271235"><a name="p358110271235"></a><a name="p358110271235"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2581327192318"><a name="p2581327192318"></a><a name="p2581327192318"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row858114273233"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7581162715238"><a name="p7581162715238"></a><a name="p7581162715238"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b650433582416"><a name="b650433582416"></a><a name="b650433582416"></a>const AscendIndexIVFSP&amp;</strong>：常量AscendIndexIVFSP。</p>
</td>
</tr>
<tr id="row5581152722313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p458117278231"><a name="p458117278231"></a><a name="p458117278231"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1258111270239"><a name="p1258111270239"></a><a name="p1258111270239"></a>无</p>
</td>
</tr>
<tr id="row4581162702318"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p15581202722317"><a name="p15581202722317"></a><a name="p15581202722317"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p3581127182311"><a name="p3581127182311"></a><a name="p3581127182311"></a>无</p>
</td>
</tr>
<tr id="row125811227162312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p45811227132319"><a name="p45811227132319"></a><a name="p45811227132319"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table186918413239"></a>
<table><tbody><tr id="row1386916412234"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p78691441132310"><a name="p78691441132310"></a><a name="p78691441132310"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1386914415234"><a name="p1386914415234"></a><a name="p1386914415234"></a>virtual ~AscendIndexIVFSP();</p>
</td>
</tr>
<tr id="row686920419239"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1886910414237"><a name="p1886910414237"></a><a name="p1886910414237"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p178691241122313"><a name="p178691241122313"></a><a name="p178691241122313"></a>AscendIndexIVFSP的析构函数，销毁AscendIndexIVFSP对象，释放资源。</p>
</td>
</tr>
<tr id="row28695418235"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128698415234"><a name="p128698415234"></a><a name="p128698415234"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p086914116238"><a name="p086914116238"></a><a name="p086914116238"></a>无</p>
</td>
</tr>
<tr id="row19869641142315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18691641192314"><a name="p18691641192314"></a><a name="p18691641192314"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2869204132310"><a name="p2869204132310"></a><a name="p2869204132310"></a>无</p>
</td>
</tr>
<tr id="row6869134122317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p16869114115232"><a name="p16869114115232"></a><a name="p16869114115232"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p9869164162316"><a name="p9869164162316"></a><a name="p9869164162316"></a>无</p>
</td>
</tr>
<tr id="row3869841102310"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19869441112317"><a name="p19869441112317"></a><a name="p19869441112317"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p186914112231"><a name="p186914112231"></a><a name="p186914112231"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table241282321712"></a>
<table><tbody><tr id="row1441202301711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1541222315179"><a name="p1541222315179"></a><a name="p1541222315179"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15204727163415"><a name="p15204727163415"></a><a name="p15204727163415"></a>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</p>
</td>
</tr>
<tr id="row84121238175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3412122315172"><a name="p3412122315172"></a><a name="p3412122315172"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p56033285334"><a name="p56033285334"></a><a name="p56033285334"></a>AscendIndexIVFSP的构造函数，根据“config”中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row164121237173"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p164121723181717"><a name="p164121723181717"></a><a name="p164121723181717"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul534674763310"></a><a name="ul534674763310"></a><ul id="ul534674763310"><li>int dims：AscendIndexIVFSP管理的一组特征向量的维度。</li><li>int nonzeroNum：特征向量压缩降维后非零维度个数。</li><li>int nlist：聚类中心的个数，与<a href="../user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP业务算子模型文件生成”中的&lt;centroid num&gt;参数值对应。</li><li>faiss::ScalarQuantizer::QuantizerType qType：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</li><li>faiss::MetricType metric：AscendIndex在执行特征向量相似度检索时使用的距离度量类型。当前“faiss::MetricType metric”仅支持“METRIC_L2”。</li><li>AscendIndexIVFSPConfig：Device侧资源配置。</li></ul>
</td>
</tr>
<tr id="row6413192313178"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p6413142341717"><a name="p6413142341717"></a><a name="p6413142341717"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p19203142716347"><a name="p19203142716347"></a><a name="p19203142716347"></a>无</p>
</td>
</tr>
<tr id="row341316237178"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p24131423201720"><a name="p24131423201720"></a><a name="p24131423201720"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p6203427123419"><a name="p6203427123419"></a><a name="p6203427123419"></a>无</p>
</td>
</tr>
<tr id="row7413102361711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p6413323131715"><a name="p6413323131715"></a><a name="p6413323131715"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul162613915340"></a><a name="ul162613915340"></a><ul id="ul162613915340"><li>当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。</li><li>“nonzeroNum”需为16的倍数且小于等于min(128, dims)。</li><li>metric ∈ {faiss::MetricType::METRIC_L2}。</li></ul>
</td>
</tr>
</tbody>
</table>

#### loadAllData接口<a id="ZH-CN_TOPIC_0000001585736172"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void loadAllData(const char *dataPath);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>将Index结构从磁盘读入Device，包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b102253712317"><a name="b102253712317"></a><a name="b102253712317"></a>const char *dataPath：</strong>数据文件路径。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a><span class="parmvalue" id="parmvalue292811515461"><a name="parmvalue292811515461"></a><a name="parmvalue292811515461"></a>“dataPath”</span>对应的文件应该是调用saveAllData方法得到的落盘文件，程序执行用户对其有读权限；且文件不能为软链接。</p>
<p id="p1430141710323"><a name="p1430141710323"></a><a name="p1430141710323"></a>该接口无法共享码本，如需共享码本，建议使用loadAllData。</p>
</td>
</tr>
</tbody>
</table>

<a name="table115591219131513"></a>
<table><tbody><tr id="row1955918198153"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p255921981517"><a name="p255921981517"></a><a name="p255921981517"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p390319328341"><a name="p390319328341"></a><a name="p390319328341"></a>static std::shared_ptr&lt;AscendIndexIVFSP&gt; loadAllData(const AscendIndexIVFSPConfig &amp;config, const uint8_t *data, size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx = nullptr);</p>
</td>
</tr>
<tr id="row10559191931517"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1559111916158"><a name="p1559111916158"></a><a name="p1559111916158"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p144174125351"><a name="p144174125351"></a><a name="p144174125351"></a>从内存中恢复AscendIndexIVFSP对象。</p>
</td>
</tr>
<tr id="row4559219161516"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5559519101517"><a name="p5559519101517"></a><a name="p5559519101517"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul3105123212352"></a><a name="ul3105123212352"></a><ul id="ul3105123212352"><li><strong id="b3975154112358"><a name="b3975154112358"></a><a name="b3975154112358"></a>const AscendIndexIVFSPConfig &amp;config</strong>：Device侧资源配置，当前只需设置config.deviceList以及config.resourceSize即可，其他配置参数会从内存中恢复。</li><li><strong id="b8319545173518"><a name="b8319545173518"></a><a name="b8319545173518"></a>const uint8_t *data</strong>：由saveAllData方法得到的内存指针。</li><li><strong id="b1312214484354"><a name="b1312214484354"></a><a name="b1312214484354"></a>size_t dataLen</strong>：data指针的真实长度。</li><li><strong id="b125458521353"><a name="b125458521353"></a><a name="b125458521353"></a>const AscendIndexIVFSP *codeBookSharedIdx</strong>：共享码本的AscendIndexIVFSP指针，默认为nullptr，即不共享码本。</li></ul>
</td>
</tr>
<tr id="row18559201914151"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1255961941516"><a name="p1255961941516"></a><a name="p1255961941516"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1555981921519"><a name="p1555981921519"></a><a name="p1555981921519"></a>无</p>
</td>
</tr>
<tr id="row855915191150"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6560131941520"><a name="p6560131941520"></a><a name="p6560131941520"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1671312145366"><a name="p1671312145366"></a><a name="p1671312145366"></a>从内存中恢复的AscendIndexIVFSP智能指针对象。</p>
</td>
</tr>
<tr id="row956019190157"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5560111920158"><a name="p5560111920158"></a><a name="p5560111920158"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4752731193612"></a><a name="ul4752731193612"></a><ul id="ul4752731193612"><li>data需要为非空的合法指针。</li><li>dataLen为指针data的真实长度，否则可能出现越界读写错误并引起程序崩溃。</li><li>codeBookSharedIdx共享码本的码本配置要与当前Index的码本配置相同，且配置相同的Device资源。</li></ul>
</td>
</tr>
</tbody>
</table>

#### operator = 接口<a name="ZH-CN_TOPIC_0000001635975413"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>AscendIndexIVFSP&amp; operator=(const AscendIndexIVFSP&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b88405472511"><a name="b88405472511"></a><a name="b88405472511"></a>const AscendIndexIVFSP&amp;</strong>：常量AscendIndexIVFSP。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### remove\_ids接口<a name="ZH-CN_TOPIC_0000001635576085"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>实现AscendIndexIVFSP删除底库中指定的特征向量的接口。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b15391516143014"><a name="b15391516143014"></a><a name="b15391516143014"></a>const faiss::IDSelector &amp;sel</strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>返回被删除的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### reset接口<a name="ZH-CN_TOPIC_0000001635815485"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void reset() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>清空该AscendIndexIVFSP的底库向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### saveAllData接口<a name="ZH-CN_TOPIC_0000001635696053"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void saveAllData(const char *dataPath);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>将Index结构从Device侧写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b19557108143418"><a name="b19557108143418"></a><a name="b19557108143418"></a>const char *dataPath</strong>：保存的数据文件路径。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>用户应该保证<span class="parmvalue" id="parmvalue972214587421"><a name="parmvalue972214587421"></a><a name="parmvalue972214587421"></a>“dataPath”</span>文件路径所在的目录存在，且执行用户对目录具有写权限；出于安全加固的考虑，目录层级中不能含有软链接。</p>
<p id="p10274445174214"><a name="p10274445174214"></a><a name="p10274445174214"></a>当<span class="parmvalue" id="parmvalue4193131715434"><a name="parmvalue4193131715434"></a><a name="parmvalue4193131715434"></a>“dataPath”</span>对应的文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</p>
</td>
</tr>
</tbody>
</table>

<a name="table11876949141314"></a>
<table><tbody><tr id="row12876549141317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p20876649191320"><a name="p20876649191320"></a><a name="p20876649191320"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7338202573713"><a name="p7338202573713"></a><a name="p7338202573713"></a>void saveAllData(uint8_t *&amp;data, size_t &amp;dataLen) const;</p>
</td>
</tr>
<tr id="row1587654912137"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p148761549201313"><a name="p148761549201313"></a><a name="p148761549201313"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6203133116375"><a name="p6203133116375"></a><a name="p6203133116375"></a>将AscendIndexIVFSP对象存储至内存中。</p>
</td>
</tr>
<tr id="row17876184916136"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p38761849171311"><a name="p38761849171311"></a><a name="p38761849171311"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1333871953710"><a name="p1333871953710"></a><a name="p1333871953710"></a>无</p>
</td>
</tr>
<tr id="row7876174971312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p138768495136"><a name="p138768495136"></a><a name="p138768495136"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1142124363714"><a name="p1142124363714"></a><a name="p1142124363714"></a><strong id="b7190348133717"><a name="b7190348133717"></a><a name="b7190348133717"></a>uint8_t *&amp;data</strong>：存储AscendIndexIVFSP数据的内存指针。</p>
<p id="p10142204317379"><a name="p10142204317379"></a><a name="p10142204317379"></a><strong id="b133501052103714"><a name="b133501052103714"></a><a name="b133501052103714"></a>size_t &amp;dataLen</strong>：data指针的真实长度。</p>
</td>
</tr>
<tr id="row487615490131"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3876949141317"><a name="p3876949141317"></a><a name="p3876949141317"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p233621903719"><a name="p233621903719"></a><a name="p233621903719"></a>无</p>
</td>
</tr>
<tr id="row987624981313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5876149181310"><a name="p5876149181310"></a><a name="p5876149181310"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1418092193810"><a name="p1418092193810"></a><a name="p1418092193810"></a>传入的data需要为空指针，且接口返回后需要用户使用完data后通过delete来释放其内存，否则会造成内存泄漏。</p>
</td>
</tr>
</tbody>
</table>

#### search接口<a name="ZH-CN_TOPIC_0000001635815489"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1587414619374"><a name="p1587414619374"></a><a name="p1587414619374"></a>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>实现AscendIndexIVFSP特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1246083018397"><a name="p1246083018397"></a><a name="p1246083018397"></a><strong id="b1246015305391"><a name="b1246015305391"></a><a name="b1246015305391"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p17460230163917"><a name="p17460230163917"></a><a name="p17460230163917"></a><strong id="b12460133016398"><a name="b12460133016398"></a><a name="b12460133016398"></a>const float *x</strong>：特征向量数据。</p>
<p id="p546073017399"><a name="p546073017399"></a><a name="p546073017399"></a><strong id="b5460630173912"><a name="b5460630173912"></a><a name="b5460630173912"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p13712185717441"><a name="p13712185717441"></a><a name="p13712185717441"></a><strong id="b15637734194512"><a name="b15637734194512"></a><a name="b15637734194512"></a>const SearchParameters *params：</strong>Faiss的可选参数，默认为“nullptr”，暂不支持该参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p9711201512411"><a name="p9711201512411"></a><a name="p9711201512411"></a><strong id="b127111415174119"><a name="b127111415174119"></a><a name="b127111415174119"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname294711157429"><a name="parmname294711157429"></a><a name="parmname294711157429"></a>“k”</span>个向量间的距离值。当有效的检索结果不足<span class="parmname" id="parmname322862410423"><a name="parmname322862410423"></a><a name="parmname322862410423"></a>“k”</span>个时，剩余无效距离用65504或-65504填充。</p>
<p id="p4711515104119"><a name="p4711515104119"></a><a name="p4711515104119"></a><strong id="b571161514419"><a name="b571161514419"></a><a name="b571161514419"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname117371282423"><a name="parmname117371282423"></a><a name="parmname117371282423"></a>“k”</span>个向量的ID。当有效的检索结果不足<span class="parmname" id="parmname176161826144211"><a name="parmname176161826144211"></a><a name="parmname176161826144211"></a>“k”</span>个时，剩余无效label用<span class="parmvalue" id="parmvalue6403131919432"><a name="parmvalue6403131919432"></a><a name="parmvalue6403131919432"></a>“-1”</span>填充。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>查询的特征向量数据<span class="parmname" id="parmname1087952510438"><a name="parmname1087952510438"></a><a name="parmname1087952510438"></a>“x”</span>的长度应该为dims * <strong id="b1279975684114"><a name="b1279975684114"></a><a name="b1279975684114"></a>n</strong>，<span class="parmname" id="parmname948613316435"><a name="parmname948613316435"></a><a name="parmname948613316435"></a>“distances”</span>以及<span class="parmname" id="parmname7858139114317"><a name="parmname7858139114317"></a><a name="parmname7858139114317"></a>“labels”</span>的长度应该为<strong id="b187993567415"><a name="b187993567415"></a><a name="b187993567415"></a>k</strong> * <strong id="b27992562413"><a name="b27992562413"></a><a name="b27992562413"></a>n</strong>，否则可能会出现越界读写的情况，引起程序的崩溃。此处<span class="parmname" id="parmname1838195014437"><a name="parmname1838195014437"></a><a name="parmname1838195014437"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9；<span class="parmname" id="parmname10590145844313"><a name="parmname10590145844313"></a><a name="parmname10590145844313"></a>“k”</span>通常不允许超过4096。</p>
</td>
</tr>
</tbody>
</table>

#### search\_with\_filter接口<a name="ZH-CN_TOPIC_0000001585736176"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>AscendIndexIVFSP的特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname1297124204519"><a name="parmname1297124204519"></a><a name="parmname1297124204519"></a>“k”</span>条特征的ID。提供基于CID过滤的功能，“filters”为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b12607182024312"><a name="b12607182024312"></a><a name="b12607182024312"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><strong id="b960782054317"><a name="b960782054317"></a><a name="b960782054317"></a>const float *x</strong>：特征向量数据。</p>
<p id="p176073203438"><a name="p176073203438"></a><a name="p176073203438"></a><strong id="b9607820114310"><a name="b9607820114310"></a><a name="b9607820114310"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p76071120164313"><a name="p76071120164313"></a><a name="p76071120164313"></a><strong id="b16607122014317"><a name="b16607122014317"></a><a name="b16607122014317"></a>const void *filters</strong>：过滤条件。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14659337439"><a name="p14659337439"></a><a name="p14659337439"></a><strong id="b96520338430"><a name="b96520338430"></a><a name="b96520338430"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname993415618467"><a name="parmname993415618467"></a><a name="parmname993415618467"></a>“k”</span>个向量间的距离值。</p>
<p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b76513333436"><a name="b76513333436"></a><a name="b76513333436"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname23731212124614"><a name="parmname23731212124614"></a><a name="parmname23731212124614"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li><span class="parmname" id="parmname17755173284714"><a name="parmname17755173284714"></a><a name="parmname17755173284714"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li><strong id="b2977182414719"><a name="b2977182414719"></a><a name="b2977182414719"></a>“k”</strong>通常不允许超过4096。</li><li><span class="parmname" id="parmname106134074720"><a name="parmname106134074720"></a><a name="parmname106134074720"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b189302361443"><a name="b189302361443"></a><a name="b189302361443"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname17598461489"><a name="parmname17598461489"></a><a name="parmname17598461489"></a>“distances”</span>、<span class="parmname" id="parmname1527410554487"><a name="parmname1527410554487"></a><a name="parmname1527410554487"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b3930193664416"><a name="b3930193664416"></a><a name="b3930193664416"></a>k</strong> * <strong id="b1493017362449"><a name="b1493017362449"></a><a name="b1493017362449"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname148675510478"><a name="parmname148675510478"></a><a name="parmname148675510478"></a>“filters”</span>需要为非空指针，且长度为n * 6的uint32_t的数组，否则可能出现越界读的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### setNumProbes接口<a name="ZH-CN_TOPIC_0000001635576089"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void setNumProbes(int nprobes);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>设置检索时总的候选桶数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b16217144619214"><a name="b16217144619214"></a><a name="b16217144619214"></a>int nprobes</strong>：AscendIndexIVFSP的nprobe数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p24932033718"><a name="p24932033718"></a><a name="p24932033718"></a><span class="parmname" id="parmname248227153713"><a name="parmname248227153713"></a><a name="parmname248227153713"></a>“nprobes”</span>为16的倍数且符合0 &lt; nprobes ≤ nlist。</p>
</td>
</tr>
</tbody>
</table>

#### setVerbose接口<a name="ZH-CN_TOPIC_0000001586055516"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void setVerbose(bool verbose);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>设置是否显式添加特征向量到底库的进度。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b14187193523611"><a name="b14187193523611"></a><a name="b14187193523611"></a>bool verbose</strong>：是否显式添加特征向量到底库的进度。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### trainCodeBook接口<a name="ZH-CN_TOPIC_0000002148530670"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p108681253358"><a name="p108681253358"></a><a name="p108681253358"></a>void trainCodeBook(const AscendIndexCodeBookInitParams &amp;codeBookInitParams) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5404143115354"><a name="p5404143115354"></a><a name="p5404143115354"></a>IVFSP码本训练接口。如果训练速度较慢，可能是安装OpenBLAS时限制了使用单线程，可以设置环境变量export OMP_NUM_THREADS=4 进行加速</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12524203743519"><a name="p12524203743519"></a><a name="p12524203743519"></a>const AscendIndexCodeBookInitParams &amp;codeBookInitParams：训练码本所需的初始化参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184441341163411"><a name="p184441341163411"></a><a name="p184441341163411"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1944394111342"><a name="p1944394111342"></a><a name="p1944394111342"></a>参考<a href="./approximate_retrieval.md#ascendindexgreatinitparams接口">AscendIndexCodeBookInitParams接口</a>。</p>
</td>
</tr>
</tbody>
</table>

#### addCodeBook接口<a name="ZH-CN_TOPIC_0000002148372594"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p091033816364"><a name="p091033816364"></a><a name="p091033816364"></a>void addCodeBook(const char *codeBookPath);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16438944153611"><a name="p16438944153611"></a><a name="p16438944153611"></a>添加训练好的码本。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p7964249173612"><a name="p7964249173612"></a><a name="p7964249173612"></a>const char *codeBookPath：码本路径。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184441341163411"><a name="p184441341163411"></a><a name="p184441341163411"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p198901214595"><a name="p198901214595"></a><a name="p198901214595"></a>“codeBookPath”对应的文件是调用trainCodeBook方法得到的码本文件，程序执行用户对其有读权限；且文件不能为软链接。</p>
</td>
</tr>
</tbody>
</table>

#### AscendIndexCodeBookInitParams接口<a name="ZH-CN_TOPIC_0000002183731529"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4159142563912"><a name="p4159142563912"></a><a name="p4159142563912"></a>AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum, std::string codeBookOutputDir, std::string learnDataPath, bool verbose);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p123622413395"><a name="p123622413395"></a><a name="p123622413395"></a>IVFSP训练码本的初始化结构体。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1044544115344"><a name="p1044544115344"></a><a name="p1044544115344"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>参数值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p12329155424114"><a name="p12329155424114"></a><a name="p12329155424114"></a><strong id="b1039615564575"><a name="b1039615564575"></a><a name="b1039615564575"></a>int numIter</strong>：训练迭代次数参数，默认为<span class="parmvalue" id="parmvalue15177195954914"><a name="parmvalue15177195954914"></a><a name="parmvalue15177195954914"></a>“1”</span>。</p>
<p id="p632945434111"><a name="p632945434111"></a><a name="p632945434111"></a><strong id="b4257125835714"><a name="b4257125835714"></a><a name="b4257125835714"></a>int device</strong>：设备逻辑ID，默认为<span class="parmvalue" id="parmvalue15961200195018"><a name="parmvalue15961200195018"></a><a name="parmvalue15961200195018"></a>“0”</span>。</p>
<p id="p1032911541415"><a name="p1032911541415"></a><a name="p1032911541415"></a><strong id="b4898195914578"><a name="b4898195914578"></a><a name="b4898195914578"></a>float ratio</strong>：训练用原始样本的采样率，默认为<span class="parmvalue" id="parmvalue1728113155017"><a name="parmvalue1728113155017"></a><a name="parmvalue1728113155017"></a>“1.0”</span>。</p>
<p id="p15329175434116"><a name="p15329175434116"></a><a name="p15329175434116"></a><strong id="b166116165813"><a name="b166116165813"></a><a name="b166116165813"></a>int batchSize</strong>：训练时以batchSize大小执行训练。与<a href="../user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP训练算子模型文件生成”中的&lt;batch_size&gt;保持一致，默认值为“32768”。</p>
<p id="p9329185464116"><a name="p9329185464116"></a><a name="p9329185464116"></a><strong id="b11743147185810"><a name="b11743147185810"></a><a name="b11743147185810"></a>int codeNum</strong>：每次最大按codeNum样本数量操作码本，必须为2的幂次。与<a href="../user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP训练算子模型文件生成”中的&lt;codebook_batch_size&gt;保持一致，默认为<span class="parmvalue" id="parmvalue2730106115111"><a name="parmvalue2730106115111"></a><a name="parmvalue2730106115111"></a>“32768”</span>。</p>
<p id="p1232955415418"><a name="p1232955415418"></a><a name="p1232955415418"></a><strong id="b596299165819"><a name="b596299165819"></a><a name="b596299165819"></a>std::string codeBookOutputDir</strong>：生成的码本文件输出到的目录，用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。</p>
<p id="p163291154204114"><a name="p163291154204114"></a><a name="p163291154204114"></a><strong id="b1851871235817"><a name="b1851871235817"></a><a name="b1851871235817"></a>std::string learnDataPath</strong>：训练用的原始特征文件路径，支持bin、npy格式，bin存储方式为行优先，数据类型为<span class="parmvalue" id="parmvalue31381046194918"><a name="parmvalue31381046194918"></a><a name="parmvalue31381046194918"></a>“float32”</span>。</p>
<p id="p103292545418"><a name="p103292545418"></a><a name="p103292545418"></a><strong id="b1032851425818"><a name="b1032851425818"></a><a name="b1032851425818"></a>bool verbose</strong>：是否开启额外打印信息，默认为<span class="parmname" id="parmname55226406496"><a name="parmname55226406496"></a><a name="parmname55226406496"></a>“true”</span>。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1076619519437"></a><a name="ul1076619519437"></a><ul id="ul1076619519437"><li>numIter∈ (0, 20]。</li><li>ratio∈ (0, 1.0]。</li><li>batchSize∈ (0, 32768]。</li><li>codeNum∈ (0, 32768]。</li><li>当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</li><li>在执行训练生成码本前，请先参考<a href="../user_guide.md#ivfsp">IVFSP</a>生成训练算子模型文件。</li></ul>
</td>
</tr>
</tbody>
</table>

#### trainCodeBookFromMem接口<a name="ZH-CN_TOPIC_0000002257319034"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4398205434419"><a name="p4398205434419"></a><a name="p4398205434419"></a>void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>IVFSP码本训练接口。训练数据从内存中加载，如果训练速度较慢，可能是安装OpenBLAS时限制了使用单线程，可以设置环境变量export OMP_NUM_THREADS=4进行加速。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams：训练码本所需的初始化参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11397185416444"><a name="p11397185416444"></a><a name="p11397185416444"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p8383354134415"><a name="p8383354134415"></a><a name="p8383354134415"></a>了解AscendIndexCodeBookInitFromMemParams相关说明，请参见<a href="#ascendindexcodebookinitfrommemparams接口">AscendIndexCodeBookInitFromMemParams</a>。</p>
</td>
</tr>
</tbody>
</table>

#### AscendIndexCodeBookInitFromMemParams接口<a name="ZH-CN_TOPIC_0000002291969193"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7524719186"><a name="p7524719186"></a><a name="p7524719186"></a>AscendIndexCodeBookInitFromMemParams (int numIter, int device, float ratio, int batchSize, int codeNum,bool verbose,std::string codeBookOutputDir,const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>IVFSP训练码本的初始化结构体。从内存中加载训练数据。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>参数值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><strong id="b3354181515167"><a name="b3354181515167"></a><a name="b3354181515167"></a>int numIter：</strong>训练迭代次数参数，默认为“1”。</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><strong id="b13953183012163"><a name="b13953183012163"></a><a name="b13953183012163"></a>int device：</strong>设备逻辑ID，默认为“0”。</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><strong id="b229253513169"><a name="b229253513169"></a><a name="b229253513169"></a>float ratio：</strong>训练用原始样本的采样率，默认为“1.0”。</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><strong id="b1712474051618"><a name="b1712474051618"></a><a name="b1712474051618"></a>int batchSize：</strong>训练时以batchSize大小执行训练。与<a href="../user_guide.md#ivfsp">IVFSP训练算子模型文件生成</a>中的&lt;batch_size&gt;保持一致，要求大于“0”，默认值为“32768”。</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><strong id="b76015463165"><a name="b76015463165"></a><a name="b76015463165"></a>int codeNum：</strong>每次最大按codeNum样本数量操作码本，必须为2的幂次。与<a href="../user_guide.md#ivfsp">IVFSP训练算子模型文件生成</a>中的&lt;codebook_batch_size&gt;保持一致，要求大于0，默认为“32768”。</p>
<p id="p16645243598"><a name="p16645243598"></a><a name="p16645243598"></a><strong id="b16826155511614"><a name="b16826155511614"></a><a name="b16826155511614"></a>std::string codeBookOutputDir：</strong>生成的码本文件输出到的目录。用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。</p>
<p id="p864518435917"><a name="p864518435917"></a><a name="p864518435917"></a><strong id="b172846021716"><a name="b172846021716"></a><a name="b172846021716"></a>bool verbose：</strong>是否开启额外打印信息，默认为“true”。</p>
<p id="p146455433910"><a name="p146455433910"></a><a name="p146455433910"></a><strong id="b049205121720"><a name="b049205121720"></a><a name="b049205121720"></a>const float *memLearnData：</strong>内存中数据指针，默认为空指针。</p>
<p id="p1864544319911"><a name="p1864544319911"></a><a name="p1864544319911"></a><strong id="b3740189191719"><a name="b3740189191719"></a><a name="b3740189191719"></a>size_t memLearnDataSize：</strong>内存中数据长度，默认为0。</p>
<p id="p106451943193"><a name="p106451943193"></a><a name="p106451943193"></a><strong id="b28410155178"><a name="b28410155178"></a><a name="b28410155178"></a>bool isTrainAndAdd：</strong>是否训练码本后直接添加到Index开关，默认为false。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul129009408302"></a><a name="ul129009408302"></a><ul id="ul129009408302"><li>numIter∈ (0, 20]</li><li>ratio∈ (0, 1.0]</li><li>memLearnDataSize % dim == 0</li><li>memLearnDataSize≤25G</li></ul>
<a name="ul154204603015"></a><a name="ul154204603015"></a><ul id="ul154204603015"><li>当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</li><li>在执行训练生成码本前，请先参考<a href="../user_guide.md#ivfsp">IVFSP</a>章节生成训练算子模型文件。</li></ul>
<a name="ul547975410309"></a><a name="ul547975410309"></a><ul id="ul547975410309"><li>当isTrainAndAdd为true时，码本训练好之后直接添加到Index中，不会进行落盘；</li><li>当isTrainAndAdd为false时，码本会保存到codeBookOutputDir路径下，需调用addCodeBook手动添加。</li><li>memLearnDataSize为指针memLearnData的真实长度，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

### AscendIndexIVFSPConfig<a id="ZH-CN_TOPIC_0000001635696057"></a>

AscendIndexIVFSP需要使用对应的AscendIndexIVFSPConfig执行对应资源的初始化。

**公共参数<a name="section17656114673616"></a>**

|参数名|数据类型|参数说明|
|--|--|--|
|handleBatch|int|检索时每次下发计算的候选桶数量，默认值为64。|
|nprobe|int|检索时总的候选桶数量，默认值为64。|
|searchListSize|int|检索时每次下发计算的每个桶的最大样本数量，默认值为32768。若桶太大，程序会自动根据searchListSize将桶拆成多次算子下发计算距离。|

**接口说明<a name="section74781713710"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>inline AscendIndexIVFSPConfig();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>默认构造函数，默认devices为{0}，使用第0个<span id="ph79732210444"><a name="ph79732210444"></a><a name="ph79732210444"></a>昇腾AI处理器</span>进行计算，默认resources为128MB。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
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

<a name="table121971648373"></a>
<table><tbody><tr id="row13197134820716"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p51977481976"><a name="p51977481976"></a><a name="p51977481976"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161970481773"><a name="p161970481773"></a><a name="p161970481773"></a>inline explicit AscendIndexIVFSPConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</p>
</td>
</tr>
<tr id="row141971748972"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1419717481876"><a name="p1419717481876"></a><a name="p1419717481876"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSPConfig构造函数，生成AscendIndexIVFSPConfig，指定Device侧设备ID和资源池大小。</p>
</td>
</tr>
<tr id="row191973486716"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1019719485712"><a name="p1019719485712"></a><a name="p1019719485712"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p59096365498"><a name="p59096365498"></a><a name="p59096365498"></a><strong id="b2851659184912"><a name="b2851659184912"></a><a name="b2851659184912"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p9909173617494"><a name="p9909173617494"></a><a name="p9909173617494"></a><strong id="b61011425012"><a name="b61011425012"></a><a name="b61011425012"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmvalue" id="parmvalue134388571724"><a name="parmvalue134388571724"></a><a name="parmvalue134388571724"></a>“IVF_SP_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p1990912367496"><a name="p1990912367496"></a><a name="p1990912367496"></a><strong id="b19708106145014"><a name="b19708106145014"></a><a name="b19708106145014"></a>uint32_t blockSize</strong>：预置的内存块大小，单位为Byte。默认参数为头文件中的<span class="parmvalue" id="parmvalue380144918213"><a name="parmvalue380144918213"></a><a name="parmvalue380144918213"></a>“DEFAULT_BLOCK_SIZE”</span>。</p>
</td>
</tr>
<tr id="row61979480720"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p91979481711"><a name="p91979481711"></a><a name="p91979481711"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1719734811717"><a name="p1719734811717"></a><a name="p1719734811717"></a>无</p>
</td>
</tr>
<tr id="row1919711482718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p181974481777"><a name="p181974481777"></a><a name="p181974481777"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p919717486712"><a name="p919717486712"></a><a name="p919717486712"></a>无</p>
</td>
</tr>
<tr id="row1719719481072"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p181973481672"><a name="p181973481672"></a><a name="p181973481672"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20548163415505"></a><a name="ul20548163415505"></a><ul id="ul20548163415505"><li><span class="parmname" id="parmname956713818220"><a name="parmname956713818220"></a><a name="parmname956713818220"></a>“devices”</span>需要为合法有效不重复的设备ID，当前仅支持1个NPU设备。</li><li><span class="parmname" id="parmname798183519217"><a name="parmname798183519217"></a><a name="parmname798183519217"></a>“resources”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table56061252785"></a>
<table><tbody><tr id="row6606552282"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p06062521781"><a name="p06062521781"></a><a name="p06062521781"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p108121410115113"><a name="p108121410115113"></a><a name="p108121410115113"></a>inline explicit AscendIndexIVFSPConfig(std::vector&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</p>
</td>
</tr>
<tr id="row156061352486"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p460610521381"><a name="p460610521381"></a><a name="p460610521381"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8606152685"><a name="p8606152685"></a><a name="p8606152685"></a>AscendIndexIVFSPConfig构造函数，生成AscendIndexIVFSPConfig，指定Device侧设备ID和资源池大小。</p>
</td>
</tr>
<tr id="row146067521289"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p46064521488"><a name="p46064521488"></a><a name="p46064521488"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p15921231125111"><a name="p15921231125111"></a><a name="p15921231125111"></a><strong id="b18248195515616"><a name="b18248195515616"></a><a name="b18248195515616"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p09211631145120"><a name="p09211631145120"></a><a name="p09211631145120"></a><strong id="b11958155810618"><a name="b11958155810618"></a><a name="b11958155810618"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmvalue" id="parmvalue3723441276"><a name="parmvalue3723441276"></a><a name="parmvalue3723441276"></a>“IVF_SP_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p139219313517"><a name="p139219313517"></a><a name="p139219313517"></a><strong id="b159531621471"><a name="b159531621471"></a><a name="b159531621471"></a>uint32_t blockSize</strong>：预置的内存块大小，单位为Byte。默认参数为头文件中的<span class="parmvalue" id="parmvalue232315188714"><a name="parmvalue232315188714"></a><a name="parmvalue232315188714"></a>“DEFAULT_BLOCK_SIZE”</span>。</p>
</td>
</tr>
<tr id="row1160718521816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9607252586"><a name="p9607252586"></a><a name="p9607252586"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p106076525815"><a name="p106076525815"></a><a name="p106076525815"></a>无</p>
</td>
</tr>
<tr id="row36075522089"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6607125210819"><a name="p6607125210819"></a><a name="p6607125210819"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p26071521080"><a name="p26071521080"></a><a name="p26071521080"></a>无</p>
</td>
</tr>
<tr id="row8607152585"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p160714523812"><a name="p160714523812"></a><a name="p160714523812"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul899544135119"></a><a name="ul899544135119"></a><ul id="ul899544135119"><li><span class="parmname" id="parmname18598358477"><a name="parmname18598358477"></a><a name="parmname18598358477"></a>“devices”</span>需要为合法有效不重复的设备ID，当前仅支持1个NPU设备。</li><li><span class="parmname" id="parmname1019313551472"><a name="parmname1019313551472"></a><a name="parmname1019313551472"></a>“resources”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

### AscendIndexIVFSQ<a name="ZH-CN_TOPIC_0000001506334625"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456694964"></a>

AscendIndexIVFSQ利用IVF来进行加速，是二级近似检索算法。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

#### AscendIndexIVFSQ接口<a name="ZH-CN_TOPIC_0000001506414893"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a>AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQ的构造函数，基于一个已有的index创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1580317419509"><a name="b1580317419509"></a><a name="b1580317419509"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧资源配置。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b89210445502"><a name="b89210445502"></a><a name="b89210445502"></a>AscendIndexIVFSQConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname12160171235212"><a name="parmname12160171235212"></a><a name="parmname12160171235212"></a>“index”</span>需要为合法有效的CPU Index指针。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1823217151014"></a>
<table><tbody><tr id="row178231617161011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1882331711020"><a name="p1882331711020"></a><a name="p1882331711020"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p18649496410"><a name="p18649496410"></a><a name="p18649496410"></a>AscendIndexIVFSQ(int dims, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</p>
</td>
</tr>
<tr id="row8823317171017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7823617131019"><a name="p7823617131019"></a><a name="p7823617131019"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18823117111010"><a name="p18823117111010"></a><a name="p18823117111010"></a>AscendIndexIVFSQ的构造函数，生成AscendIndexIVFSQ，此时根据<span class="parmname" id="parmname16824101217816"><a name="parmname16824101217816"></a><a name="parmname16824101217816"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row1582381741012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128231617131015"><a name="p128231617131015"></a><a name="p128231617131015"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b342242917528"><a name="b342242917528"></a><a name="b342242917528"></a>int dims</strong>：AscendIndexIVFSQ管理的一组特征向量的维度。</p>
<p id="p169755411358"><a name="p169755411358"></a><a name="p169755411358"></a><strong id="b050293135214"><a name="b050293135214"></a><a name="b050293135214"></a>int nlist</strong>：聚类中心的个数，与算子生成脚本中的<span class="parmname" id="parmname1915820151081"><a name="parmname1915820151081"></a><a name="parmname1915820151081"></a>“coarse_centroid_num”</span>参数对应。</p>
<p id="p895114473339"><a name="p895114473339"></a><a name="p895114473339"></a><strong id="b42321338105211"><a name="b42321338105211"></a><a name="b42321338105211"></a>faiss::ScalarQuantizer::QuantizerType qtype</strong>：AscendIndexIVFSQ的量化器类型。</p>
<p id="p7823317181017"><a name="p7823317181017"></a><a name="p7823317181017"></a><strong id="b11282104020522"><a name="b11282104020522"></a><a name="b11282104020522"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p5823115014619"><a name="p5823115014619"></a><a name="p5823115014619"></a><strong id="b15262643135212"><a name="b15262643135212"></a><a name="b15262643135212"></a>bool encodeResidual</strong>：表示是否对残差编码。</p>
<p id="p168231017101016"><a name="p168231017101016"></a><a name="p168231017101016"></a><strong id="b1821144512529"><a name="b1821144512529"></a><a name="b1821144512529"></a>AscendIndexIVFSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row168231917191016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p6824121714106"><a name="p6824121714106"></a><a name="p6824121714106"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p138241317201019"><a name="p138241317201019"></a><a name="p138241317201019"></a>无</p>
</td>
</tr>
<tr id="row10824101711014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p68241317131018"><a name="p68241317131018"></a><a name="p68241317131018"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p682420176103"><a name="p682420176103"></a><a name="p682420176103"></a>无</p>
</td>
</tr>
<tr id="row5824161731013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p982431701017"><a name="p982431701017"></a><a name="p982431701017"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3234195217524"></a><a name="ul3234195217524"></a><ul id="ul3234195217524"><li>dims ∈ {64, 128, 256, 384, 512}</li><li>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>qtype = ScalarQuantizer::QuantizerType::QT_8bit，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}<div class="note" id="note123188311292"><a name="note123188311292"></a><a name="note123188311292"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p2318163115919"><a name="p2318163115919"></a><a name="p2318163115919"></a>当前<span class="parmname" id="parmname1831820318915"><a name="parmname1831820318915"></a><a name="parmname1831820318915"></a>“encodeResidual”</span>在<span class="parmvalue" id="parmvalue1631933115916"><a name="parmvalue1631933115916"></a><a name="parmvalue1631933115916"></a>“metric=faiss::MetricType::METRIC_INNER_PRODUCT”</span>下，仅支持<span class="parmvalue" id="parmvalue153191931795"><a name="parmvalue153191931795"></a><a name="parmvalue153191931795"></a>“false”</span>取值，即当前并不支持对残差编码的IVFSQ方法，当取值为<span class="parmvalue" id="parmvalue1531915311293"><a name="parmvalue1531915311293"></a><a name="parmvalue1531915311293"></a>“true”</span>时能够运行成功但存在精度问题。</p>
</div></div>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table134501935171012"></a>
<table><tbody><tr id="row11451103521010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p44511935121011"><a name="p44511935121011"></a><a name="p44511935121011"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1445153561020"><a name="p1445153561020"></a><a name="p1445153561020"></a>AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);</p>
</td>
</tr>
<tr id="row1945123511015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p04512353102"><a name="p04512353102"></a><a name="p04512353102"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8451173581017"><a name="p8451173581017"></a><a name="p8451173581017"></a>AscendIndexIVFSQ的构造函数，生成AscendIndexIVFSQ，此时根据<span class="parmname" id="parmname445173519102"><a name="parmname445173519102"></a><a name="parmname445173519102"></a>“config”</span>中配置的值设置Device侧资源。此接口不执行初始化，由子类执行初始化相关功能，后续会废弃此接口，请勿使用。</p>
</td>
</tr>
<tr id="row1645163571015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p74511835171017"><a name="p74511835171017"></a><a name="p74511835171017"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p64511935141018"><a name="p64511935141018"></a><a name="p64511935141018"></a><strong id="b54513358104"><a name="b54513358104"></a><a name="b54513358104"></a>int dims</strong>：AscendIndexIVFSQ管理的一组特征向量的维度。</p>
<p id="p13451183517103"><a name="p13451183517103"></a><a name="p13451183517103"></a><strong id="b1645153519106"><a name="b1645153519106"></a><a name="b1645153519106"></a>int nlist</strong>：聚类中心的个数，与算子生成脚本中的<span class="parmname" id="parmname1645193510102"><a name="parmname1645193510102"></a><a name="parmname1645193510102"></a>“coarse_centroid_num”</span>参数对应。</p>
<p id="p10451153591010"><a name="p10451153591010"></a><a name="p10451153591010"></a><strong id="b14451535111016"><a name="b14451535111016"></a><a name="b14451535111016"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p54513357103"><a name="p54513357103"></a><a name="p54513357103"></a><strong id="b16451435121012"><a name="b16451435121012"></a><a name="b16451435121012"></a>AscendIndexIVFSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row8451113510107"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p5451123541012"><a name="p5451123541012"></a><a name="p5451123541012"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p12451435121015"><a name="p12451435121015"></a><a name="p12451435121015"></a>无</p>
</td>
</tr>
<tr id="row194511735181010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p17451163513101"><a name="p17451163513101"></a><a name="p17451163513101"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p5451203551020"><a name="p5451203551020"></a><a name="p5451203551020"></a>无</p>
</td>
</tr>
<tr id="row1945183511016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1345123518101"><a name="p1345123518101"></a><a name="p1345123518101"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul15452103551014"></a><a name="ul15452103551014"></a><ul id="ul15452103551014"><li>dims ∈ {64, 128, 256, 384, 512}</li><li>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table663150151113"></a>
<table><tbody><tr id="row176440181111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p064509114"><a name="p064509114"></a><a name="p064509114"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p885213589106"><a name="p885213589106"></a><a name="p885213589106"></a>AscendIndexIVFSQ(const AscendIndexIVFSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row186417021110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p664405110"><a name="p664405110"></a><a name="p664405110"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p76470121111"><a name="p76470121111"></a><a name="p76470121111"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row964505113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2642019118"><a name="p2642019118"></a><a name="p2642019118"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b1399954415513"><a name="b1399954415513"></a><a name="b1399954415513"></a>const AscendIndexIVFSQ&amp;</strong>：常量AscendIndexIVFSQ。</p>
</td>
</tr>
<tr id="row8641601111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p13648019116"><a name="p13648019116"></a><a name="p13648019116"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p06420013110"><a name="p06420013110"></a><a name="p06420013110"></a>无</p>
</td>
</tr>
<tr id="row1641608114"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p96418010111"><a name="p96418010111"></a><a name="p96418010111"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1264107115"><a name="p1264107115"></a><a name="p1264107115"></a>无</p>
</td>
</tr>
<tr id="row176420181110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p126412017119"><a name="p126412017119"></a><a name="p126412017119"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16647010117"><a name="p16647010117"></a><a name="p16647010117"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### \~AscendIndexIVFSQ接口<a name="ZH-CN_TOPIC_0000001456534936"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexIVFSQ();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQ的析构函数，销毁AscendIndexIVFSQ对象，释放资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
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

#### copyFrom接口<a name="ZH-CN_TOPIC_0000001456375244"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1215384082314"><a name="p1215384082314"></a><a name="p1215384082314"></a>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQ基于一个已有的index拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b2023965517563"><a name="b2023965517563"></a><a name="b2023965517563"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1346620304216"><a name="p1346620304216"></a><a name="p1346620304216"></a><span class="parmname" id="parmname13234938125"><a name="parmname13234938125"></a><a name="parmname13234938125"></a>“index”</span>需要为合法有效的CPU Index指针，Index的维度d参数取值范围为{64, 128, 256, 384, 512}，</p>
<p id="p74662030922"><a name="p74662030922"></a><a name="p74662030922"></a>Index的聚类中心的个数nlist参数取值范围{1024, 2048, 4096, 8192, 16384, 32768}</p>
<p id="p154661430523"><a name="p154661430523"></a><a name="p154661430523"></a>总的候选桶数量nprobe的取值范围0 &lt; nprobe ≤ nlist</p>
<p id="p646610301721"><a name="p646610301721"></a><a name="p646610301721"></a>底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
<p id="p64663301228"><a name="p64663301228"></a><a name="p64663301228"></a>sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

#### copyTo接口<a name="ZH-CN_TOPIC_0000001506334649"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexIVFScalarQuantizer *index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1720318284418"><a name="p1720318284418"></a><a name="p1720318284418"></a>将AscendIndexIVFSQ的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b750912215531"><a name="b750912215531"></a><a name="b750912215531"></a>faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname95661875712"><a name="parmname95661875712"></a><a name="parmname95661875712"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

#### operator = 接口<a name="ZH-CN_TOPIC_0000001456854860"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a>AscendIndexIVFSQ&amp; operator=(const AscendIndexIVFSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b115570395614"><a name="b115570395614"></a><a name="b115570395614"></a>const AscendIndexIVFSQ&amp;</strong>：常量AscendIndexIVFSQ。</p>
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

#### train接口<a name="ZH-CN_TOPIC_0000001456854976"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void train(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对AscendIndexIVFSQ执行训练，继承AscendIndex中的相关接口并提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b351832435710"><a name="b351832435710"></a><a name="b351832435710"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b17199113075712"><a name="b17199113075712"></a><a name="b17199113075712"></a>const float *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>训练采用k-means进行聚类，训练集比较小可能会影响查询精度。</li><li>此处<span class="parmname" id="parmname125783489316"><a name="parmname125783489316"></a><a name="parmname125783489316"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname95481642105713"><a name="parmname95481642105713"></a><a name="parmname95481642105713"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

### AscendIndexIVFSQConfig<a name="ZH-CN_TOPIC_0000001456375204"></a>

AscendIndexIVFSQ需要使用对应的AscendIndexIVFSQConfig执行对应资源的初始化。

**AscendIndexIVFSQConfig<a name="section015013311183"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexIVFSQConfig();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>默认构造函数，默认devices为{0}，使用第0个<span id="ph79732210444"><a name="ph79732210444"></a><a name="ph79732210444"></a>昇腾AI处理器</span>进行计算，默认resource为384MB。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
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

<a name="table19736185071817"></a>
<table><tbody><tr id="row673665061814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p13736350131811"><a name="p13736350131811"></a><a name="p13736350131811"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14172141316118"><a name="p14172141316118"></a><a name="p14172141316118"></a>inline AscendIndexIVFSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</p>
</td>
</tr>
<tr id="row1773645071818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1373675014185"><a name="p1373675014185"></a><a name="p1373675014185"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQConfig的构造函数，生成AscendIndexIVFSQConfig，此时根据<span class="parmname" id="parmname880211685812"><a name="parmname880211685812"></a><a name="parmname880211685812"></a>“devices”</span>中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认的初始化。</p>
</td>
</tr>
<tr id="row37368508181"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p127361650191819"><a name="p127361650191819"></a><a name="p127361650191819"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p127360501187"><a name="p127360501187"></a><a name="p127360501187"></a><strong id="b97961612194319"><a name="b97961612194319"></a><a name="b97961612194319"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b52891554312"><a name="b52891554312"></a><a name="b52891554312"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname5831817164320"><a name="parmname5831817164320"></a><a name="parmname5831817164320"></a>“IVFSQ_DEFAULT_TEMP_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row573613503187"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p11736135011819"><a name="p11736135011819"></a><a name="p11736135011819"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1773695013184"><a name="p1773695013184"></a><a name="p1773695013184"></a>无</p>
</td>
</tr>
<tr id="row173619506182"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1273618502186"><a name="p1273618502186"></a><a name="p1273618502186"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p57361450111811"><a name="p57361450111811"></a><a name="p57361450111811"></a>无</p>
</td>
</tr>
<tr id="row8736205051816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p147360502189"><a name="p147360502189"></a><a name="p147360502189"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul107261247431"></a><a name="ul107261247431"></a><ul id="ul107261247431"><li><span class="parmname" id="parmname9379202674310"><a name="parmname9379202674310"></a><a name="parmname9379202674310"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname238112819435"><a name="parmname238112819435"></a><a name="parmname238112819435"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1056711401917"></a>
<table><tbody><tr id="row1956720419193"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18567124191912"><a name="p18567124191912"></a><a name="p18567124191912"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p656714416197"><a name="p656714416197"></a><a name="p656714416197"></a>inline AscendIndexIVFSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</p>
</td>
</tr>
<tr id="row25671541197"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p105673420197"><a name="p105673420197"></a><a name="p105673420197"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p115671411920"><a name="p115671411920"></a><a name="p115671411920"></a>AscendIndexIVFSQConfig的构造函数，生成AscendIndexIVFSQConfig，此时根据<span class="parmname" id="parmname07048285588"><a name="parmname07048285588"></a><a name="parmname07048285588"></a>“devices”</span>中配置的值设置Device侧<span id="ph185671543194"><a name="ph185671543194"></a><a name="ph185671543194"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认的初始化。</p>
</td>
</tr>
<tr id="row556720415197"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p956718420193"><a name="p956718420193"></a><a name="p956718420193"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p165676418192"><a name="p165676418192"></a><a name="p165676418192"></a><strong id="b2157122616458"><a name="b2157122616458"></a><a name="b2157122616458"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p15678461910"><a name="p15678461910"></a><a name="p15678461910"></a><strong id="b12875133214511"><a name="b12875133214511"></a><a name="b12875133214511"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname3581123624515"><a name="parmname3581123624515"></a><a name="parmname3581123624515"></a>“IVFSQ_DEFAULT_TEMP_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row256744171918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1856714121912"><a name="p1856714121912"></a><a name="p1856714121912"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p17567114101912"><a name="p17567114101912"></a><a name="p17567114101912"></a>无</p>
</td>
</tr>
<tr id="row15567164161919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3567144111913"><a name="p3567144111913"></a><a name="p3567144111913"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p95677416193"><a name="p95677416193"></a><a name="p95677416193"></a>无</p>
</td>
</tr>
<tr id="row13567194101919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11567449197"><a name="p11567449197"></a><a name="p11567449197"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9973203054911"></a><a name="ul9973203054911"></a><ul id="ul9973203054911"><li><span class="parmname" id="parmname2614369494"><a name="parmname2614369494"></a><a name="parmname2614369494"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname1549143813490"><a name="parmname1549143813490"></a><a name="parmname1549143813490"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

**SetDefaultIVFSQConfig<a name="section039015215286"></a>**

<a name="table1185313082915"></a>
<table><tbody><tr id="row18531107298"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p128531018292"><a name="p128531018292"></a><a name="p128531018292"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58281361827"><a name="p58281361827"></a><a name="p58281361827"></a>inline void SetDefaultIVFSQConfig();</p>
</td>
</tr>
<tr id="row198530002911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p8853200102915"><a name="p8853200102915"></a><a name="p8853200102915"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p685316016291"><a name="p685316016291"></a><a name="p685316016291"></a>执行默认的初始化，设置迭代数为16，每个centroids最多设置512个点。</p>
</td>
</tr>
<tr id="row68536022919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p13853804297"><a name="p13853804297"></a><a name="p13853804297"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p158531705291"><a name="p158531705291"></a><a name="p158531705291"></a>无</p>
</td>
</tr>
<tr id="row785340142913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18538016297"><a name="p18538016297"></a><a name="p18538016297"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p198531703295"><a name="p198531703295"></a><a name="p198531703295"></a>无</p>
</td>
</tr>
<tr id="row1885319062919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1853120102913"><a name="p1853120102913"></a><a name="p1853120102913"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p485319062912"><a name="p485319062912"></a><a name="p485319062912"></a>无</p>
</td>
</tr>
<tr id="row188538018295"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p78535032910"><a name="p78535032910"></a><a name="p78535032910"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p128538010296"><a name="p128538010296"></a><a name="p128538010296"></a>无</p>
</td>
</tr>
</tbody>
</table>

### AscendIndexIVFSQT<a name="ZH-CN_TOPIC_0000001456375224"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001506615005"></a>

AscendIndexIVFSQT类，包含降维算法的三级检索IVFSQ算法，需要传入两个参数指明降维前后的维度信息，要求降维后维度能整除降维前的维度。适用于1000万级底库的场景。

需要按照IVFSQT算子生成方式，生成三级检索所需算子。

该类型带有模糊聚类功能：入桶前，使用threshold参数控制模糊程度。请根据底库容量和可用内存大小设置threshold参数值，过大的threshold会引起内存不足，导致失败。Atlas 200/300/500 推理产品环境建议设置\[1.0, 1.1\]，Atlas 推理系列产品环境建议设置\[1.0, 1.5\]。搜索时建议使用**batch size = 65536**。

使用流程为：1.构建index对象；2.train数据；3.add数据；4.update数据；5.search检索数据；6.析构index对象。update后不支持继续add数据。有新数据需要进行检索时，请将原来的index对象析构后，重新按照流程使用。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

#### AscendIndexIVFSQT接口<a name="ZH-CN_TOPIC_0000001506495685"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a>AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQT的构造函数，基于一个已有的<span class="parmname" id="parmname102792478176"><a name="parmname102792478176"></a><a name="parmname102792478176"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b46201291626"><a name="b46201291626"></a><a name="b46201291626"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧的Index资源。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b159906314214"><a name="b159906314214"></a><a name="b159906314214"></a>AscendIndexIVFSQTConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1113653515219"></a><a name="ul1113653515219"></a><ul id="ul1113653515219"><li><span class="parmname" id="parmname173302381721"><a name="parmname173302381721"></a><a name="parmname173302381721"></a>“index”</span>需要为合法有效的CPU Index指针。</li><li>index-&gt;d ∈ {256}。</li><li>index-&gt;sq.d ∈ {32, 64, 128}。</li><li><span class="parmname" id="parmname144451835155310"><a name="parmname144451835155310"></a><a name="parmname144451835155310"></a>“index”</span>的维度必须大于index-&gt;sq的维度且可以被index-&gt;sq的维度整除。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table124585216195"></a>
<table><tbody><tr id="row164575271917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p134518527191"><a name="p134518527191"></a><a name="p134518527191"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p18649496410"><a name="p18649496410"></a><a name="p18649496410"></a>AscendIndexIVFSQT(int dimIn, int dimOut, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</p>
</td>
</tr>
<tr id="row1045152101914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1245185213193"><a name="p1245185213193"></a><a name="p1245185213193"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1845115251915"><a name="p1845115251915"></a><a name="p1845115251915"></a>AscendIndexIVFSQT的构造函数，生成AscendIndexIVFSQT，此时根据<span class="parmname" id="parmname74506566178"><a name="parmname74506566178"></a><a name="parmname74506566178"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row16451352141910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p204585213198"><a name="p204585213198"></a><a name="p204585213198"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b162991621040"><a name="b162991621040"></a><a name="b162991621040"></a>int dimIn</strong>：AscendIndexIVFSQT管理的一组原始特征向量的维度。</p>
<p id="p055565814416"><a name="p055565814416"></a><a name="p055565814416"></a><strong id="b18128104346"><a name="b18128104346"></a><a name="b18128104346"></a>int dimOut</strong>：AscendIndexIVFSQT管理的一组降维目标特征向量的维度。</p>
<p id="p169755411358"><a name="p169755411358"></a><a name="p169755411358"></a><strong id="b11908056418"><a name="b11908056418"></a><a name="b11908056418"></a>int nlist</strong>：聚类中心的个数，与算子生成脚本中的<span class="parmname" id="parmname45896241598"><a name="parmname45896241598"></a><a name="parmname45896241598"></a>“coarse_centroid_num”</span>参数对应。</p>
<p id="p895114473339"><a name="p895114473339"></a><a name="p895114473339"></a><strong id="b5178209448"><a name="b5178209448"></a><a name="b5178209448"></a>faiss::ScalarQuantizer::QuantizerType qtype</strong>：AscendIndexIVFSQT的量化器类型。</p>
<p id="p174585217192"><a name="p174585217192"></a><a name="p174585217192"></a><strong id="b27578121644"><a name="b27578121644"></a><a name="b27578121644"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p345135213199"><a name="p345135213199"></a><a name="p345135213199"></a><strong id="b155089152412"><a name="b155089152412"></a><a name="b155089152412"></a>AscendIndexIVFSQTConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row19459527195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p94525214193"><a name="p94525214193"></a><a name="p94525214193"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p204520521198"><a name="p204520521198"></a><a name="p204520521198"></a>无</p>
</td>
</tr>
<tr id="row10451352191917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p44514527191"><a name="p44514527191"></a><a name="p44514527191"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p17451152151917"><a name="p17451152151917"></a><a name="p17451152151917"></a>无</p>
</td>
</tr>
<tr id="row154545211199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15456526194"><a name="p15456526194"></a><a name="p15456526194"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3942173318420"></a><a name="ul3942173318420"></a><ul id="ul3942173318420"><li>dimIn ∈ {256}。</li><li>dimOut ∈ {32, 64, 128}。</li><li>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}。</li><li>qtype = ScalarQuantizer::QuantizerType::QT_8bit，当前仅支持<span class="parmvalue" id="parmvalue35921391642"><a name="parmvalue35921391642"></a><a name="parmvalue35921391642"></a>“ScalarQuantizer::QuantizerType::QT_8bit”</span>量化器类型。</li><li>metric = faiss::MetricType::METRIC_INNER_PRODUCT （当前仅支持 <span class="parmvalue" id="parmvalue20208125320410"><a name="parmvalue20208125320410"></a><a name="parmvalue20208125320410"></a>“faiss::MetricType::METRIC_INNER_PRODUCT”</span>。）</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table68594118203"></a>
<table><tbody><tr id="row12859818205"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1885911122013"><a name="p1885911122013"></a><a name="p1885911122013"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p885213589106"><a name="p885213589106"></a><a name="p885213589106"></a>AscendIndexIVFSQT(const AscendIndexIVFSQT&amp;) = delete;</p>
</td>
</tr>
<tr id="row158592122017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p68591616209"><a name="p68591616209"></a><a name="p68591616209"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p78592113207"><a name="p78592113207"></a><a name="p78592113207"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row18859201122014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18859141122017"><a name="p18859141122017"></a><a name="p18859141122017"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b3710328459"><a name="b3710328459"></a><a name="b3710328459"></a>const AscendIndexIVFSQT&amp;</strong>：AscendIndexIVFSQT对象。</p>
</td>
</tr>
<tr id="row28605142020"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p286001112012"><a name="p286001112012"></a><a name="p286001112012"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16860017208"><a name="p16860017208"></a><a name="p16860017208"></a>无</p>
</td>
</tr>
<tr id="row1186001132017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1586020118202"><a name="p1586020118202"></a><a name="p1586020118202"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p886015192016"><a name="p886015192016"></a><a name="p886015192016"></a>无</p>
</td>
</tr>
<tr id="row38604142015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1586012112011"><a name="p1586012112011"></a><a name="p1586012112011"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### \~AscendIndexIVFSQT接口<a name="ZH-CN_TOPIC_0000001456854984"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexIVFSQT();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQT的析构函数，销毁AscendIndexIVFSQT对象，释放资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
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

#### copyFrom接口<a name="ZH-CN_TOPIC_0000001456695060"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1215384082314"><a name="p1215384082314"></a><a name="p1215384082314"></a>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVSQT基于一个已有的<span class="parmname" id="parmname4930151917214"><a name="parmname4930151917214"></a><a name="parmname4930151917214"></a>“index”</span>拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b9345217864"><a name="b9345217864"></a><a name="b9345217864"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1430942010615"><a name="parmname1430942010615"></a><a name="parmname1430942010615"></a>“index”</span>需要为合法有效的CPU Index指针。</p>
<a name="ul1113653515219"></a><a name="ul1113653515219"></a><ul id="ul1113653515219"><li>index-&gt;d ∈ {256}。</li><li>index-&gt;sq.d ∈ {32, 64, 128}。</li><li><span class="parmname" id="parmname144451835155310"><a name="parmname144451835155310"></a><a name="parmname144451835155310"></a>“index”</span>的维度必须大于index-&gt;sq的维度，且可以被index-&gt;sq的维度整除。</li><li>update过的对象请勿调用该接口。</li></ul>
</td>
</tr>
</tbody>
</table>

#### copyTo接口<a name="ZH-CN_TOPIC_0000001506495825"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexIVFScalarQuantizer *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1720318284418"><a name="p1720318284418"></a><a name="p1720318284418"></a>将AscendIndexIVFSQT的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b158351007610"><a name="b158351007610"></a><a name="b158351007610"></a>faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname99881731766"><a name="parmname99881731766"></a><a name="parmname99881731766"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

#### fineTune接口<a name="ZH-CN_TOPIC_0000001456694860"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p193348812010"><a name="p193348812010"></a><a name="p193348812010"></a>void fineTune(size_t n, const float *x);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对中心进行微调和优化，避免分桶不均匀的问题。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p492321510014"><a name="p492321510014"></a><a name="p492321510014"></a><strong id="b101231056175619"><a name="b101231056175619"></a><a name="b101231056175619"></a>size_t n</strong>：特征向量的条数。</p>
<p id="p12314314436"><a name="p12314314436"></a><a name="p12314314436"></a><strong id="b3709358115620"><a name="b3709358115620"></a><a name="b3709358115620"></a>const float *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p1529214384393"><a name="p1529214384393"></a><a name="p1529214384393"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### getFuzzyK接口<a name="ZH-CN_TOPIC_0000001456855008"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>int getFuzzyK() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>获取入桶时每个向量的最大值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b8278171308"><a name="b8278171308"></a><a name="b8278171308"></a>int</strong>：每个向量入桶时的最大值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### getListCodesAndIds接口<a name="ZH-CN_TOPIC_0000001687739112"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量和对应ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b234205219283"><a name="b234205219283"></a><a name="b234205219283"></a>int listId</strong>：AscendIndexIVFSQT的nlist中的特定nlistId。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p812472610226"><a name="p812472610226"></a><a name="p812472610226"></a><strong id="b8752144372820"><a name="b8752144372820"></a><a name="b8752144372820"></a>std::vector&lt;uint8_t&gt;&amp; codes</strong>：AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量。</p>
<p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b198817462287"><a name="b198817462287"></a><a name="b198817462287"></a>std::vector&lt;ascend_idx_t&gt;&amp; ids</strong>：AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p13621611141120"><a name="p13621611141120"></a><a name="p13621611141120"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### getListLength接口<a name="ZH-CN_TOPIC_0000001735977797"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>uint32_t getListLength(int listId) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVFSQT的nlist中的特定nlistId上的长度。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b121461446192713"><a name="b121461446192713"></a><a name="b121461446192713"></a>int listId</strong>：AscendIndexIVFSQT的nlist中的特定nlistId。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>AscendIndexIVFSQT的nlist中的特定nlistId上的长度。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p13621611141120"><a name="p13621611141120"></a><a name="p13621611141120"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### getLowerBound接口<a name="ZH-CN_TOPIC_0000001506614885"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p139751803263"><a name="p139751803263"></a><a name="p139751803263"></a>int getLowerBound() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>返回二级分簇的阈值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p192391322172611"><a name="p192391322172611"></a><a name="p192391322172611"></a>二级分簇的阈值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### getMergeThres接口<a name="ZH-CN_TOPIC_0000001506615073"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p898555319557"><a name="p898555319557"></a><a name="p898555319557"></a>int getMergeThres() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p476310541227"><a name="p476310541227"></a><a name="p476310541227"></a>获取合并子桶阈值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>合并子桶阈值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### getQMax接口<a name="ZH-CN_TOPIC_0000001456535208"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>float getQMax() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>返回特征向量的最大值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>特征向量的最大值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### getQMin接口<a name="ZH-CN_TOPIC_0000001506615029"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p114441322513"><a name="p114441322513"></a><a name="p114441322513"></a>float getQMin() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p810264716532"><a name="p810264716532"></a><a name="p810264716532"></a>返回特征向量的最小值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>特征向量的最小值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### getThreshold接口<a name="ZH-CN_TOPIC_0000001506334633"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>float getThreshold() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>获取判断向量是否入多个桶的阈值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b4330591711"><a name="b4330591711"></a><a name="b4330591711"></a>float</strong>：判断向量是否入多个桶的阈值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### operator = 接口<a name="ZH-CN_TOPIC_0000001506615085"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a>AscendIndexIVFSQT&amp; operator=(const AscendIndexIVFSQT&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b0567942255"><a name="b0567942255"></a><a name="b0567942255"></a>const AscendIndexIVFSQT&amp;</strong>：AscendIndexIVFSQT对象。</p>
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

#### remove\_ids接口<a name="ZH-CN_TOPIC_0000001506615053"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>根据ID删除底库特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b8255123114917"><a name="b8255123114917"></a><a name="b8255123114917"></a>const faiss::IDSelector &amp;sel</strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>返回被删除的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>当前版本暂不支持该接口。</p>
</td>
</tr>
</tbody>
</table>

#### reset接口<a name="ZH-CN_TOPIC_0000001506334789"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>void reset() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p10290194315362"><a name="p10290194315362"></a><a name="p10290194315362"></a>重置索引，特征数据清零。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1414182616338"><a name="p1414182616338"></a><a name="p1414182616338"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>调用后请勿继续使用该对象。</p>
</td>
</tr>
</tbody>
</table>

#### setAddTotal接口<a name="ZH-CN_TOPIC_0000001456375316"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p11341163171010"><a name="p11341163171010"></a><a name="p11341163171010"></a>void setAddTotal(size_t addTotal);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p19691924201019"><a name="p19691924201019"></a><a name="p19691924201019"></a>设置待添加的底库向量总数，默认值<span class="parmvalue" id="parmvalue199423381138"><a name="parmvalue199423381138"></a><a name="parmvalue199423381138"></a>“100000000”</span>。需要先设置<span class="parmname" id="parmname1710103964710"><a name="parmname1710103964710"></a><a name="parmname1710103964710"></a>“PreciseMemControl”</span>为<span class="parmvalue" id="parmvalue185542312415"><a name="parmvalue185542312415"></a><a name="parmvalue185542312415"></a>“true”</span>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b999816351837"><a name="b999816351837"></a><a name="b999816351837"></a>size_t addTotal</strong>：待添加的底库向量总数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p8927121410219"><a name="p8927121410219"></a><a name="p8927121410219"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### setFuzzyK接口<a name="ZH-CN_TOPIC_0000001456534940"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>void setFuzzyK(int value);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>设置入桶时每个向量的最大值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a><strong id="b144117156596"><a name="b144117156596"></a><a name="b144117156596"></a>int value</strong>：每个向量入桶时的最大值，建议固定为默认值<span class="parmvalue" id="parmvalue19251921185917"><a name="parmvalue19251921185917"></a><a name="parmvalue19251921185917"></a>“3”</span>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p64091154105412"><a name="p64091154105412"></a><a name="p64091154105412"></a>value的取值范围是（0,10]。</p>
</td>
</tr>
</tbody>
</table>

#### setLowerBound接口<a name="ZH-CN_TOPIC_0000001506334777"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p620411196166"><a name="p620411196166"></a><a name="p620411196166"></a>void setLowerBound(int lowerBound);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p6359132829"><a name="p6359132829"></a><a name="p6359132829"></a>设置二级分簇的阈值，默认值为<span class="parmvalue" id="parmvalue1967655810458"><a name="parmvalue1967655810458"></a><a name="parmvalue1967655810458"></a>“32”</span>。</p>
<p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>若一级分簇桶中元素大于lowerBound则进行二次分簇，否则保留原状。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b173513119118"><a name="b173513119118"></a><a name="b173513119118"></a>int lowerBound</strong>：二级分簇的阈值。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p7518112174013"><a name="p7518112174013"></a><a name="p7518112174013"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### setMemoryLimit接口<a name="ZH-CN_TOPIC_0000001506614917"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p388113312502"><a name="p388113312502"></a><a name="p388113312502"></a>void setMemoryLimit(float memoryLimit);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>设置Host内存限制，默认值为<span class="parmvalue" id="parmvalue1967655810458"><a name="parmvalue1967655810458"></a><a name="parmvalue1967655810458"></a>“32”</span>，单位“GB”。需要先设置<span class="parmname" id="parmname1710103964710"><a name="parmname1710103964710"></a><a name="parmname1710103964710"></a>“PreciseMemControl”</span>为<span class="parmvalue" id="parmvalue185542312415"><a name="parmvalue185542312415"></a><a name="parmvalue185542312415"></a>“true”</span>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b177901237145615"><a name="b177901237145615"></a><a name="b177901237145615"></a>float memoryLimit</strong>：内存限制。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p8927121410219"><a name="p8927121410219"></a><a name="p8927121410219"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### setMergeThres接口<a name="ZH-CN_TOPIC_0000001456694900"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>void setMergeThres(int mergeThres);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p476310541227"><a name="p476310541227"></a><a name="p476310541227"></a>设置合并子桶阈值，默认值为<span class="parmvalue" id="parmvalue1967655810458"><a name="parmvalue1967655810458"></a><a name="parmvalue1967655810458"></a>“5”</span>。</p>
<p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>若二级分簇后某子桶中元素小于mergeThres，则合并该子桶元素至其他子桶中。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b117388471122"><a name="b117388471122"></a><a name="b117388471122"></a>int mergeThres</strong>：合并子桶阈值。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p276517505390"><a name="p276517505390"></a><a name="p276517505390"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### setNumProbes接口<a name="ZH-CN_TOPIC_0000001736410013"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>void setNumProbes(int nprobes) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>设置当前的AscendIndexIVFSQT的nprobe数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b16217144619214"><a name="b16217144619214"></a><a name="b16217144619214"></a>int nprobes</strong>：AscendIndexIVFSQT的nprobe数。建议保持为默认值<span class="parmvalue" id="parmvalue1410214422510"><a name="parmvalue1410214422510"></a><a name="parmvalue1410214422510"></a>“64”</span>。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li>nprobes ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }</li><li>l2Probe ≥ nprobes, l2Probe≤ l3SegmentNum, l2Probe≤ nprobes * 64</li><li>l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }</li><li>l2Probe和l3SegmentNum的设置可参见<a href="#setsearchparams接口">setSearchParams</a>。</li><li>setNumProbes接口预计2025年9月废除，请使用<a href="#setsearchparams接口">setSearchParams</a>。</li></ul>
</td>
</tr>
</tbody>
</table>

#### setPreciseMemControl接口<a name="ZH-CN_TOPIC_0000001506334681"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p534012467165"><a name="p534012467165"></a><a name="p534012467165"></a>void setPreciseMemControl(bool preciseMemControl);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p19691924201019"><a name="p19691924201019"></a><a name="p19691924201019"></a>是否精确限制Host侧的内存大小。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b9961232154"><a name="b9961232154"></a><a name="b9961232154"></a>bool preciseMemControl</strong>：默认为<span class="parmvalue" id="parmvalue185542312415"><a name="parmvalue185542312415"></a><a name="parmvalue185542312415"></a>“false”</span>，表示停用对Host侧内存大小精确限制；为<span class="parmvalue" id="parmvalue228313252247"><a name="parmvalue228313252247"></a><a name="parmvalue228313252247"></a>“true”</span>时表示启用。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

#### setSearchParams接口<a name="ZH-CN_TOPIC_0000002052679693"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1115121782220"><a name="p1115121782220"></a><a name="p1115121782220"></a>void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p10890925192214"><a name="p10890925192214"></a><a name="p10890925192214"></a>设置影响检索精度和性能的参数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p14571833142216"><a name="p14571833142216"></a><a name="p14571833142216"></a>int nprobe：AscendIndexIVFSQT的nprobe数。建议保持为默认值“64”。</p>
<p id="p9571533132212"><a name="p9571533132212"></a><a name="p9571533132212"></a>int l2Probe：二级检索选择子桶的数量，默认值为“48”。</p>
<p id="p8571033152213"><a name="p8571033152213"></a><a name="p8571033152213"></a>int l3SegmentNum：L3算子处理的段数，影响查找的base总数，默认值为“96”。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li>nprobe ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }</li><li>l2Probe ≥ nprobe, l2Probe≤ l3SegmentNum, l2Probe≤ nprobe * 64</li><li>l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }</li></ul>
</td>
</tr>
</tbody>
</table>

#### setSortMode接口<a name="ZH-CN_TOPIC_0000002165943965"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void setSortMode(int mode);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>设置topk排序模式。模式0为近似排序；模式1为精确排序。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.3.1 "><p id="p6307181718287"><a name="p6307181718287"></a><a name="p6307181718287"></a>int mode：topk排序模式。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.6.1 "><a name="ul998918501528"></a><a name="ul998918501528"></a><ul id="ul998918501528"><li>该接口需要在Search接口之前使用。</li><li><span class="parmname" id="parmname4750135914115"><a name="parmname4750135914115"></a><a name="parmname4750135914115"></a>“mode”</span>支持模式0或模式1，默认为模式0。<a name="ul73211618141111"></a><a name="ul73211618141111"></a><ul id="ul73211618141111"><li>模式0：近似排序会截断部分topk结果，提升性能。</li><li>模式1：精确排序，会提升检索精度，牺牲部分性能。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

#### setThreshold接口<a name="ZH-CN_TOPIC_0000001456854808"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>void setThreshold(float value);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>设置判断向量是否入多个桶的阈值，默认值为<span class="parmvalue" id="parmvalue756673153110"><a name="parmvalue756673153110"></a><a name="parmvalue756673153110"></a>“1.0”</span>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a><strong id="b07338551508"><a name="b07338551508"></a><a name="b07338551508"></a>float value</strong>：判断向量是否入多个桶的阈值，建议设置[1.0, 1.5]。由于Device侧内存存在限额，当使用内存达到限额后，会触发OOM机制，导致进程被杀死。用户可先查看Device侧的内存限额数据。（/sys/fs/cgroup/memory/usermemory/memory.limit_in_bytes），来评估添加底库的大小，若内存不充裕时，参数值建议在[1.0, 1.1]范围。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>value的取值范围是[0, fuzzyK- 1]，fuzzyK的取值请参见<a href="#getfuzzyk接口">getFuzzyK接口</a>。</p>
</td>
</tr>
</tbody>
</table>

#### setUseCpuUpdate接口<a name="ZH-CN_TOPIC_0000002167379329"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p2026619114105"><a name="p2026619114105"></a><a name="p2026619114105"></a>setUseCpuUpdate(int numThreads);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p7266411101019"><a name="p7266411101019"></a><a name="p7266411101019"></a>是否使用CPU进行<a href="#update接口">update</a>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p172659115101"><a name="p172659115101"></a><a name="p172659115101"></a><strong id="b11603133521113"><a name="b11603133521113"></a><a name="b11603133521113"></a>int numThreads</strong>：用于进行update的CPU核数，默认值为当前CPU的核数。</p>
<a name="ul76628243368"></a><a name="ul76628243368"></a><ul id="ul76628243368"><li>若当前CPU的核数&gt;96：<a name="ul11814123683920"></a><a name="ul11814123683920"></a><ul id="ul11814123683920"><li>当前CPU核数＜输入的numThreads，<strong id="b2028810524444"><a name="b2028810524444"></a><a name="b2028810524444"></a>numThreads</strong> =96；</li><li>96＜输入的numThreads≤当前CPU核数，<strong id="b324754114419"><a name="b324754114419"></a><a name="b324754114419"></a>numThreads</strong>=96；</li><li>输入的numThreads≤96，numThreads为输入值。</li></ul>
</li><li>若当前CPU的核数≤96：<a name="ul106753468457"></a><a name="ul106753468457"></a><ul id="ul106753468457"><li>当前CPU核数＜输入的numThreads ≤ 96，<strong id="b20758111294711"><a name="b20758111294711"></a><a name="b20758111294711"></a>numThreads</strong>为当前CPU核数；</li><li>0＜输入的numThreads≤当前CPU核数，<strong id="b1695581814610"><a name="b1695581814610"></a><a name="b1695581814610"></a>numThreads</strong>为输入值<strong id="b2955318204612"><a name="b2955318204612"></a><a name="b2955318204612"></a>。</strong></li></ul>
</li></ul>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p152641411141014"><a name="p152641411141014"></a><a name="p152641411141014"></a>无。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1426321151015"><a name="p1426321151015"></a><a name="p1426321151015"></a>无。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul20955612171320"></a><a name="ul20955612171320"></a><ul id="ul20955612171320"><li><strong id="b1276762719138"><a name="b1276762719138"></a><a name="b1276762719138"></a>numThreads</strong>取值需大于0。</li><li>需要在使用<a href="#update接口">update</a>前配置。</li></ul>
</td>
</tr>
</tbody>
</table>

#### train接口<a name="ZH-CN_TOPIC_0000001456375352"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void train(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对AscendIndexIVFSQT执行训练，继承AscendIndexIVFSQ中的相关接口并提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b141925213710"><a name="b141925213710"></a><a name="b141925213710"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b1196267978"><a name="b1196267978"></a><a name="b1196267978"></a>const float *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul15165212077"></a><a name="ul15165212077"></a><ul id="ul15165212077"><li>训练采用k-means进行聚类，训练集比较小可能会影响查询精度。</li><li>此处<span class="parmname" id="parmname11827122719317"><a name="parmname11827122719317"></a><a name="parmname11827122719317"></a>“n”</span>的取值范围：nlist ≤ n ≤ 7,000,000。</li><li>此处指针<span class="parmname" id="parmname19120718371"><a name="parmname19120718371"></a><a name="parmname19120718371"></a>“x”</span>需要为非空指针，且长度应该为dimIn * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### update接口<a name="ZH-CN_TOPIC_0000001506414869"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void update(bool cleanData = true);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>三级检索的第二级，在add完毕全部的底库数据后，执行search前，用于训练子桶中心并根据子桶中心入桶。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p6307181718287"><a name="p6307181718287"></a><a name="p6307181718287"></a>cleanData：是否清除中间数据，默认为<span class="parmvalue" id="parmvalue137057376475"><a name="parmvalue137057376475"></a><a name="parmvalue137057376475"></a>“true”</span>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p117332418161"><a name="p117332418161"></a><a name="p117332418161"></a>一次检索全流程中该接口只需要调用一次。</p>
</td>
</tr>
</tbody>
</table>

#### updateTParams接口<a name="ZH-CN_TOPIC_0000001456854936"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1418915311587"><a name="p1418915311587"></a><a name="p1418915311587"></a>void updateTParams(int l2Probe, int l3SegmentNum);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>测试时传入三级检索所需参数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b95897151286"><a name="b95897151286"></a><a name="b95897151286"></a>int l2Probe</strong>：二级检索选择子桶的数量，默认值为<span class="parmvalue" id="parmvalue81701953182510"><a name="parmvalue81701953182510"></a><a name="parmvalue81701953182510"></a>“48”</span>。</p>
<p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b439191185"><a name="b439191185"></a><a name="b439191185"></a>int l3SegmentNum</strong>：L3算子处理的段数，影响查找的base总数，默认值为<span class="parmvalue" id="parmvalue1345127261"><a name="parmvalue1345127261"></a><a name="parmvalue1345127261"></a>“96”</span>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li>nprobe ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }</li><li>l2Probe ≥ nprobe, l2Probe≤ l3SegmentNum, l2Probe≤ nprobe * 64</li><li>l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }</li><li>nprobe的设置可参见<a href="#setsearchparams接口">setSearchParams</a>。</li><li>updateTParams接口预计2026年9月废除，请使用<a href="#setsearchparams接口">setSearchParams</a>。</li></ul>
</td>
</tr>
</tbody>
</table>

### AscendIndexIVFSQTConfig<a name="ZH-CN_TOPIC_0000001506495881"></a>

AscendIndexIVFSQT需要使用对应的AscendIndexIVFSQTConfig执行对应资源的初始化。

**AscendIndexIVFSQTConfig<a name="section6579185362314"></a>**

> [!NOTE] 说明 
>AscendIndexIVFSQTConfig继承于[AscendIndexIVFSQConfig](#ascendindexivfsqconfig)。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>inline AscendIndexIVFSQTConfig();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>默认构造函数，默认devices为{0}，使用第0个<span id="ph79732210444"><a name="ph79732210444"></a><a name="ph79732210444"></a>昇腾AI处理器</span>进行计算，默认resource为384MB。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
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

<a name="table42413462115"></a>
<table><tbody><tr id="row1524133414212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1224153422117"><a name="p1224153422117"></a><a name="p1224153422117"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14172141316118"><a name="p14172141316118"></a><a name="p14172141316118"></a>inline AscendIndexIVFSQTConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</p>
</td>
</tr>
<tr id="row72433412120"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p324153419217"><a name="p324153419217"></a><a name="p324153419217"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQTConfig的构造函数，生成AscendIndexIVFSQTConfig，此时根据<span class="parmname" id="parmname1788912146594"><a name="parmname1788912146594"></a><a name="parmname1788912146594"></a>“devices”</span>中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认的初始化。</p>
</td>
</tr>
<tr id="row124103412219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p72414340215"><a name="p72414340215"></a><a name="p72414340215"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1624123417213"><a name="p1624123417213"></a><a name="p1624123417213"></a><strong id="b20622101413581"><a name="b20622101413581"></a><a name="b20622101413581"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b2777720145818"><a name="b2777720145818"></a><a name="b2777720145818"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname742452385810"><a name="parmname742452385810"></a><a name="parmname742452385810"></a>“IVFSQT_DEFAULT_TEMP_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row62417348212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p142463482114"><a name="p142463482114"></a><a name="p142463482114"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p624134112112"><a name="p624134112112"></a><a name="p624134112112"></a>无</p>
</td>
</tr>
<tr id="row202443411211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p225193492114"><a name="p225193492114"></a><a name="p225193492114"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16255347215"><a name="p16255347215"></a><a name="p16255347215"></a>无</p>
</td>
</tr>
<tr id="row9251334172114"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11251534172114"><a name="p11251534172114"></a><a name="p11251534172114"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1626263017587"></a><a name="ul1626263017587"></a><ul id="ul1626263017587"><li><span class="parmname" id="parmname512943295811"><a name="parmname512943295811"></a><a name="parmname512943295811"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname162541419202"><a name="parmname162541419202"></a><a name="parmname162541419202"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table0812225238"></a>
<table><tbody><tr id="row681152292314"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p181722132314"><a name="p181722132314"></a><a name="p181722132314"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15811522152312"><a name="p15811522152312"></a><a name="p15811522152312"></a>inline AscendIndexIVFSQTConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</p>
</td>
</tr>
<tr id="row681722142311"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p12811722132310"><a name="p12811722132310"></a><a name="p12811722132310"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1981132202311"><a name="p1981132202311"></a><a name="p1981132202311"></a>AscendIndexIVFSQTConfig的构造函数，生成AscendIndexIVFSQTConfig，此时根据<span class="parmname" id="parmname223102255918"><a name="parmname223102255918"></a><a name="parmname223102255918"></a>“devices”</span>中配置的值设置Device侧<span id="ph78117225236"><a name="ph78117225236"></a><a name="ph78117225236"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认的初始化。</p>
</td>
</tr>
<tr id="row158132218234"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1481522122316"><a name="p1481522122316"></a><a name="p1481522122316"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19811822132311"><a name="p19811822132311"></a><a name="p19811822132311"></a><strong id="b1316015619589"><a name="b1316015619589"></a><a name="b1316015619589"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p1081182213237"><a name="p1081182213237"></a><a name="p1081182213237"></a><strong id="b9205192205911"><a name="b9205192205911"></a><a name="b9205192205911"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname128498505910"><a name="parmname128498505910"></a><a name="parmname128498505910"></a>“IVFSQT_DEFAULT_TEMP_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row281522152313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18811922152314"><a name="p18811922152314"></a><a name="p18811922152314"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1381192212233"><a name="p1381192212233"></a><a name="p1381192212233"></a>无</p>
</td>
</tr>
<tr id="row4811922172317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p28113227234"><a name="p28113227234"></a><a name="p28113227234"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1181322132318"><a name="p1181322132318"></a><a name="p1181322132318"></a>无</p>
</td>
</tr>
<tr id="row081622142316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p18811622102319"><a name="p18811622102319"></a><a name="p18811622102319"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1092012916596"></a><a name="ul1092012916596"></a><ul id="ul1092012916596"><li><span class="parmname" id="parmname169381516145920"><a name="parmname169381516145920"></a><a name="parmname169381516145920"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname98182212319"><a name="parmname98182212319"></a><a name="parmname98182212319"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

**SetDefaultIVFSQConfig<a name="section18396165022414"></a>**

<a name="table14953182017255"></a>
<table><tbody><tr id="row1495372015250"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p495312072515"><a name="p495312072515"></a><a name="p495312072515"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58281361827"><a name="p58281361827"></a><a name="p58281361827"></a>inline void SetDefaultIVFSQConfig();</p>
</td>
</tr>
<tr id="row69531020142513"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p14953152072517"><a name="p14953152072517"></a><a name="p14953152072517"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p19535201256"><a name="p19535201256"></a><a name="p19535201256"></a>执行默认的初始化，设置迭代数为16，每个centroids最多设置512个点。</p>
</td>
</tr>
<tr id="row495362020258"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5953182032513"><a name="p5953182032513"></a><a name="p5953182032513"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p3953420122514"><a name="p3953420122514"></a><a name="p3953420122514"></a>无</p>
</td>
</tr>
<tr id="row1895392013254"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p139538203259"><a name="p139538203259"></a><a name="p139538203259"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p895322019257"><a name="p895322019257"></a><a name="p895322019257"></a>无</p>
</td>
</tr>
<tr id="row109531520182518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p995312017256"><a name="p995312017256"></a><a name="p995312017256"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7953152032518"><a name="p7953152032518"></a><a name="p7953152032518"></a>无</p>
</td>
</tr>
<tr id="row1795392052510"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1095382052518"><a name="p1095382052518"></a><a name="p1095382052518"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p2095302062513"><a name="p2095302062513"></a><a name="p2095302062513"></a>无</p>
</td>
</tr>
</tbody>
</table>

### AscendIndexVStar<a name="ZH-CN_TOPIC_0000002044351677"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000002044510693"></a>

昇腾自研向量检索算法，为用户提供昇腾侧高维大底库近似检索能力。使用自研矩阵近似策略，压缩特征向量后存底库，最后使用自研检索策略在底库中检索得到topK个最近似向量结果。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

此算法主要针对大底库场景的近似模糊搜索，相较暴力检索精度已有一定损失。在小底库场景，建议适当加大超参值，可改善精度损失问题。

#### AscendIndexVStar接口<a name="ZH-CN_TOPIC_0000002044513265"></a>

> [!NOTE] 说明 
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p11443153591119"><a name="p11443153591119"></a><a name="p11443153591119"></a><strong id="b16412146163213"><a name="b16412146163213"></a><a name="b16412146163213"></a>const AscendIndexVstarInitParams&amp; params</strong>：构造配置参数，具体请参见<a href="#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p127451568374"><a name="p127451568374"></a><a name="p127451568374"></a>具体请参见<a href="#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>。</p>
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

#### LoadIndex接口<a name="ZH-CN_TOPIC_0000002008232688"></a>

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

#### WriteIndex接口<a name="ZH-CN_TOPIC_0000002044351681"></a>

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

#### AddCodeBooksByIndex接口<a name="ZH-CN_TOPIC_0000002044510697"></a>

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

#### AddCodeBooksByPath接口<a name="ZH-CN_TOPIC_0000002008390980"></a>

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

#### Add接口<a name="ZH-CN_TOPIC_0000002008232692"></a>

<a name="table18288921121213"></a>
<table><tbody><tr id="row63171421121210"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.1.1"><p id="p1931702111125"><a name="p1931702111125"></a><a name="p1931702111125"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.1.1 "><p id="p2317721111218"><a name="p2317721111218"></a><a name="p2317721111218"></a>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseData);</p>
</td>
</tr>
<tr id="row031782114123"><th class="firstcol" valign="top" width="20.32%" id="mcps1.1.3.2.1"><p id="p13177211124"><a name="p13177211124"></a><a name="p13177211124"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.67999999999999%" headers="mcps1.1.3.2.1 "><p id="p331702181218"><a name="p331702181218"></a><a name="p331702181218"></a>实现AscendIndexVStar建库和往底库中添加新的特征向量的功能。</p>
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

> [!NOTE] 说明 
>
>- Add接口不能与AddWithIds接口混用。
>- 使用Add接口后，Search结果的labels可能会重复，如果业务上对label有要求，建议使用[AddWithIds接口](#addwithids接口)。

#### AddWithIds接口<a name="ZH-CN_TOPIC_0000002044351685"></a>

<a name="table32483414124"></a>
<table><tbody><tr id="row856113412127"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1156034171214"><a name="p1156034171214"></a><a name="p1156034171214"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p256183451212"><a name="p256183451212"></a><a name="p256183451212"></a>APP_ERROR AddWithIds(const std::vector&lt;float&gt;&amp; baseData, const std::vector&lt;int64_t&gt;&amp; ids);</p>
</td>
</tr>
<tr id="row145683451213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p05683441216"><a name="p05683441216"></a><a name="p05683441216"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p12561734201214"><a name="p12561734201214"></a><a name="p12561734201214"></a>实现AscendIndexVStar建库和往底库中添加新的特征向量的功能。允许用户指定添加底库的ID。</p>
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

#### DeleteByIds接口<a name="ZH-CN_TOPIC_0000002044510701"></a>

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

#### DeleteById接口<a name="ZH-CN_TOPIC_0000002008390984"></a>

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

#### DeleteByRange接口<a name="ZH-CN_TOPIC_0000002008232696"></a>

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

#### Search接口<a name="ZH-CN_TOPIC_0000002044351689"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p18254194711350"><a name="p18254194711350"></a><a name="p18254194711350"></a><strong id="b10734716153611"><a name="b10734716153611"></a><a name="b10734716153611"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="#ascendindexsearchparams接口">AscendIndexSearchParams</a>。</p>
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

#### SearchWithMask接口<a name="ZH-CN_TOPIC_0000002044510705"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p18254194711350"><a name="p18254194711350"></a><a name="p18254194711350"></a><strong id="b8266155353513"><a name="b8266155353513"></a><a name="b8266155353513"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="#ascendindexsearchparams接口">AscendIndexSearchParams</a></p>
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

#### MultiSearch接口<a name="ZH-CN_TOPIC_0000002008390988"></a>

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
<p id="p18254194711350"><a name="p18254194711350"></a><a name="p18254194711350"></a><strong id="b7965111993515"><a name="b7965111993515"></a><a name="b7965111993515"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="#ascendindexsearchparams接口">AscendIndexSearchParams</a></p>
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

#### MultiSearchWithMask接口<a name="ZH-CN_TOPIC_0000002008232700"></a>

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
<p id="p1112710311432"><a name="p1112710311432"></a><a name="p1112710311432"></a><strong id="b14830144543418"><a name="b14830144543418"></a><a name="b14830144543418"></a>const AscendIndexSearchParams&amp; params</strong>：检索参数，具体请参见<a href="#ascendindexsearchparams接口">AscendIndexSearchParams</a>。</p>
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

#### SetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044351693"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14891331164012"><a name="p14891331164012"></a><a name="p14891331164012"></a><strong id="b92771420133413"><a name="b92771420133413"></a><a name="b92771420133413"></a>const AscendIndexVstarHyperParams&amp; params</strong>：检索时超参，具体请见<a href="#ascendindexvstarhyperparams接口">AscendIndexVstarHyperParams</a>。</p>
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

#### GetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044510709"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111384864011"><a name="p111384864011"></a><a name="p111384864011"></a><strong id="b5901151112341"><a name="b5901151112341"></a><a name="b5901151112341"></a>AscendIndexVstarHyperParams&amp; params</strong>：检索时超参，具体请见<a href="#ascendindexvstarhyperparams接口">AscendIndexVstarHyperParams</a>。</p>
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

#### GetDim接口<a name="ZH-CN_TOPIC_0000002008390992"></a>

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

#### GetNTotal接口<a name="ZH-CN_TOPIC_0000002008232704"></a>

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

#### Reset接口<a name="ZH-CN_TOPIC_0000002044351697"></a>

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

#### operator= 接口<a name="ZH-CN_TOPIC_0000002008390996"></a>

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

### AscendIndexGreat<a name="ZH-CN_TOPIC_0000002044829945"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000002008751966"></a>

自研向量检索算法，为用户提供昇腾侧和鲲鹏侧高维大底库近似检索能力。使用自研检索策略在底库中检索得到topK个最近似向量结果。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

此算法主要针对大底库场景的近似模糊搜索，相较暴力检索精度已有一定损失。在小底库场景，建议适当加大超参值，可改善精度损失问题。

> [!NOTE] 说明 
>
>- 创建Index实例时传入的参数params，需根据实际情况设置其中的dim。
>- Index分为两种算法模式：KMode仅使用鲲鹏侧算法，AKMode昇腾加鲲鹏算法，在AKMode模式下需要提前生成对应算子。
>- subSpaceDimnlist应与码本训练时对应参数保持一致。

#### AscendIndexGreat接口<a name="ZH-CN_TOPIC_0000002044829953"></a>

<a name="table5404639201712"></a>
<table><tbody><tr id="row194338394172"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p843363911171"><a name="p843363911171"></a><a name="p843363911171"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p144331039111710"><a name="p144331039111710"></a><a name="p144331039111710"></a>AscendIndexGreat(const std::string&amp; mode, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</p>
</td>
</tr>
<tr id="row043313981717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p24337391179"><a name="p24337391179"></a><a name="p24337391179"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6433139151710"><a name="p6433139151710"></a><a name="p6433139151710"></a>AscendIndexGreat的构造函数，创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row124339399175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p164331391173"><a name="p164331391173"></a><a name="p164331391173"></a><strong id="b5433193991713"><a name="b5433193991713"></a><a name="b5433193991713"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12433193918175"><a name="p12433193918175"></a><a name="p12433193918175"></a><strong id="b16722123816518"><a name="b16722123816518"></a><a name="b16722123816518"></a>const std::string&amp; mode</strong>：指定算法模式。</p>
<p id="p104331039161715"><a name="p104331039161715"></a><a name="p104331039161715"></a><strong id="b769912415512"><a name="b769912415512"></a><a name="b769912415512"></a>const std::vector&lt;int&gt;&amp; deviceList</strong>：指定的NPU侧设备ID。</p>
<p id="p204336391174"><a name="p204336391174"></a><a name="p204336391174"></a><strong id="b10327344205112"><a name="b10327344205112"></a><a name="b10327344205112"></a>bool verbose</strong>：指定是否开启verbose选项，开启后部分操作提供额外的打印提示。默认值为<span class="parmvalue" id="parmvalue810213723716"><a name="parmvalue810213723716"></a><a name="parmvalue810213723716"></a>“false”</span>。</p>
</td>
</tr>
<tr id="row443343901718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p8433133912177"><a name="p8433133912177"></a><a name="p8433133912177"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p204331639171718"><a name="p204331639171718"></a><a name="p204331639171718"></a>无</p>
</td>
</tr>
<tr id="row1443373991714"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p743343921714"><a name="p743343921714"></a><a name="p743343921714"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p174331439151714"><a name="p174331439151714"></a><a name="p174331439151714"></a>无</p>
</td>
</tr>
<tr id="row04331639141719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p114331039171716"><a name="p114331039171716"></a><a name="p114331039171716"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1462219489335"></a><a name="ul1462219489335"></a><ul id="ul1462219489335"><li>mode：只支持“KMode”和“AKMode”两种模式。</li><li>deviceList：请使用<strong id="b3555159175110"><a name="b3555159175110"></a><a name="b3555159175110"></a>npu-smi</strong>命令查询对应的NPUID，仅支持一个device设备ID。</li><li>使用此构造函数创建Index实例后，需要先调用“LoadIndex”加载事先落盘后的Index实例，然后再进行其他操作。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table72261454131719"></a>
<table><tbody><tr id="row18251175431713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16251154121710"><a name="p16251154121710"></a><a name="p16251154121710"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p76841847185413"><a name="p76841847185413"></a><a name="p76841847185413"></a>explicit AscendIndexGreat(const AscendIndexGreatInitParams&amp; kModeInitParams);</p>
</td>
</tr>
<tr id="row11251125451717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p172516541177"><a name="p172516541177"></a><a name="p172516541177"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p11251135461713"><a name="p11251135461713"></a><a name="p11251135461713"></a>AscendIndexGreat的构造函数，创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row82514548179"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p325155441716"><a name="p325155441716"></a><a name="p325155441716"></a><strong id="b2251254161713"><a name="b2251254161713"></a><a name="b2251254161713"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p82516541176"><a name="p82516541176"></a><a name="p82516541176"></a>Index所需的初始化参数kModeInitParams，具体请参见<a href="#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>。</p>
</td>
</tr>
<tr id="row3251354151719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2252115451717"><a name="p2252115451717"></a><a name="p2252115451717"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1125255410178"><a name="p1125255410178"></a><a name="p1125255410178"></a>无</p>
</td>
</tr>
<tr id="row1725275471717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2252054121720"><a name="p2252054121720"></a><a name="p2252054121720"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row13252145413171"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p125285413172"><a name="p125285413172"></a><a name="p125285413172"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20252954121720"><a name="p20252954121720"></a><a name="p20252954121720"></a>参见<a href="#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>中的参数说明和参数约束。</p>
</td>
</tr>
</tbody>
</table>

<a name="table198261931819"></a>
<table><tbody><tr id="row78491591183"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1484915918184"><a name="p1484915918184"></a><a name="p1484915918184"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1584913901820"><a name="p1584913901820"></a><a name="p1584913901820"></a>AscendIndexGreat(const AscendIndexVstarInitParams&amp; aModeInitParams, const AscendIndexGreatInitParams&amp; kModeInitParams);</p>
</td>
</tr>
<tr id="row284999121814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p78498919183"><a name="p78498919183"></a><a name="p78498919183"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13849791188"><a name="p13849791188"></a><a name="p13849791188"></a>AscendIndexGreat的构造函数，创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row784912981810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1184919911189"><a name="p1184919911189"></a><a name="p1184919911189"></a><strong id="b9849692189"><a name="b9849692189"></a><a name="b9849692189"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1984999111812"><a name="p1984999111812"></a><a name="p1984999111812"></a>Index所需的初始化参数aModeInitParams和kModeInitParams，具体请参见<a href="#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>和<a href="#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>。</p>
</td>
</tr>
<tr id="row5850179121814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1285009121812"><a name="p1285009121812"></a><a name="p1285009121812"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14850159181811"><a name="p14850159181811"></a><a name="p14850159181811"></a>无</p>
</td>
</tr>
<tr id="row13850199151817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1185019912183"><a name="p1185019912183"></a><a name="p1185019912183"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p123604384355"><a name="p123604384355"></a><a name="p123604384355"></a><strong id="b536183815357"><a name="b536183815357"></a><a name="b536183815357"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row16850109161814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p985029101810"><a name="p985029101810"></a><a name="p985029101810"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p12850169151811"><a name="p12850169151811"></a><a name="p12850169151811"></a>参考<a href="#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>和<a href="#ascendindexcodebookinitparams接口">AscendIndexGreatInitParams</a>中的参数说明和参数约束。</p>
<p id="p122218114501"><a name="p122218114501"></a><a name="p122218114501"></a>aModeInitParams和kModeInitParams的dim必须保持一致。</p>
</td>
</tr>
</tbody>
</table>

<a name="table32891532172215"></a>
<table><tbody><tr id="row1731883213226"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p183181632182220"><a name="p183181632182220"></a><a name="p183181632182220"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3318832182215"><a name="p3318832182215"></a><a name="p3318832182215"></a>AscendIndexGreat(const AscendIndexGreat&amp;) = delete;</p>
</td>
</tr>
<tr id="row831813213224"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p231819329229"><a name="p231819329229"></a><a name="p231819329229"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6318232102217"><a name="p6318232102217"></a><a name="p6318232102217"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row731817327226"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p631813272217"><a name="p631813272217"></a><a name="p631813272217"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p153181732102215"><a name="p153181732102215"></a><a name="p153181732102215"></a><strong id="b113181932132210"><a name="b113181932132210"></a><a name="b113181932132210"></a>const AscendIndexGreat&amp;</strong>：常量AscendIndexGreat对象。</p>
</td>
</tr>
<tr id="row20318632102217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p431833212210"><a name="p431833212210"></a><a name="p431833212210"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15318173218228"><a name="p15318173218228"></a><a name="p15318173218228"></a>无</p>
</td>
</tr>
<tr id="row8318153215220"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1831811326227"><a name="p1831811326227"></a><a name="p1831811326227"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p83180322222"><a name="p83180322222"></a><a name="p83180322222"></a>无</p>
</td>
</tr>
<tr id="row131933214228"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p103190320223"><a name="p103190320223"></a><a name="p103190320223"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p7319133272214"><a name="p7319133272214"></a><a name="p7319133272214"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### \~AscendIndexGreat接口<a name="ZH-CN_TOPIC_0000002013257524"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p128341740125117"><a name="p128341740125117"></a><a name="p128341740125117"></a>virtual ~AscendIndexGreat() = default;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexGreat的析构函数，销毁AscendIndexGreat对象，释放资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
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

#### operator =接口<a name="ZH-CN_TOPIC_0000002008751990"></a>

<a name="table39961720122213"></a>
<table><tbody><tr id="row3176213227"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p71752119228"><a name="p71752119228"></a><a name="p71752119228"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1917321192217"><a name="p1917321192217"></a><a name="p1917321192217"></a>AscendIndexGreat &amp;operator=(const AscendIndexGreat&amp;) = delete;</p>
</td>
</tr>
<tr id="row111762152213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p71719215225"><a name="p71719215225"></a><a name="p71719215225"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p817921142214"><a name="p817921142214"></a><a name="p817921142214"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row1217121122217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p41710213220"><a name="p41710213220"></a><a name="p41710213220"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p111732182220"><a name="p111732182220"></a><a name="p111732182220"></a><strong id="b101742192218"><a name="b101742192218"></a><a name="b101742192218"></a>const AscendIndexGreat&amp;</strong>：常量AscendIndexGreat对象。</p>
</td>
</tr>
<tr id="row3171321172218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1717521132215"><a name="p1717521132215"></a><a name="p1717521132215"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p191714215226"><a name="p191714215226"></a><a name="p191714215226"></a>无</p>
</td>
</tr>
<tr id="row181713210221"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p917182117222"><a name="p917182117222"></a><a name="p917182117222"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p917921142212"><a name="p917921142212"></a><a name="p917921142212"></a>无</p>
</td>
</tr>
<tr id="row117172118229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p618162116224"><a name="p618162116224"></a><a name="p618162116224"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1318182111227"><a name="p1318182111227"></a><a name="p1318182111227"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### Add接口<a name="ZH-CN_TOPIC_0000002044950953"></a>

<a name="table11133547191811"></a>
<table><tbody><tr id="row1159447111810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12159184761818"><a name="p12159184761818"></a><a name="p12159184761818"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161591747181812"><a name="p161591747181812"></a><a name="p161591747181812"></a>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseRawData);</p>
</td>
</tr>
<tr id="row10159194716180"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p5159847181814"><a name="p5159847181814"></a><a name="p5159847181814"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015916478183"><a name="p2015916478183"></a><a name="p2015916478183"></a>向AscendIndexGreat底库中添加新的特征向量。</p>
</td>
</tr>
<tr id="row11159847131815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p115914712182"><a name="p115914712182"></a><a name="p115914712182"></a><strong id="b615918476186"><a name="b615918476186"></a><a name="b615918476186"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1215974741811"><a name="p1215974741811"></a><a name="p1215974741811"></a><strong id="b14159184715188"><a name="b14159184715188"></a><a name="b14159184715188"></a>const std::vector&lt;float&gt;&amp; baseRawData：</strong>添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row11159647171819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18159184718181"><a name="p18159184718181"></a><a name="p18159184718181"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2159147101812"><a name="p2159147101812"></a><a name="p2159147101812"></a>无</p>
</td>
</tr>
<tr id="row615904711813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1815954712186"><a name="p1815954712186"></a><a name="p1815954712186"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row2160144719188"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19160947121812"><a name="p19160947121812"></a><a name="p19160947121812"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul19630537566"></a><a name="ul19630537566"></a><ul id="ul19630537566"><li>此处数组“baseRawData”的长度应该为dim * nTotal。nTotal为准备添加进入底库内部的向量数量，dim为每个向量的维度。</li><li>底库向量总数的取值范围：10000 ≤ nTotal ≤ 1e8。</li><li>该算法不支持添加完底库之后再次添加。Add接口不能与AddWithIds接口混用。</li></ul>
</td>
</tr>
</tbody>
</table>

#### AddWithIds接口<a name="ZH-CN_TOPIC_0000002044829957"></a>

<a name="table2436200181918"></a>
<table><tbody><tr id="row6468120161919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p164681020199"><a name="p164681020199"></a><a name="p164681020199"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13468150121916"><a name="p13468150121916"></a><a name="p13468150121916"></a>APP_ERROR AddWithIds (const std::vector&lt;float&gt;&amp; baseRawData, const std::vector&lt;int64_t&gt;&amp; ids);</p>
</td>
</tr>
<tr id="row1846914041914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p546913041920"><a name="p546913041920"></a><a name="p546913041920"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1646980171915"><a name="p1646980171915"></a><a name="p1646980171915"></a>向AscendIndexGreat底库中添加新的特征向量。使用AddWithIds接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
</td>
</tr>
<tr id="row174691500199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5469130151918"><a name="p5469130151918"></a><a name="p5469130151918"></a><strong id="b2469170181919"><a name="b2469170181919"></a><a name="b2469170181919"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul14839152610572"></a><a name="ul14839152610572"></a><ul id="ul14839152610572"><li><strong id="b480732014573"><a name="b480732014573"></a><a name="b480732014573"></a>const std::vector&lt;float&gt;&amp; baseRawData</strong>：添加进底库的特征向量。</li><li><strong id="b41751123125714"><a name="b41751123125714"></a><a name="b41751123125714"></a>const std::vector&lt;int64_t&gt;&amp; ids</strong>：添加进底库的特征向量ID。ID在Index实例中需唯一。</li></ul>
</td>
</tr>
<tr id="row64692010199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1446920091916"><a name="p1446920091916"></a><a name="p1446920091916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1046918051915"><a name="p1046918051915"></a><a name="p1046918051915"></a>无</p>
</td>
</tr>
<tr id="row34692006195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1546910171910"><a name="p1546910171910"></a><a name="p1546910171910"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row5469140141913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p154691800193"><a name="p154691800193"></a><a name="p154691800193"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul112903341573"></a><a name="ul112903341573"></a><ul id="ul112903341573"><li>此处数组“baseRawData”的长度应该为dim * nTotal。nTotal为准备添加进入底库内部的向量数量，dim为每个向量的维度。</li><li>底库向量总数的取值范围：10000 ≤ nTotal ≤ 1e8。</li><li><span class="parmname" id="parmname1942422062518"><a name="parmname1942422062518"></a><a name="parmname1942422062518"></a>“ids”</span>长度必须为nTotal，用户需要根据自己的业务场景，保证<span class="parmname" id="parmname20685131262110"><a name="parmname20685131262110"></a><a name="parmname20685131262110"></a>“ids”</span>的合法性，如底库中存在重复的ID，检索结果中的"label"将无法对应具体的底库向量。</li><li>该算法不支持添加完底库之后再次添加。AddWithIds接口不能与Add接口混用。</li></ul>
</td>
</tr>
</tbody>
</table>

#### LoadIndex接口<a name="ZH-CN_TOPIC_0000002008751978"></a>

<a name="table17789162191912"></a>
<table><tbody><tr id="row8827202101911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18271621181913"><a name="p18271621181913"></a><a name="p18271621181913"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p20827821181912"><a name="p20827821181912"></a><a name="p20827821181912"></a>APP_ERROR LoadIndex(const std::string&amp; indexPath);</p>
</td>
</tr>
<tr id="row158271121181911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1482782119190"><a name="p1482782119190"></a><a name="p1482782119190"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p282792116190"><a name="p282792116190"></a><a name="p282792116190"></a>将Index结构从磁盘读入，包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row2082762118194"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p282832121910"><a name="p282832121910"></a><a name="p282832121910"></a><strong id="b178281421121916"><a name="b178281421121916"></a><a name="b178281421121916"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1828121191916"><a name="p1828121191916"></a><a name="p1828121191916"></a><strong id="b208843453576"><a name="b208843453576"></a><a name="b208843453576"></a>const std::string&amp; indexPath</strong>：加载KMode索引的路径。</p>
</td>
</tr>
<tr id="row168281213195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p4828621151916"><a name="p4828621151916"></a><a name="p4828621151916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p182842113195"><a name="p182842113195"></a><a name="p182842113195"></a>无</p>
</td>
</tr>
<tr id="row138282021121914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p8828521151915"><a name="p8828521151915"></a><a name="p8828521151915"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1282819214195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p198281721161912"><a name="p198281721161912"></a><a name="p198281721161912"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p8828122191913"><a name="p8828122191913"></a><a name="p8828122191913"></a>“indexPath”对应的文件为调用WriteIndex方法得到的落盘文件，程序执行用户对其有读权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

<a name="table98570373191"></a>
<table><tbody><tr id="row17884153751918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p15884537171910"><a name="p15884537171910"></a><a name="p15884537171910"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2884737121918"><a name="p2884737121918"></a><a name="p2884737121918"></a>APP_ERROR LoadIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</p>
</td>
</tr>
<tr id="row38841379192"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1588453721918"><a name="p1588453721918"></a><a name="p1588453721918"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1488510377191"><a name="p1488510377191"></a><a name="p1488510377191"></a>将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和原始数据。</p>
</td>
</tr>
<tr id="row888533717196"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128851337151917"><a name="p128851337151917"></a><a name="p128851337151917"></a><strong id="b588573715193"><a name="b588573715193"></a><a name="b588573715193"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1588563711918"><a name="p1588563711918"></a><a name="p1588563711918"></a><strong id="b89721956105715"><a name="b89721956105715"></a><a name="b89721956105715"></a>const std::string&amp; aModeIndexPath</strong>：加载AMode索引的路径。</p>
<p id="p11885193717191"><a name="p11885193717191"></a><a name="p11885193717191"></a><strong id="b1920219018585"><a name="b1920219018585"></a><a name="b1920219018585"></a>const std::string&amp; kModeIndexPath</strong>：加载KMode索引的路径。</p>
</td>
</tr>
<tr id="row7885163731911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2885133721916"><a name="p2885133721916"></a><a name="p2885133721916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2088593741910"><a name="p2088593741910"></a><a name="p2088593741910"></a>无</p>
</td>
</tr>
<tr id="row208858371192"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p288543731913"><a name="p288543731913"></a><a name="p288543731913"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p19824113093620"><a name="p19824113093620"></a><a name="p19824113093620"></a><strong id="b1482412303365"><a name="b1482412303365"></a><a name="b1482412303365"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row788513751910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12885173771912"><a name="p12885173771912"></a><a name="p12885173771912"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20885143791917"><a name="p20885143791917"></a><a name="p20885143791917"></a>“aModeIndexPath”和“kModeIndexPath”对应的文件为调用WriteIndex方法得到的落盘文件，程序执行用户对其有读权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

#### WriteIndex接口<a name="ZH-CN_TOPIC_0000002044950957"></a>

<a name="table84194504191"></a>
<table><tbody><tr id="row1244255016194"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11442165011196"><a name="p11442165011196"></a><a name="p11442165011196"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1442950161914"><a name="p1442950161914"></a><a name="p1442950161914"></a>APP_ERROR WriteIndex(const std::string&amp; indexPath);</p>
</td>
</tr>
<tr id="row4442135021918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p154421650201911"><a name="p154421650201911"></a><a name="p154421650201911"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p94421506192"><a name="p94421506192"></a><a name="p94421506192"></a>将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row19442050191916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p4442205041914"><a name="p4442205041914"></a><a name="p4442205041914"></a><strong id="b1044275020194"><a name="b1044275020194"></a><a name="b1044275020194"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6442250101914"><a name="p6442250101914"></a><a name="p6442250101914"></a>无</p>
</td>
</tr>
<tr id="row134421050151911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p444385091919"><a name="p444385091919"></a><a name="p444385091919"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p11443650111916"><a name="p11443650111916"></a><a name="p11443650111916"></a><strong id="b1940413117581"><a name="b1940413117581"></a><a name="b1940413117581"></a>const std::string&amp; indexPath</strong>：写入KMode索引的路径。</p>
</td>
</tr>
<tr id="row1844395011910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p644345031917"><a name="p644345031917"></a><a name="p644345031917"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1144355061917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p344375061912"><a name="p344375061912"></a><a name="p344375061912"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6443450161914"><a name="p6443450161914"></a><a name="p6443450161914"></a>用户需要保证“indexPath”文件路径所在的目录存在，且执行用户对目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

<a name="table14392122132014"></a>
<table><tbody><tr id="row441919215201"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9419132122010"><a name="p9419132122010"></a><a name="p9419132122010"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1741916213205"><a name="p1741916213205"></a><a name="p1741916213205"></a>APP_ERROR WriteIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</p>
</td>
</tr>
<tr id="row1141920242016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p134192292018"><a name="p134192292018"></a><a name="p134192292018"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p142017252016"><a name="p142017252016"></a><a name="p142017252016"></a>将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row18420827206"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p124203215208"><a name="p124203215208"></a><a name="p124203215208"></a><strong id="b4420132162013"><a name="b4420132162013"></a><a name="b4420132162013"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p18420520202"><a name="p18420520202"></a><a name="p18420520202"></a>无</p>
</td>
</tr>
<tr id="row642013252016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p154202232018"><a name="p154202232018"></a><a name="p154202232018"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><a name="ul71911043155818"></a><a name="ul71911043155818"></a><ul id="ul71911043155818"><li>const std::string&amp; aModeIndexPath：写入AMode索引的路径。</li><li>const std::string&amp; kModeIndexPath：写入KMode索引的路径。</li></ul>
</td>
</tr>
<tr id="row1842013232015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p842012215201"><a name="p842012215201"></a><a name="p842012215201"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7875154210361"><a name="p7875154210361"></a><a name="p7875154210361"></a><strong id="b187554215365"><a name="b187554215365"></a><a name="b187554215365"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row142011213209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1242010282015"><a name="p1242010282015"></a><a name="p1242010282015"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p54203216202"><a name="p54203216202"></a><a name="p54203216202"></a>用户需要保证“aModeIndexPath”和“kModeIndexPath”文件路径所在的目录存在，且执行用户对目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

#### AddCodeBooks接口<a name="ZH-CN_TOPIC_0000002008751982"></a>

<a name="table339181620207"></a>
<table><tbody><tr id="row20640163209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1564616122013"><a name="p1564616122013"></a><a name="p1564616122013"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3641716102017"><a name="p3641716102017"></a><a name="p3641716102017"></a>APP_ERROR AddCodeBooks(const std::string&amp; codeBooksPath);</p>
</td>
</tr>
<tr id="row66411167203"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p2064121632013"><a name="p2064121632013"></a><a name="p2064121632013"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p964716102017"><a name="p964716102017"></a><a name="p964716102017"></a>加载已经生成完毕的码本到Index。</p>
</td>
</tr>
<tr id="row7647167203"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p764101617204"><a name="p764101617204"></a><a name="p764101617204"></a><strong id="b1664111618206"><a name="b1664111618206"></a><a name="b1664111618206"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1364116172012"><a name="p1364116172012"></a><a name="p1364116172012"></a><strong id="b178911721592"><a name="b178911721592"></a><a name="b178911721592"></a>const std::string&amp; codeBooksPath</strong>：加载已经生成完毕的码本路径。</p>
</td>
</tr>
<tr id="row1564121612202"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p46441612208"><a name="p46441612208"></a><a name="p46441612208"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1064916152011"><a name="p1064916152011"></a><a name="p1064916152011"></a>无</p>
</td>
</tr>
<tr id="row76410163209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p764016192011"><a name="p764016192011"></a><a name="p764016192011"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row16641816182015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15640163208"><a name="p15640163208"></a><a name="p15640163208"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15641164207"><a name="p15641164207"></a><a name="p15641164207"></a>该接口仅能在索引初始化“AKMode”时使用。</p>
<p id="p1865141682018"><a name="p1865141682018"></a><a name="p1865141682018"></a>用户应该保证“codeBooksPath”文件路径所在的目录存在，且该文件内容必须为有效的码本。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

#### Search接口<a name="ZH-CN_TOPIC_0000002008910274"></a>

<a name="table537563852013"></a>
<table><tbody><tr id="row04171138192013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16417103813201"><a name="p16417103813201"></a><a name="p16417103813201"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1241714381201"><a name="p1241714381201"></a><a name="p1241714381201"></a>APP_ERROR Search(const AscendIndexSearchParams&amp; searchParams);</p>
</td>
</tr>
<tr id="row9417173892011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3417123810208"><a name="p3417123810208"></a><a name="p3417123810208"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p741743817204"><a name="p741743817204"></a><a name="p741743817204"></a>实现AscendIndexGreat特征向量查询接口，根据输入的特征向量返回最相似的“topK”条特征的距离及ID。</p>
</td>
</tr>
<tr id="row16417123814205"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7417173812013"><a name="p7417173812013"></a><a name="p7417173812013"></a><strong id="b4417738192010"><a name="b4417738192010"></a><a name="b4417738192010"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p963412085712"><a name="p963412085712"></a><a name="p963412085712"></a>searchParams结构体见<a href="#ascendindexsearchparams接口">AscendIndexSearchParams接口</a>。</p>
<p id="p12417638202012"><a name="p12417638202012"></a><a name="p12417638202012"></a><strong id="b4417163812018"><a name="b4417163812018"></a><a name="b4417163812018"></a>size_t n：</strong>查询的特征向量的条数<strong id="b741718386209"><a name="b741718386209"></a><a name="b741718386209"></a>。</strong></p>
<p id="p101561712152015"><a name="p101561712152015"></a><a name="p101561712152015"></a><strong id="b5821914326"><a name="b5821914326"></a><a name="b5821914326"></a>std::vector&lt;float&gt;&amp; queryData：</strong>特征向量数据<strong id="b24171438182010"><a name="b24171438182010"></a><a name="b24171438182010"></a>。</strong></p>
<p id="p124173383205"><a name="p124173383205"></a><a name="p124173383205"></a><strong id="b94171338162017"><a name="b94171338162017"></a><a name="b94171338162017"></a>int topK：</strong>需要返回的最相似的结果个数<strong id="b441718389200"><a name="b441718389200"></a><a name="b441718389200"></a>。</strong></p>
</td>
</tr>
<tr id="row44171383207"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1441833813206"><a name="p1441833813206"></a><a name="p1441833813206"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p37091635161713"><a name="p37091635161713"></a><a name="p37091635161713"></a><strong id="b717718132327"><a name="b717718132327"></a><a name="b717718132327"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前“topK”个向量间的距离值。</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b185081513216"><a name="b185081513216"></a><a name="b185081513216"></a>std::vector&lt;int64_t&gt;&amp; labels</strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row3418638112011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p04181638182019"><a name="p04181638182019"></a><a name="p04181638182019"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1841853812014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p641863810207"><a name="p641863810207"></a><a name="p641863810207"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1571185610598"></a><a name="ul1571185610598"></a><ul id="ul1571185610598"><li>topK ∈ (0, 4096]</li><li><strong id="b189911330384"><a name="b189911330384"></a><a name="b189911330384"></a>n</strong>∈ (0, 10000]</li><li>queryData不能为空，且数据长度必须大于等于n * dim。</li><li>dists不能为空，且数据长度必须大于等于n * topK。</li><li>labels不能为空，且数据长度必须大于等于n * topK。</li></ul>
</td>
</tr>
</tbody>
</table>

#### SearchWithMask接口<a name="ZH-CN_TOPIC_0000002044950961"></a>

<a name="table186956182018"></a>
<table><tbody><tr id="row1252165642015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p552956182011"><a name="p552956182011"></a><a name="p552956182011"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p165218566205"><a name="p165218566205"></a><a name="p165218566205"></a>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; searchParams, const std::vector&lt;uint8_t&gt;&amp; mask);</p>
</td>
</tr>
<tr id="row1352165682013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1052456192017"><a name="p1052456192017"></a><a name="p1052456192017"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p95317562207"><a name="p95317562207"></a><a name="p95317562207"></a>实现AscendIndexGreat特征向量查询接口，根据输入的特征向量返回最相似的“topK”条特征的距离及ID，且用户可以输入一个uint8数组来掩盖特定底库ID，使该ID对应的特征向量不参与检索。</p>
</td>
</tr>
<tr id="row13531356182016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p753135672013"><a name="p753135672013"></a><a name="p753135672013"></a><strong id="b17531566204"><a name="b17531566204"></a><a name="b17531566204"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p963412085712"><a name="p963412085712"></a><a name="p963412085712"></a>searchParams结构体见<a href="#ascendindexsearchparams接口">AscendIndexSearchParams接口</a></p>
<p id="p153656142016"><a name="p153656142016"></a><a name="p153656142016"></a><strong id="b185325613204"><a name="b185325613204"></a><a name="b185325613204"></a>size_t n：</strong>查询的特征向量的条数。</p>
<p id="p1753145682011"><a name="p1753145682011"></a><a name="p1753145682011"></a><strong id="b145725499307"><a name="b145725499307"></a><a name="b145725499307"></a>std::vector&lt;float&gt;&amp; queryData：</strong>特征向量数据。</p>
<p id="p353256152013"><a name="p353256152013"></a><a name="p353256152013"></a><strong id="b853656202019"><a name="b853656202019"></a><a name="b853656202019"></a>int topK：</strong>需要返回的最相似的结果个数。</p>
<p id="p14531556102014"><a name="p14531556102014"></a><a name="p14531556102014"></a><strong id="b36191154133015"><a name="b36191154133015"></a><a name="b36191154133015"></a>const std::vector&lt;uint8_t&gt;&amp; mask</strong>：外部输入的额外的过滤mask，以bit为单位，0代表过滤该条特征；1代表选中该条特征。</p>
</td>
</tr>
<tr id="row19531156172019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p15355682013"><a name="p15355682013"></a><a name="p15355682013"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1620165312183"><a name="p1620165312183"></a><a name="p1620165312183"></a><strong id="b77168589309"><a name="b77168589309"></a><a name="b77168589309"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前“topK”个向量间的距离值。</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b1474221153119"><a name="b1474221153119"></a><a name="b1474221153119"></a>std::vector&lt;int64_t&gt;&amp;</strong> <strong id="b148237305104"><a name="b148237305104"></a><a name="b148237305104"></a>labels</strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row65385692010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1531756152010"><a name="p1531756152010"></a><a name="p1531756152010"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1453105616207"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p253155611208"><a name="p253155611208"></a><a name="p253155611208"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul104452596013"></a><a name="ul104452596013"></a><ul id="ul104452596013"><li>topK ∈ (0, 4096]</li><li>n ∈ (0, 10000]</li><li>queryData不能为空，且数据长度必须大于等于n * dim。</li><li>dists不能为空，且指向的数据长度必须大于等于n * topK。</li><li>labels不能为空，且指向的数据长度必须大于等于n * topK。</li><li>mask指向的数据总量必须大于等于n * ceil(nTotal / 8)。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetNTotal接口<a name="ZH-CN_TOPIC_0000002044829965"></a>

<a name="table971712872115"></a>
<table><tbody><tr id="row11742385218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p474298192115"><a name="p474298192115"></a><a name="p474298192115"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p47421585211"><a name="p47421585211"></a><a name="p47421585211"></a>APP_ERROR GetNTotal (uint64_t&amp; nTotal) const;</p>
</td>
</tr>
<tr id="row207427862111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1074298172118"><a name="p1074298172118"></a><a name="p1074298172118"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p117428872117"><a name="p117428872117"></a><a name="p117428872117"></a>获取AscendIndexGreat已添加进底库的特征向量数量。</p>
</td>
</tr>
<tr id="row47428812217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p19742168172110"><a name="p19742168172110"></a><a name="p19742168172110"></a><strong id="b107421685215"><a name="b107421685215"></a><a name="b107421685215"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p174218812216"><a name="p174218812216"></a><a name="p174218812216"></a>无</p>
</td>
</tr>
<tr id="row1574215810219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2742188142115"><a name="p2742188142115"></a><a name="p2742188142115"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p474213822110"><a name="p474213822110"></a><a name="p474213822110"></a><strong id="b53981719918"><a name="b53981719918"></a><a name="b53981719918"></a>uint64_t&amp; nTotal</strong>：已添加进底库的特征向量数量。</p>
</td>
</tr>
<tr id="row374210816218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p11742108182113"><a name="p11742108182113"></a><a name="p11742108182113"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1174319812119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p27431285217"><a name="p27431285217"></a><a name="p27431285217"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p157432812211"><a name="p157432812211"></a><a name="p157432812211"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### GetDim接口<a name="ZH-CN_TOPIC_0000002008751986"></a>

<a name="table113422226216"></a>
<table><tbody><tr id="row336622232116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p336632292112"><a name="p336632292112"></a><a name="p336632292112"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p836612222213"><a name="p836612222213"></a><a name="p836612222213"></a>APP_ERROR GetDim(int&amp; dim) const;</p>
</td>
</tr>
<tr id="row11366122212211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p123660225216"><a name="p123660225216"></a><a name="p123660225216"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p17366182232119"><a name="p17366182232119"></a><a name="p17366182232119"></a>获取AscendIndexGreat已添加进底库的特征向量的维度。</p>
</td>
</tr>
<tr id="row13366172242112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p9366112272118"><a name="p9366112272118"></a><a name="p9366112272118"></a><strong id="b10366142210217"><a name="b10366142210217"></a><a name="b10366142210217"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16366182210213"><a name="p16366182210213"></a><a name="p16366182210213"></a>无</p>
</td>
</tr>
<tr id="row10366152211214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p63661422162118"><a name="p63661422162118"></a><a name="p63661422162118"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18366422142117"><a name="p18366422142117"></a><a name="p18366422142117"></a><strong id="b736392718117"><a name="b736392718117"></a><a name="b736392718117"></a>int&amp; dim</strong>：已添加进底库的特征向量的维度。</p>
</td>
</tr>
<tr id="row9366182282110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12366122213212"><a name="p12366122213212"></a><a name="p12366122213212"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row14367522172117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p536720226216"><a name="p536720226216"></a><a name="p536720226216"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p5367922162119"><a name="p5367922162119"></a><a name="p5367922162119"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### Reset接口<a name="ZH-CN_TOPIC_0000002008910278"></a>

<a name="table1974793512118"></a>
<table><tbody><tr id="row5768235152116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1768183515214"><a name="p1768183515214"></a><a name="p1768183515214"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1676816355218"><a name="p1676816355218"></a><a name="p1676816355218"></a>APP_ERROR Reset();</p>
</td>
</tr>
<tr id="row1576843562115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p11768335132110"><a name="p11768335132110"></a><a name="p11768335132110"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p776983510216"><a name="p776983510216"></a><a name="p776983510216"></a>清空该Index数据保存的数据包括压缩降维后的特征向量和码本数据，同时保留用户初始化索引时输入的参数。</p>
</td>
</tr>
<tr id="row776916357212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17769235112111"><a name="p17769235112111"></a><a name="p17769235112111"></a><strong id="b7769123532119"><a name="b7769123532119"></a><a name="b7769123532119"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p187691735182114"><a name="p187691735182114"></a><a name="p187691735182114"></a>无</p>
</td>
</tr>
<tr id="row157697353212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9769193511213"><a name="p9769193511213"></a><a name="p9769193511213"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8769143514214"><a name="p8769143514214"></a><a name="p8769143514214"></a>无</p>
</td>
</tr>
<tr id="row77691535162118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1676963518218"><a name="p1676963518218"></a><a name="p1676963518218"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row4770735172115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12770123572120"><a name="p12770123572120"></a><a name="p12770123572120"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10770123572119"><a name="p10770123572119"></a><a name="p10770123572119"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### SetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044950965"></a>

<a name="table1011347192118"></a>
<table><tbody><tr id="row7231478219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1824047142117"><a name="p1824047142117"></a><a name="p1824047142117"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1124114712115"><a name="p1124114712115"></a><a name="p1124114712115"></a>APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams&amp; params);</p>
</td>
</tr>
<tr id="row324114712112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p6241447112115"><a name="p6241447112115"></a><a name="p6241447112115"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p924194712114"><a name="p924194712114"></a><a name="p924194712114"></a>设置该Index检索时的超参。</p>
</td>
</tr>
<tr id="row22484711211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p132404719216"><a name="p132404719216"></a><a name="p132404719216"></a><strong id="b1424114782113"><a name="b1424114782113"></a><a name="b1424114782113"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p112464718211"><a name="p112464718211"></a><a name="p112464718211"></a><strong id="b105404714118"><a name="b105404714118"></a><a name="b105404714118"></a>const AscendIndexHyperParams&amp; params</strong>：检索时的超参，具体请参见<a href="#ascendindexhyperparams接口">AscendIndexHyperParams</a>。</p>
</td>
</tr>
<tr id="row1824174715218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1241647102111"><a name="p1241647102111"></a><a name="p1241647102111"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16248474211"><a name="p16248474211"></a><a name="p16248474211"></a>无</p>
</td>
</tr>
<tr id="row1424647162118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p324194716213"><a name="p324194716213"></a><a name="p324194716213"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row12484710212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p72464711212"><a name="p72464711212"></a><a name="p72464711212"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4241047102110"><a name="p4241047102110"></a><a name="p4241047102110"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### GetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002400547905"></a>

<a name="table749915518225"></a>
<table><tbody><tr id="row18522851227"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p15221751229"><a name="p15221751229"></a><a name="p15221751229"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2052235122213"><a name="p2052235122213"></a><a name="p2052235122213"></a>APP_ERROR GetHyperSearchParams(AscendIndexHyperParams&amp; params) const;</p>
</td>
</tr>
<tr id="row1752219552217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1852235192213"><a name="p1852235192213"></a><a name="p1852235192213"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1752295162213"><a name="p1752295162213"></a><a name="p1752295162213"></a>获取该Index检索时的检索超参。</p>
</td>
</tr>
<tr id="row15522451223"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p45228522213"><a name="p45228522213"></a><a name="p45228522213"></a><strong id="b1052285172215"><a name="b1052285172215"></a><a name="b1052285172215"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p352213552213"><a name="p352213552213"></a><a name="p352213552213"></a>无</p>
</td>
</tr>
<tr id="row352210511229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14522155102216"><a name="p14522155102216"></a><a name="p14522155102216"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p141731750113612"><a name="p141731750113612"></a><a name="p141731750113612"></a><strong id="b17417143113299"><a name="b17417143113299"></a><a name="b17417143113299"></a>AscendIndexHyperParams&amp; params</strong>：检索时的超参，具体请参见<a href="#ascendindexhyperparams接口">AscendIndexHyperParams</a>。</p>
</td>
</tr>
<tr id="row1252219592211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p185221754223"><a name="p185221754223"></a><a name="p185221754223"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1024995915361"><a name="p1024995915361"></a><a name="p1024995915361"></a><strong id="b324975910365"><a name="b324975910365"></a><a name="b324975910365"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1652316522217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1752318572219"><a name="p1752318572219"></a><a name="p1752318572219"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1252316522219"><a name="p1252316522219"></a><a name="p1252316522219"></a>无</p>
</td>
</tr>
</tbody>
</table>

### AscendIndexMixSearchParams<a name="ZH-CN_TOPIC_0000002008910258"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000002045034929"></a>

AscendIndexMixSearchParams.h文件，提供AscendIndexGreat和AscendIndexVStar需要的结构体。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

#### AscendIndexGreatInitParams接口<a name="ZH-CN_TOPIC_0000002049404289"></a>

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

#### AscendIndexVstarInitParams接口<a name="ZH-CN_TOPIC_0000002013246410"></a>

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

#### AscendIndexVstarHyperParams接口<a name="ZH-CN_TOPIC_0000002013404694"></a>

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

#### AscendIndexHyperParams接口<a name="ZH-CN_TOPIC_0000002049325253"></a>

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

#### AscendIndexSearchParams接口<a name="ZH-CN_TOPIC_0000002044950949"></a>

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

### AscendIndexIVFFlat<a name="ZH-CN_TOPIC_0000002478095516"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000002510095475"></a>

AscendIndexIVFFlat利用IVF进行加速，是二级近似检索算法。当前仅支持IP距离。

#### AscendIndexIVFFlat接口<a name="ZH-CN_TOPIC_0000002509975505"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1857144175420"><a name="p1857144175420"></a><a name="p1857144175420"></a>AscendIndexIVFFlat(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFFlatConfig config)</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFFlat的构造函数，创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1580317419509"><a name="b1580317419509"></a><a name="b1580317419509"></a>int dims</strong>：底库检索向量的维度。</p>
<p id="p9902601825"><a name="p9902601825"></a><a name="p9902601825"></a><strong id="b19494101811220"><a name="b19494101811220"></a><a name="b19494101811220"></a>faiss::MetricType metric</strong>：距离类型，当前只支持faiss::METRIC_INNER_PRODUCT。</p>
<p id="p15757141212318"><a name="p15757141212318"></a><a name="p15757141212318"></a><strong id="b1966283819616"><a name="b1966283819616"></a><a name="b1966283819616"></a>int nlist</strong>：IVF分桶数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1191265511617"><a name="b1191265511617"></a><a name="b1191265511617"></a>AscendIndexIVFFlatConfig&nbsp;config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul79274019592"></a><a name="ul79274019592"></a><ul id="ul79274019592"><li>dims目前仅支持128。</li><li>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table663150151113"></a>
<table><tbody><tr id="row176440181111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p064509114"><a name="p064509114"></a><a name="p064509114"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14739204217152"><a name="p14739204217152"></a><a name="p14739204217152"></a>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row186417021110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p664405110"><a name="p664405110"></a><a name="p664405110"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p76470121111"><a name="p76470121111"></a><a name="p76470121111"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row964505113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2642019118"><a name="p2642019118"></a><a name="p2642019118"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b3738139972"><a name="b3738139972"></a><a name="b3738139972"></a>const AscendIndexIVFFlat&amp;</strong>：常量AscendIndexIVFFlat。</p>
</td>
</tr>
<tr id="row8641601111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p13648019116"><a name="p13648019116"></a><a name="p13648019116"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p06420013110"><a name="p06420013110"></a><a name="p06420013110"></a>无</p>
</td>
</tr>
<tr id="row1641608114"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p96418010111"><a name="p96418010111"></a><a name="p96418010111"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1264107115"><a name="p1264107115"></a><a name="p1264107115"></a>无</p>
</td>
</tr>
<tr id="row176420181110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p126412017119"><a name="p126412017119"></a><a name="p126412017119"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16647010117"><a name="p16647010117"></a><a name="p16647010117"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### \~AscendIndexIVFFlat接口<a name="ZH-CN_TOPIC_0000002477935546"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19588544116"><a name="p19588544116"></a><a name="p19588544116"></a>~AscendIndexIVFFlat()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFFlat的析构函数，销毁AscendIndexIVFFlat对象，释放资源。</p>
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

#### operate = 接口<a name="ZH-CN_TOPIC_0000002484264062"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b142971612074"><a name="b142971612074"></a><a name="b142971612074"></a>const AscendIndexIVFFlat&amp;</strong>：常量AscendIndexIVFFlat。</p>
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

#### train接口<a name="ZH-CN_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void train(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对AscendIndexIVFFlat执行训练，继承AscendIndex中的相关接口并提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b351832435710"><a name="b351832435710"></a><a name="b351832435710"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b17199113075712"><a name="b17199113075712"></a><a name="b17199113075712"></a>const float *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>训练采用k-means进行聚类，训练集比较小可能会影响查询精度。</li><li>此处<span class="parmname" id="parmname125783489316"><a name="parmname125783489316"></a><a name="parmname125783489316"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname95481642105713"><a name="parmname95481642105713"></a><a name="parmname95481642105713"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>当前只支持CPU聚类，不支持<span class="parmname" id="parmname13911501094"><a name="parmname13911501094"></a><a name="parmname13911501094"></a>“useKmeansPP”</span>参数设置为<span class="parmvalue" id="parmvalue10995953394"><a name="parmvalue10995953394"></a><a name="parmvalue10995953394"></a>“true”</span>。</li></ul>
</td>
</tr>
</tbody>
</table>
