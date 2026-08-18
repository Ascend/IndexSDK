# IndexILFlat<a name="ZH-CN_TOPIC_0000001506614925"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506414785"></a>

IndexILFlat继承自IndexIL，为纯Device侧检索方案，利用昇腾AI处理器和AI Core等资源进行各个接口的使能。程序需要在Host侧编译生成二进制文件，然后将二进制文件和相关运行时依赖部署到Device侧执行。IndexILFlat需要使用[Init](#init接口)指定对应资源的初始化，初始化完之后会申请一段完整空间用于存储底库。在使用完之后，需要调用[Finalize](#finalize接口)接口对资源进行释放。

IndexILFlat方案当前只在<term>Atlas 推理系列产品</term>上进行功能和性能的维护，底库和query向量由用户保证归一化，接口当前仅支持向量内积距离，具体使用方法请参见[IndexILFlat](#indexilflat)。（该算法运行成功依赖TIK算子的om文件，纯Device场景需要用户确保部署的是基于Index SDK交付件生成的om文件，需要确保om文件不被篡改。）

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AddFeatures接口<a name="ZH-CN_TOPIC_0000001456854852"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>向特征库插入<span class="parmname" id="parmname91461549125412"><a name="parmname91461549125412"></a><a name="parmname91461549125412"></a>“n”</span>个指定下标索引的特征向量，如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b645815911297"><a name="b645815911297"></a><a name="b645815911297"></a>int n</strong>：插入特征向量数目。</p>
<p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b118193183015"><a name="b118193183015"></a><a name="b118193183015"></a>const float16_t *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b119812563013"><a name="b119812563013"></a><a name="b119812563013"></a>const idx_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为n。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul10674191294110"></a><a name="ul10674191294110"></a><ul id="ul10674191294110"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname423619296407"><a name="varname423619296407"></a><a name="varname423619296407"></a>capacity</span></i>)之间。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname3816191310019"><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## ComputeDistance接口<a name="ZH-CN_TOPIC_0000001456535116"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19771939144811"><a name="p19771939144811"></a><a name="p19771939144811"></a>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname11281111175619"><a name="parmname11281111175619"></a><a name="parmname11281111175619"></a>“n”</span>条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0 且table为非空指针），则输出经过映射后的距离。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul167968714447"></a><a name="ul167968714447"></a><ul id="ul167968714447"><li><strong id="b1141410121444"><a name="b1141410121444"></a><a name="b1141410121444"></a>n</strong>：合理的n值应该在[0, <i><span class="varname" id="varname1835520162442"><a name="varname1835520162442"></a><a name="varname1835520162442"></a>capacity</span></i>]之间。</li><li><strong id="b429253634917"><a name="b429253634917"></a><a name="b429253634917"></a>distances</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname1598134614491"><a name="parmname1598134614491"></a><a name="parmname1598134614491"></a>“ntotalPad”</span>为(<i><span class="varname" id="varname1654664914915"><a name="varname1654664914915"></a><a name="varname1654664914915"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname145712539496"><a name="parmname145712539496"></a><a name="parmname145712539496"></a>“ntotal”</span>对16补齐的结果，每个query的有效比对距离存储在前<span class="parmname" id="parmname15322121501"><a name="parmname15322121501"></a><a name="parmname15322121501"></a>“ntotal”</span>的空间，补齐部分数据没有实际意义）。<p id="p0715202405018"><a name="p0715202405018"></a><a name="p0715202405018"></a>推荐使用<strong id="b1894113045017"><a name="b1894113045017"></a><a name="b1894113045017"></a>aclrtmalloc</strong>接口，可以申请到全量的物理内存来使用，优化处理时延。</p>
</li><li>传递<span class="parmname" id="parmname391111228612"><a name="parmname391111228612"></a><a name="parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="parmname19267121920619"><a name="parmname19267121920619"></a><a name="parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="b5371616181211"><a name="b5371616181211"></a><a name="b5371616181211"></a>distance</strong>进行映射：<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>首先将<strong id="b13840714131216"><a name="b13840714131216"></a><a name="b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="b7555131016121"><a name="b7555131016121"></a><a name="b7555131016121"></a>f1</strong>，然后用<strong id="b199806129123"><a name="b199806129123"></a><a name="b199806129123"></a>f1</strong>乘上<span class="parmname" id="parmname14917143791"><a name="parmname14917143791"></a><a name="parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="b1399121919123"><a name="b1399121919123"></a><a name="b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="parmname266193771110"><a name="parmname266193771110"></a><a name="parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="b12230192011219"><a name="b12230192011219"></a><a name="b12230192011219"></a>score</strong>，即完成映射，将<strong id="b1622952141216"><a name="b1622952141216"></a><a name="b1622952141216"></a>score</strong>存入<span class="parmname" id="parmname106381556121113"><a name="parmname106381556121113"></a><a name="parmname106381556121113"></a>“distance”</span> 。</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000001456694920"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p7384196195012"><a name="p7384196195012"></a><a name="p7384196195012"></a>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>与ComputeDistance类似，区别在于ComputeDistance计算待查询向量与所有底库向量的距离，而该接口ComputeDistanceByIdx只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1178514265435"><a name="b1178514265435"></a><a name="b1178514265435"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b79742028204313"><a name="b79742028204313"></a><a name="b79742028204313"></a>const float16_t *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname1441759144217"><a name="parmname1441759144217"></a><a name="parmname1441759144217"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a>const int *num</strong>： 给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b632503394315"><a name="b632503394315"></a><a name="b632503394315"></a>const idx_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname2711154912437"><a name="parmname2711154912437"></a><a name="parmname2711154912437"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname742124364316"><a name="parmname742124364316"></a><a name="parmname742124364316"></a>“indices”</span>长度为n * max(num)。</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname82723561324"><a name="varname82723561324"></a><a name="varname82723561324"></a>capacity</span></i>]之间。</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0， ntotal]之间。</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname343119418149"><a name="parmname343119418149"></a><a name="parmname343119418149"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## ComputeDistanceByThreshold接口<a name="ZH-CN_TOPIC_0000001506615117"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p137564208498"><a name="p137564208498"></a><a name="p137564208498"></a>APP_ERROR ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>在ComputeDistance的基础上增加了阈值筛选，只返回满足阈值条件的距离。如传递有效的映射表（tableLen &gt; 0且table为非空指针），则distances为映射后再进行阈值过滤的结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b6232619123815"><a name="b6232619123815"></a><a name="b6232619123815"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p144875610364"><a name="p144875610364"></a><a name="p144875610364"></a><strong id="b6143623203816"><a name="b6143623203816"></a><a name="b6143623203816"></a>float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p1924692795017"><a name="p1924692795017"></a><a name="p1924692795017"></a><strong id="b183957257385"><a name="b183957257385"></a><a name="b183957257385"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname18151822141710"><a name="parmname18151822141710"></a><a name="parmname18151822141710"></a>“threshold”</span>进行过滤。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b124943270387"><a name="b124943270387"></a><a name="b124943270387"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b13344132915381"><a name="b13344132915381"></a><a name="b13344132915381"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname15667378386"><a name="parmname15667378386"></a><a name="parmname15667378386"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue729933145515"><a name="parmvalue729933145515"></a><a name="parmvalue729933145515"></a>“48”</span>，即*table指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b5394194013386"><a name="b5394194013386"></a><a name="b5394194013386"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量长度为n。</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b787564210382"><a name="b787564210382"></a><a name="b787564210382"></a>idx_t *indices</strong>：满足阈值条件的底库向量下标索引，每个query符合条件的底库数量不同，当从前往后记录所有有效的index之后，按<span class="parmname" id="parmname887025614551"><a name="parmname887025614551"></a><a name="parmname887025614551"></a>“ntotalPad”</span>补齐占用的空间，<span class="parmname" id="parmname1488799145620"><a name="parmname1488799145620"></a><a name="parmname1488799145620"></a>“indices”</span>的总长度应该为n * nTotalPad（<span class="parmname" id="parmname10121121717561"><a name="parmname10121121717561"></a><a name="parmname10121121717561"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname13631434155615"><a name="varname13631434155615"></a><a name="varname13631434155615"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname0810103815562"><a name="parmname0810103815562"></a><a name="parmname0810103815562"></a>“ntotal”</span>对16补齐）。</p>
<p id="p03841120175217"><a name="p03841120175217"></a><a name="p03841120175217"></a><strong id="b17674164983818"><a name="b17674164983818"></a><a name="b17674164983818"></a>float *distances</strong>：满足阈值条件的底库向量与待查向量距离，有效值记录方式和空间size与<span class="parmname" id="parmname1581985716568"><a name="parmname1581985716568"></a><a name="parmname1581985716568"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b292575217384"><a name="b292575217384"></a><a name="b292575217384"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul192831928125717"></a><a name="ul192831928125717"></a><ul id="ul192831928125717"><li><strong id="b657843016578"><a name="b657843016578"></a><a name="b657843016578"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname1334393525713"><a name="varname1334393525713"></a><a name="varname1334393525713"></a>capacity</span></i>]之间。</li><li><strong id="b841218507570"><a name="b841218507570"></a><a name="b841218507570"></a>indices</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname127617300585"><a name="parmname127617300585"></a><a name="parmname127617300585"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname380193518587"><a name="varname380193518587"></a><a name="varname380193518587"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname1536074119588"><a name="parmname1536074119588"></a><a name="parmname1536074119588"></a>“ntotal”</span>对16补齐的结果，第<strong id="b107114541585"><a name="b107114541585"></a><a name="b107114541585"></a>i</strong>个query比对过滤后，有效底库的索引存储在<span class="parmname" id="parmname19106128135820"><a name="parmname19106128135820"></a><a name="parmname19106128135820"></a>“ntotalPad”</span>的前*(num + i) 的空间，补齐部分数据没有实际意义）。</li><li><strong id="b19683195213576"><a name="b19683195213576"></a><a name="b19683195213576"></a>distances</strong>：需要提供的空间长度为n * ntotalPad。</li><li><span class="parmname" id="parmname315020014583"><a name="parmname315020014583"></a><a name="parmname315020014583"></a>“indices”</span>和<span class="parmname" id="parmname28841122585"><a name="parmname28841122585"></a><a name="parmname28841122585"></a>“distances”</span>推荐使用<strong id="b4371184115813"><a name="b4371184115813"></a><a name="b4371184115813"></a>aclrtmalloc</strong>接口，可以申请到全量的物理内存来使用，优化处理时延。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname95571128225"><a name="parmname95571128225"></a><a name="parmname95571128225"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## Finalize接口<a name="ZH-CN_TOPIC_0000001506414845"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR Finalize() override;</p>
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

## GetFeatures接口<a name="ZH-CN_TOPIC_0000001456854992"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) override;</p>
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
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a>const idx_t *indices</strong>：需要获取的n个底库向量对应的索引值。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b043314127333"><a name="b043314127333"></a><a name="b043314127333"></a>float16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
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

## GetNTotal接口<a name="ZH-CN_TOPIC_0000001456375336"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a>int GetNTotal() const override;</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a>int ntotal</strong>：请参见功能描述。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>

## IndexILFlat接口<a name="ZH-CN_TOPIC_0000001456694872"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a>IndexILFlat();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>IndexILFlat的构造函数。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table194381755582"></a>
<table><tbody><tr id="row1438055581"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11438155155815"><a name="p11438155155815"></a><a name="p11438155155815"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1887018312271"><a name="p1887018312271"></a><a name="p1887018312271"></a>IndexILFlat(const IndexILFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row20438551584"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p74381159583"><a name="p74381159583"></a><a name="p74381159583"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p087012313276"><a name="p087012313276"></a><a name="p087012313276"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row24385511589"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p54381519581"><a name="p54381519581"></a><a name="p54381519581"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1786920310278"><a name="p1786920310278"></a><a name="p1786920310278"></a><strong id="b1129362910278"><a name="b1129362910278"></a><a name="b1129362910278"></a>const IndexILFlat&amp;：</strong>IndexILFlat对象。</p>
</td>
</tr>
<tr id="row84387585820"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1643812512585"><a name="p1643812512585"></a><a name="p1643812512585"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p154381517589"><a name="p154381517589"></a><a name="p154381517589"></a>无</p>
</td>
</tr>
<tr id="row043813535813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1443815510581"><a name="p1443815510581"></a><a name="p1443815510581"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184381457585"><a name="p184381457585"></a><a name="p184381457585"></a>无</p>
</td>
</tr>
<tr id="row2043811515580"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1643935185813"><a name="p1643935185813"></a><a name="p1643935185813"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## \~IndexILFlat接口<a name="ZH-CN_TOPIC_0000001456375172"></a>

<a name="table11904175418"></a>
<table><tbody><tr id="row49051251216"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11905115615"><a name="p11905115615"></a><a name="p11905115615"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15905125316"><a name="p15905125316"></a><a name="p15905125316"></a>virtual ~IndexILFlat();</p>
</td>
</tr>
<tr id="row139053510117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p179056510119"><a name="p179056510119"></a><a name="p179056510119"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p59051353114"><a name="p59051353114"></a><a name="p59051353114"></a>IndexILFlat的析构函数。</p>
</td>
</tr>
<tr id="row17905135915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18905456118"><a name="p18905456118"></a><a name="p18905456118"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6905656112"><a name="p6905656112"></a><a name="p6905656112"></a>无</p>
</td>
</tr>
<tr id="row199051557118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p129051050117"><a name="p129051050117"></a><a name="p129051050117"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p109051250114"><a name="p109051250114"></a><a name="p109051250114"></a>无</p>
</td>
</tr>
<tr id="row149051757115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p169052055120"><a name="p169052055120"></a><a name="p169052055120"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1690513511119"><a name="p1690513511119"></a><a name="p1690513511119"></a>无</p>
</td>
</tr>
<tr id="row29058514119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15905151318"><a name="p15905151318"></a><a name="p15905151318"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p139051951417"><a name="p139051951417"></a><a name="p139051951417"></a>无</p>
</td>
</tr>
</tbody>
</table>

## Init接口<a name="ZH-CN_TOPIC_0000001456375212"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize = -1) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>初始化特征库参数，申请底库内存资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1517311219268"><a name="b1517311219268"></a><a name="b1517311219268"></a>int dim</strong>：特征向量的维度。</p>
<p id="p1889154465814"><a name="p1889154465814"></a><a name="p1889154465814"></a><strong id="b637319417265"><a name="b637319417265"></a><a name="b637319417265"></a>AscendMetricType metricType</strong>： 特征距离类别（向量内积、欧氏距离、余弦相似度）。</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8628752620"><a name="b8628752620"></a><a name="b8628752620"></a>int capacity</strong>：底库最大容量，接口会根据<span class="parmname" id="parmname16513113011414"><a name="parmname16513113011414"></a><a name="parmname16513113011414"></a>“capacity”</span>值申请capacity * dim * sizeof(fp16) 字节内存数据。</p>
<p id="p1411722401512"><a name="p1411722401512"></a><a name="p1411722401512"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a>int64_t resourceSize</strong>：提前申请Device的缓存资源，检索接口被调用时可以直接使用这里的资源，而不必调用<strong id="b64449141510"><a name="b64449141510"></a><a name="b64449141510"></a>aclrtmalloc</strong>接口去申请内存，达到优化加速。</p>
<p id="p117241413167"><a name="p117241413167"></a><a name="p117241413167"></a>默认取值<span class="parmvalue" id="parmvalue207081128161516"><a name="parmvalue207081128161516"></a><a name="parmvalue207081128161516"></a>“-1”</span>，代表按默认size申请缓存资源（128MB），可以根据检索业务的数据量和Device上的资源使用情况来更精确地配置实际需要使用的size大小。</p>
<p id="p1703214386"><a name="p1703214386"></a><a name="p1703214386"></a>例如：query的<span class="parmname" id="parmname193097270169"><a name="parmname193097270169"></a><a name="parmname193097270169"></a>“batch”</span>为<span class="parmvalue" id="parmvalue9487237111617"><a name="parmvalue9487237111617"></a><a name="parmvalue9487237111617"></a>“64”</span>，底库总量为100万，而一个FP32数值占用4个字节，那么这里的<span class="parmname" id="parmname10173124210166"><a name="parmname10173124210166"></a><a name="parmname10173124210166"></a>“resourceSize”</span>可以设置为：64 * 1000000 * 4 = 256,000,000 Byte，注意接口内部支持申请的最大缓存资源为4GB。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1768605017262"></a><a name="ul1768605017262"></a><ul id="ul1768605017262"><li>dim ∈ {32, 64, 128, 256, 384, 512, 1024}</li><li>metricType：IndexILFlat目前只实现了向量内积距离，即只支持<span class="parmvalue" id="parmvalue194512012016"><a name="parmvalue194512012016"></a><a name="parmvalue194512012016"></a>“AscendMetricType::ASCEND_METRIC_INNER_PRODUCT”</span>。</li><li>capacity：接口允许为底库申请的内存上限设为12,288,000,000Byte，同时capacity的值域约束为(0, 12000000]。<a name="ul138816512117"></a><a name="ul138816512117"></a><ul id="ul138816512117"><li>以512维、FP16类型的底库向量为例，最大支持的<span class="parmname" id="parmname1593195143016"><a name="parmname1593195143016"></a><a name="parmname1593195143016"></a>“capacity”</span>为1200万(12288000000 / (512 * sizeof(fp_16)) )。</li><li>对于256维、FP16类型的底库向量，尽管内存约束支持更大的capacity，capacity最大也只能设为1200万。</li></ul>
</li><li>resourceSize：可以配置为-1或[134217728，4294967296]之间的值，数值的单位为Byte，相当于[128MB，4096MB]。</li></ul>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001897140809"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4759192852812"><a name="p4759192852812"></a><a name="p4759192852812"></a>IndexILFlat&amp; operator=(const IndexILFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5787143522812"><a name="p5787143522812"></a><a name="p5787143522812"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p175601140172811"><a name="p175601140172811"></a><a name="p175601140172811"></a><strong id="b9347444182810"><a name="b9347444182810"></a><a name="b9347444182810"></a>const IndexILFlat&amp;：</strong>IndexILFlat对象。</p>
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

## RemoveFeatures接口<a name="ZH-CN_TOPIC_0000001506414837"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR RemoveFeatures(int n, const idx_t *indices) override;</p>
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

## Search接口<a name="ZH-CN_TOPIC_0000001456854856"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p95721657104813"><a name="p95721657104813"></a><a name="p95721657104813"></a>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询与query向量距离最近的<span class="parmname" id="parmname66101618135916"><a name="parmname66101618135916"></a><a name="parmname66101618135916"></a>“topk”</span>个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0 且table为非空指针），则输出映射后的距离。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1346615102548"></a><a name="ul1346615102548"></a><ul id="ul1346615102548"><li><strong id="b5538111715545"><a name="b5538111715545"></a><a name="b5538111715545"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname17384527155412"><a name="varname17384527155412"></a><a name="varname17384527155412"></a>capacity</span></i>]之间。</li><li><strong id="b18681518105411"><a name="b18681518105411"></a><a name="b18681518105411"></a>topk</strong>：取值应在[0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## SearchByThreshold接口<a name="ZH-CN_TOPIC_0000001456694892"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1591144334913"><a name="p1591144334913"></a><a name="p1591144334913"></a>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
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
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b24212483396"><a name="b24212483396"></a><a name="b24212483396"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p12923104514555"><a name="p12923104514555"></a><a name="p12923104514555"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname166164371714"><a name="parmname166164371714"></a><a name="parmname166164371714"></a>“threshold”</span>进行过滤。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1578817211311"><a name="parmname1578817211311"></a><a name="parmname1578817211311"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname279351319407"><a name="parmname279351319407"></a><a name="parmname279351319407"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue376417339011"><a name="parmvalue376417339011"></a><a name="parmvalue376417339011"></a>“48”</span>，即<span class="parmname" id="parmname19896039903"><a name="parmname19896039903"></a><a name="parmname19896039903"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b662915439408"><a name="b662915439408"></a><a name="b662915439408"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b452884674019"><a name="b452884674019"></a><a name="b452884674019"></a>idx_t *indices</strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname1633953516114"><a name="parmname1633953516114"></a><a name="parmname1633953516114"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname1829610421613"><a name="parmname1829610421613"></a><a name="parmname1829610421613"></a>“indices”</span>总长度为n * topk。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname616535816"><a name="varname616535816"></a><a name="varname616535816"></a>capacity</span></i>]之间。</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a>topk</strong>：取值应在[0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname1431611816133"><a name="parmname1431611816133"></a><a name="parmname1431611816133"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## SetNTotal接口<a name="ZH-CN_TOPIC_0000001456854892"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR SetNTotal(int n) override;</p>
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
