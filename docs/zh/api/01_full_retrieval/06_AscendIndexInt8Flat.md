# AscendIndexInt8Flat<a name="ZH-CN_TOPIC_0000001506334741"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506615033"></a>

AscendIndexInt8Flat存储INT8类型特征向量并执行暴力搜索。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AscendIndexInt8Flat接口<a name="ZH-CN_TOPIC_0000001456375168"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p43030218474"><a name="p43030218474"></a><a name="p43030218474"></a>AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8Flat的构造函数，生成维度为dims的AscendIndexInt8（单个Index管理的一组向量的维度是唯一的），此时根据config中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b13765143611523"><a name="b13765143611523"></a><a name="b13765143611523"></a>int dims</strong>：AscendIndexInt8管理的一组特征向量的维度。</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1884823913524"><a name="b1884823913524"></a><a name="b1884823913524"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b713664225211"><a name="b713664225211"></a><a name="b713664225211"></a>AscendIndexInt8FlatConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul246474615523"></a><a name="ul246474615523"></a><ul id="ul246474615523"><li>dims ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table08035919302"></a>
<table><tbody><tr id="row280317933013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p080310943012"><a name="p080310943012"></a><a name="p080310943012"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a>AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</p>
</td>
</tr>
<tr id="row1880379113018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p128039983019"><a name="p128039983019"></a><a name="p128039983019"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p28031693305"><a name="p28031693305"></a><a name="p28031693305"></a>AscendIndexInt8Flat的构造函数，基于一个已有的<span class="parmname" id="parmname366313478466"><a name="parmname366313478466"></a><a name="parmname366313478466"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row168031396307"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p13803993309"><a name="p13803993309"></a><a name="p13803993309"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p198038933015"><a name="p198038933015"></a><a name="p198038933015"></a><strong id="b5607121845310"><a name="b5607121845310"></a><a name="b5607121845310"></a>const faiss::IndexScalarQuantizer *index</strong>：CPU侧Index资源。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1545613219539"><a name="b1545613219539"></a><a name="b1545613219539"></a>AscendIndexInt8FlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row1580359153014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p88038983010"><a name="p88038983010"></a><a name="p88038983010"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p138034993013"><a name="p138034993013"></a><a name="p138034993013"></a>无</p>
</td>
</tr>
<tr id="row13803119153019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p16803159103018"><a name="p16803159103018"></a><a name="p16803159103018"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p108037918306"><a name="p108037918306"></a><a name="p108037918306"></a>无</p>
</td>
</tr>
<tr id="row78038912309"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p88031919307"><a name="p88031919307"></a><a name="p88031919307"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname105161728105317"><a name="parmname105161728105317"></a><a name="parmname105161728105317"></a>“index”</span>需要为合法有效的CPU Index指针，必须为<strong id="b17970185610179"><a name="b17970185610179"></a><a name="b17970185610179"></a>AscendIndexInt8Flat</strong>执行<strong id="b470691181811"><a name="b470691181811"></a><a name="b470691181811"></a>copyTo</strong>接口生成的faiss::IndexScalarQuantizer类型指针。</p>
</td>
</tr>
</tbody>
</table>

<a name="table11312020103012"></a>
<table><tbody><tr id="row18131520123011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p51314208305"><a name="p51314208305"></a><a name="p51314208305"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p134321816154815"><a name="p134321816154815"></a><a name="p134321816154815"></a>AscendIndexInt8Flat(const faiss::IndexIDMap *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</p>
</td>
</tr>
<tr id="row14131152033015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p171311220173012"><a name="p171311220173012"></a><a name="p171311220173012"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16131202010306"><a name="p16131202010306"></a><a name="p16131202010306"></a>AscendIndexInt8Flat的构造函数，基于一个已有的<span class="parmname" id="parmname9417357194619"><a name="parmname9417357194619"></a><a name="parmname9417357194619"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row213118206301"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p613111201309"><a name="p613111201309"></a><a name="p613111201309"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1131192014300"><a name="p1131192014300"></a><a name="p1131192014300"></a><strong id="b7365124165315"><a name="b7365124165315"></a><a name="b7365124165315"></a>const faiss::IndexIDMap *index</strong>：CPU侧Index资源。</p>
<p id="p01311220103014"><a name="p01311220103014"></a><a name="p01311220103014"></a><strong id="b7835143185317"><a name="b7835143185317"></a><a name="b7835143185317"></a>AscendIndexInt8FlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row1113242019308"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14132132015303"><a name="p14132132015303"></a><a name="p14132132015303"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8132220193011"><a name="p8132220193011"></a><a name="p8132220193011"></a>无</p>
</td>
</tr>
<tr id="row8132132093017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p91321620163010"><a name="p91321620163010"></a><a name="p91321620163010"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1132720163012"><a name="p1132720163012"></a><a name="p1132720163012"></a>无</p>
</td>
</tr>
<tr id="row12132820203018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2132142003020"><a name="p2132142003020"></a><a name="p2132142003020"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p313222019308"><a name="p313222019308"></a><a name="p313222019308"></a><span class="parmname" id="parmname153575476538"><a name="parmname153575476538"></a><a name="parmname153575476538"></a>“index”</span>需要为合法有效的CPU Index指针，必须为<strong id="b13735204215184"><a name="b13735204215184"></a><a name="b13735204215184"></a>AscendIndexInt8Flat</strong>执行<strong id="b1583184418185"><a name="b1583184418185"></a><a name="b1583184418185"></a>copyTo</strong>接口生成的faiss::IndexIDMap类型指针。</p>
</td>
</tr>
</tbody>
</table>

<a name="table186285584308"></a>
<table><tbody><tr id="row11628358133010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p14628358193015"><a name="p14628358193015"></a><a name="p14628358193015"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8Flat(const AscendIndexInt8Flat&amp;) = delete;</p>
</td>
</tr>
<tr id="row1362814589304"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p5628195813011"><a name="p5628195813011"></a><a name="p5628195813011"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p96281058103020"><a name="p96281058103020"></a><a name="p96281058103020"></a>声明此Index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row56281058123019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1962825833011"><a name="p1962825833011"></a><a name="p1962825833011"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b11456125611539"><a name="b11456125611539"></a><a name="b11456125611539"></a>const AscendIndexInt8Flat&amp;</strong>：常量AscendIndexInt8Flat。</p>
</td>
</tr>
<tr id="row16281558103012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p36281758163011"><a name="p36281758163011"></a><a name="p36281758163011"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p86282058183018"><a name="p86282058183018"></a><a name="p86282058183018"></a>无</p>
</td>
</tr>
<tr id="row6628175820307"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p662825863011"><a name="p662825863011"></a><a name="p662825863011"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p6628658183012"><a name="p6628658183012"></a><a name="p6628658183012"></a>无</p>
</td>
</tr>
<tr id="row12628958103010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p6628195853016"><a name="p6628195853016"></a><a name="p6628195853016"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p962855810301"><a name="p962855810301"></a><a name="p962855810301"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table206471151315"></a>
<table><tbody><tr id="row564841517316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p464801510315"><a name="p464801510315"></a><a name="p464801510315"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexInt8Flat();</p>
</td>
</tr>
<tr id="row156481015103116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1364813150319"><a name="p1364813150319"></a><a name="p1364813150319"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1064810155319"><a name="p1064810155319"></a><a name="p1064810155319"></a>AscendIndexInt8Flat的析构函数，销毁AscendIndexInt8Flat对象，释放资源。</p>
</td>
</tr>
<tr id="row1564851515314"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1648181512311"><a name="p1648181512311"></a><a name="p1648181512311"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row156481215103116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1264891513116"><a name="p1264891513116"></a><a name="p1264891513116"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p7648121583112"><a name="p7648121583112"></a><a name="p7648121583112"></a>无</p>
</td>
</tr>
<tr id="row564812154316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p7648111593118"><a name="p7648111593118"></a><a name="p7648111593118"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p064841512313"><a name="p064841512313"></a><a name="p064841512313"></a>无</p>
</td>
</tr>
<tr id="row3648915103115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p86481115153115"><a name="p86481115153115"></a><a name="p86481115153115"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1364861518319"><a name="p1364861518319"></a><a name="p1364861518319"></a>无</p>
</td>
</tr>
</tbody>
</table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456375340"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyFrom(const faiss::IndexIDMap* index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8Flat基于一个已有的<span class="parmname" id="parmname18533121195111"><a name="parmname18533121195111"></a><a name="parmname18533121195111"></a>“index”</span>拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b7731624145618"><a name="b7731624145618"></a><a name="b7731624145618"></a>const faiss::IndexIDMap *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1742162695618"><a name="parmname1742162695618"></a><a name="parmname1742162695618"></a>“index”</span>需要为合法有效的IndexIDMap指针，该Index的成员索引维度d参数取值范围为{64, 128, 256, 384, 512, 768, 1024}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table862731073217"></a>
<table><tbody><tr id="row562716101326"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p6627201073216"><a name="p6627201073216"></a><a name="p6627201073216"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6908132519515"><a name="p6908132519515"></a><a name="p6908132519515"></a>void copyFrom(const faiss::IndexScalarQuantizer* index);</p>
</td>
</tr>
<tr id="row1562701093213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1962719100320"><a name="p1962719100320"></a><a name="p1962719100320"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16271610153210"><a name="p16271610153210"></a><a name="p16271610153210"></a>AscendIndexInt8Flat基于一个已有的<span class="parmname" id="parmname10631110135119"><a name="parmname10631110135119"></a><a name="parmname10631110135119"></a>“index”</span>拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row6627181014329"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p20627610203213"><a name="p20627610203213"></a><a name="p20627610203213"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p126277101327"><a name="p126277101327"></a><a name="p126277101327"></a><strong id="b6870101316567"><a name="b6870101316567"></a><a name="b6870101316567"></a>const faiss::IndexScalarQuantizer* index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="row362771017325"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p262713107326"><a name="p262713107326"></a><a name="p262713107326"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1762761014326"><a name="p1762761014326"></a><a name="p1762761014326"></a>无</p>
</td>
</tr>
<tr id="row18627710123214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p15627310183210"><a name="p15627310183210"></a><a name="p15627310183210"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1362715109324"><a name="p1362715109324"></a><a name="p1362715109324"></a>无</p>
</td>
</tr>
<tr id="row18627510133211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p26271100323"><a name="p26271100323"></a><a name="p26271100323"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1562781003214"><a name="p1562781003214"></a><a name="p1562781003214"></a><span class="parmname" id="parmname55631616145615"><a name="parmname55631616145615"></a><a name="parmname55631616145615"></a>“index”</span>需要为合法有效的CPU Index指针，Index的维度d参数取值范围为{64, 128, 256, 384, 512, 768, 1024}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001506334805"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexScalarQuantizer* index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>将AscendIndexInt8Flat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b419043435615"><a name="b419043435615"></a><a name="b419043435615"></a>faiss::IndexScalarQuantizer* index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname947410378567"><a name="parmname947410378567"></a><a name="parmname947410378567"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1981952413329"></a>
<table><tbody><tr id="row6819122423218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9819112423211"><a name="p9819112423211"></a><a name="p9819112423211"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1981912463211"><a name="p1981912463211"></a><a name="p1981912463211"></a>void copyTo(faiss::IndexIDMap* index) const;</p>
</td>
</tr>
<tr id="row128191424163217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p16819224173215"><a name="p16819224173215"></a><a name="p16819224173215"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p28192024163218"><a name="p28192024163218"></a><a name="p28192024163218"></a>将AscendIndexInt8Flat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row1281910243329"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p188196241328"><a name="p188196241328"></a><a name="p188196241328"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p281916241321"><a name="p281916241321"></a><a name="p281916241321"></a><strong id="b2771344195619"><a name="b2771344195619"></a><a name="b2771344195619"></a>faiss::IndexIDMap *index</strong>：CPU侧index资源。</p>
</td>
</tr>
<tr id="row2819182413219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p281972453215"><a name="p281972453215"></a><a name="p281972453215"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6819192473220"><a name="p6819192473220"></a><a name="p6819192473220"></a>无</p>
</td>
</tr>
<tr id="row14819152483212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p4819024113219"><a name="p4819024113219"></a><a name="p4819024113219"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p178191624133216"><a name="p178191624133216"></a><a name="p178191624133216"></a>无</p>
</td>
</tr>
<tr id="row381919240326"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p78191224173213"><a name="p78191224173213"></a><a name="p78191224173213"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4819132473214"><a name="p4819132473214"></a><a name="p4819132473214"></a><span class="parmname" id="parmname71714475560"><a name="parmname71714475560"></a><a name="parmname71714475560"></a>“index”</span>需要为合法有效的IndexIDMap指针，index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

## getBase接口<a name="ZH-CN_TOPIC_0000001506334753"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getBase(int deviceId, std::vector&lt;int8_t&gt; &amp;xb) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexInt8Flat在特定<span class="parmname" id="parmname1195143920472"><a name="parmname1195143920472"></a><a name="parmname1195143920472"></a>“deviceId”</span>上管理的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b196131849175412"><a name="b196131849175412"></a><a name="b196131849175412"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b07321515549"><a name="b07321515549"></a><a name="b07321515549"></a>std::vector&lt;int8_t&gt; &amp;xb</strong>：AscendIndexInt8Flat在<span class="parmname" id="parmname20870161012482"><a name="parmname20870161012482"></a><a name="parmname20870161012482"></a>“deviceId”</span>上存储的底库特征向量。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname112791054165415"><a name="parmname112791054165415"></a><a name="parmname112791054165415"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## getBaseSize接口<a name="ZH-CN_TOPIC_0000001506414709"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>size_t getBaseSize(int deviceId) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexInt8Flat在特定<span class="parmname" id="parmname141001228114714"><a name="parmname141001228114714"></a><a name="parmname141001228114714"></a>“deviceId”</span>上管理的特征向量数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b153555291543"><a name="b153555291543"></a><a name="b153555291543"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>在特定<span class="parmname" id="parmname1298021824811"><a name="parmname1298021824811"></a><a name="parmname1298021824811"></a>“deviceId”</span>上的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname20741633175413"><a name="parmname20741633175413"></a><a name="parmname20741633175413"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## getIdxMap接口<a name="ZH-CN_TOPIC_0000001506495853"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexInt8Flat在特定<span class="parmname" id="parmname7542659164914"><a name="parmname7542659164914"></a><a name="parmname7542659164914"></a>“deviceId”</span>上管理的特征向量ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b1775544965512"><a name="b1775544965512"></a><a name="b1775544965512"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b632965711198"><a name="b632965711198"></a><a name="b632965711198"></a>std::vector&lt;idx_t&gt; &amp;idxMap</strong>：AscendIndexInt8Flat在<span class="parmname" id="parmname62818562490"><a name="parmname62818562490"></a><a name="parmname62818562490"></a>“deviceId”</span>上存储的底库特征向量ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname498819558551"><a name="parmname498819558551"></a><a name="parmname498819558551"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506414909"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8Flat&amp; operator=(const AscendIndexInt8Flat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b2040411013542"><a name="b2040411013542"></a><a name="b2040411013542"></a>const AscendIndexInt8Flat&amp;</strong>：常量AscendIndexInt8Flat。</p>
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

## reset接口<a name="ZH-CN_TOPIC_0000001506495889"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void reset();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>清空该AscendIndexInt8Flat的底库向量。</p>
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

## search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001456694912"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void search_with_masks(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndexInt8特征向量查询接口，根据输入的特征向量以及<span class="parmname" id="parmname69861138105118"><a name="parmname69861138105118"></a><a name="parmname69861138105118"></a>“mask”</span>掩码返回最相似的<span class="parmname" id="parmname19493733155119"><a name="parmname19493733155119"></a><a name="parmname19493733155119"></a>“k”</span>条特征的距离及ID。mask为<strong id="b14721249104814"><a name="b14721249104814"></a><a name="b14721249104814"></a>0</strong>、<strong id="b596515503483"><a name="b596515503483"></a><a name="b596515503483"></a>1</strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong id="b295210616496"><a name="b295210616496"></a><a name="b295210616496"></a>1</strong>参与，<strong id="b93801987491"><a name="b93801987491"></a><a name="b93801987491"></a>0</strong>不参与。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><strong id="b186310558563"><a name="b186310558563"></a><a name="b186310558563"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><strong id="b531925911561"><a name="b531925911561"></a><a name="b531925911561"></a>const int8_t* x</strong>：特征向量数据。</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><strong id="b7970109576"><a name="b7970109576"></a><a name="b7970109576"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p1838795616530"><a name="p1838795616530"></a><a name="p1838795616530"></a><strong id="b5798929577"><a name="b5798929577"></a><a name="b5798929577"></a>const void* mask</strong>：底库的过滤掩码。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b113310163576"><a name="b113310163576"></a><a name="b113310163576"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname64605085110"><a name="parmname64605085110"></a><a name="parmname64605085110"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b179807179572"><a name="b179807179572"></a><a name="b179807179572"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname772411523519"><a name="parmname772411523519"></a><a name="parmname772411523519"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul197738222579"></a><a name="ul197738222579"></a><ul id="ul197738222579"><li><span class="parmname" id="parmname95757582519"><a name="parmname95757582519"></a><a name="parmname95757582519"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li><span class="parmname" id="parmname382711402810"><a name="parmname382711402810"></a><a name="parmname382711402810"></a>“k”</span>通常不允许超过4096。</li><li>指针<span class="parmname" id="parmname17661432105710"><a name="parmname17661432105710"></a><a name="parmname17661432105710"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>指针<span class="parmname" id="parmname167723415714"><a name="parmname167723415714"></a><a name="parmname167723415714"></a>“distances”</span>/<span class="parmname" id="parmname193331836115710"><a name="parmname193331836115710"></a><a name="parmname193331836115710"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b5379163513393"><a name="b5379163513393"></a><a name="b5379163513393"></a>k</strong> * <strong id="b19874738123915"><a name="b19874738123915"></a><a name="b19874738123915"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>指针<span class="parmname" id="parmname187912297571"><a name="parmname187912297571"></a><a name="parmname187912297571"></a>“mask”</span>需要为非空指针，需保证传入的掩码长度为⌈ntotal / 8⌉ * n（<span class="parmname" id="parmname8177115424411"><a name="parmname8177115424411"></a><a name="parmname8177115424411"></a>“ntotal”</span>为底库向量的条数）。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>

## setPageSize接口<a name="ZH-CN_TOPIC_0000002007453769"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p20912871594"><a name="p20912871594"></a><a name="p20912871594"></a>void setPageSize(uint16_t pageBlockNum);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p393922219912"><a name="p393922219912"></a><a name="p393922219912"></a>设置该AscendIndexInt8Flat在search时一次性连续计算底库的block数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1836319401096"><a name="p1836319401096"></a><a name="p1836319401096"></a><strong id="b16962165461012"><a name="b16962165461012"></a><a name="b16962165461012"></a>uint16_t pageBlockNum</strong>：一次性连续计算底库的block数量。不设置时，默认一次性连续计算16个block。一个block存储向量的大小由AscendIndexInt8FlatConfig中的blockSize决定。该值越大，search时占用的内存越大。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul250991973317"></a><a name="ul250991973317"></a><ul id="ul250991973317"><li><span class="parmname" id="parmname159984545424"><a name="parmname159984545424"></a><a name="parmname159984545424"></a>“pageBlockNum”</span>的取值范围：0 &lt; pageBlockNum ≤ 144</li><li>该接口主要用于大底库场景，search接口性能调优使用。该值越大，占用AscendIndexInt8FlatConfig中配置的resourceSize预置内存越大。建议申请足够大的预置内存，再利用该接口进行参数调优。</li></ul>
</td>
</tr>
</tbody>
</table>
