# AscendIndexSQ<a name="ZH-CN_TOPIC_0000001506614969"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456695120"></a>

AscendIndexSQ对输入向量执行Scalar Quantization。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AscendIndexSQ接口<a name="ZH-CN_TOPIC_0000001506614933"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a>AscendIndexSQ(const faiss::IndexScalarQuantizer* index, AscendIndexSQConfig config = AscendIndexSQConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexSQ的构造函数，基于一个已有的<span class="parmname" id="parmname8477263296"><a name="parmname8477263296"></a><a name="parmname8477263296"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b6104639815"><a name="b6104639815"></a><a name="b6104639815"></a>const faiss::IndexScalarQuantizer* index</strong>：CPU侧的Index资源。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1066616367111"><a name="b1066616367111"></a><a name="b1066616367111"></a>AscendIndexSQConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1938733313207"><a name="parmname1938733313207"></a><a name="parmname1938733313207"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}，sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

<a name="table207325212487"></a>
<table><tbody><tr id="row57316521481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1731752204815"><a name="p1731752204815"></a><a name="p1731752204815"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a>AscendIndexSQ(const faiss::IndexIDMap* index, AscendIndexSQConfig config = AscendIndexSQConfig());</p>
</td>
</tr>
<tr id="row1573165204811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p147395220488"><a name="p147395220488"></a><a name="p147395220488"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10738528483"><a name="p10738528483"></a><a name="p10738528483"></a>AscendIndexSQ的构造函数，基于一个已有的<span class="parmname" id="parmname853416312298"><a name="parmname853416312298"></a><a name="parmname853416312298"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row3731652104814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p197495254810"><a name="p197495254810"></a><a name="p197495254810"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b11364627323"><a name="b11364627323"></a><a name="b11364627323"></a>const faiss::IndexIDMap* index</strong>：CPU侧index资源。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b75252919212"><a name="b75252919212"></a><a name="b75252919212"></a>AscendIndexSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row37465224818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p147415211489"><a name="p147415211489"></a><a name="p147415211489"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8741452194814"><a name="p8741452194814"></a><a name="p8741452194814"></a>无</p>
</td>
</tr>
<tr id="row167475217487"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p474185234815"><a name="p474185234815"></a><a name="p474185234815"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p197455214820"><a name="p197455214820"></a><a name="p197455214820"></a>无</p>
</td>
</tr>
<tr id="row97455219484"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p974125219485"><a name="p974125219485"></a><a name="p974125219485"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p18741526488"><a name="p18741526488"></a><a name="p18741526488"></a><span class="parmname" id="parmname14521836192011"><a name="parmname14521836192011"></a><a name="parmname14521836192011"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的成员索引的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n ＜ 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}， sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1132217014918"></a>
<table><tbody><tr id="row1132250114917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1432220012499"><a name="p1432220012499"></a><a name="p1432220012499"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11207257504"><a name="p11207257504"></a><a name="p11207257504"></a>AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexSQConfig config = AscendIndexSQConfig());</p>
</td>
</tr>
<tr id="row1232215064915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p9322140114910"><a name="p9322140114910"></a><a name="p9322140114910"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p432219014915"><a name="p432219014915"></a><a name="p432219014915"></a>AscendIndexSQ的构造函数，生成维度为dims的AscendIndex（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="parmname1128518915434"><a name="parmname1128518915434"></a><a name="parmname1128518915434"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row23229044916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p732210174911"><a name="p732210174911"></a><a name="p732210174911"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16322607495"><a name="p16322607495"></a><a name="p16322607495"></a><strong id="b63116565211"><a name="b63116565211"></a><a name="b63116565211"></a>int dims</strong>：AscendIndexSQ管理的一组特征向量的维度。</p>
<p id="p995710373711"><a name="p995710373711"></a><a name="p995710373711"></a><strong id="b155310333495"><a name="b155310333495"></a><a name="b155310333495"></a>faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit</strong>，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
<p id="p163221204497"><a name="p163221204497"></a><a name="p163221204497"></a><strong id="b20474132614312"><a name="b20474132614312"></a><a name="b20474132614312"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p1132217074917"><a name="p1132217074917"></a><a name="p1132217074917"></a><strong id="b1687210287316"><a name="b1687210287316"></a><a name="b1687210287316"></a>AscendIndexSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row163222012498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1322110154919"><a name="p1322110154919"></a><a name="p1322110154919"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p73221102490"><a name="p73221102490"></a><a name="p73221102490"></a>无</p>
</td>
</tr>
<tr id="row6322190184913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1032260194919"><a name="p1032260194919"></a><a name="p1032260194919"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8322204495"><a name="p8322204495"></a><a name="p8322204495"></a>无</p>
</td>
</tr>
<tr id="row10322120124920"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p173221017492"><a name="p173221017492"></a><a name="p173221017492"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul193685551031"></a><a name="ul193685551031"></a><ul id="ul193685551031"><li>dims ∈ {64, 128, 256, 384, 512, 768}。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table16655810104919"></a>
<table><tbody><tr id="row19655810194912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16655710184912"><a name="p16655710184912"></a><a name="p16655710184912"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p8445440165114"><a name="p8445440165114"></a><a name="p8445440165114"></a>AscendIndexSQ(const AscendIndexSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row665561014492"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p10655161013492"><a name="p10655161013492"></a><a name="p10655161013492"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p365541024916"><a name="p365541024916"></a><a name="p365541024916"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row4655110114913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1365501024920"><a name="p1365501024920"></a><a name="p1365501024920"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b16480691410"><a name="b16480691410"></a><a name="b16480691410"></a>const AscendIndexSQ&amp;</strong>：AscendIndexSQ对象。</p>
</td>
</tr>
<tr id="row13655121044912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p15655111064920"><a name="p15655111064920"></a><a name="p15655111064920"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p146551210104911"><a name="p146551210104911"></a><a name="p146551210104911"></a>无</p>
</td>
</tr>
<tr id="row116554109498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1265512105496"><a name="p1265512105496"></a><a name="p1265512105496"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p46569101499"><a name="p46569101499"></a><a name="p46569101499"></a>无</p>
</td>
</tr>
<tr id="row16568103491"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p8656810104914"><a name="p8656810104914"></a><a name="p8656810104914"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p19656210134916"><a name="p19656210134916"></a><a name="p19656210134916"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table17704194534915"></a>
<table><tbody><tr id="row147041745174918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p170414524913"><a name="p170414524913"></a><a name="p170414524913"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexSQ();</p>
</td>
</tr>
<tr id="row370416455499"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p77042459498"><a name="p77042459498"></a><a name="p77042459498"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p97045456493"><a name="p97045456493"></a><a name="p97045456493"></a>AscendIndexSQ的析构函数，销毁AscendIndexSQ对象，释放资源。</p>
</td>
</tr>
<tr id="row470419456497"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p77041745124910"><a name="p77041745124910"></a><a name="p77041745124910"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row57042456497"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p470444514493"><a name="p470444514493"></a><a name="p470444514493"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p57041745194915"><a name="p57041745194915"></a><a name="p57041745194915"></a>无</p>
</td>
</tr>
<tr id="row4704845104910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p19704134513498"><a name="p19704134513498"></a><a name="p19704134513498"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p770454516493"><a name="p770454516493"></a><a name="p770454516493"></a>无</p>
</td>
</tr>
<tr id="row1870454504914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p77046459495"><a name="p77046459495"></a><a name="p77046459495"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p5704184513492"><a name="p5704184513492"></a><a name="p5704184513492"></a>无</p>
</td>
</tr>
</tbody>
</table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001506615037"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyFrom(const faiss::IndexScalarQuantizer* index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexSQ基于一个已有的<span class="parmname" id="parmname72491525329"><a name="parmname72491525329"></a><a name="parmname72491525329"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexSQ底库，并保持原有的AscendIndexSQ的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b113197161055"><a name="b113197161055"></a><a name="b113197161055"></a>const faiss::IndexScalarQuantizer* index</strong>：CPU侧index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1170518371159"><a name="parmname1170518371159"></a><a name="parmname1170518371159"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}，sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

<a name="table853716365015"></a>
<table><tbody><tr id="row1253763155012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p65375319502"><a name="p65375319502"></a><a name="p65375319502"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15371437503"><a name="p15371437503"></a><a name="p15371437503"></a>void copyFrom(const faiss::IndexIDMap* index);</p>
</td>
</tr>
<tr id="row95371733508"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p11537735509"><a name="p11537735509"></a><a name="p11537735509"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p153715311508"><a name="p153715311508"></a><a name="p153715311508"></a>AscendIndexSQ基于一个已有的<span class="parmname" id="parmname1570211587325"><a name="parmname1570211587325"></a><a name="parmname1570211587325"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexSQ底库，并保持原有的AscendIndexSQ的Device侧资源配置。</p>
</td>
</tr>
<tr id="row155371130507"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1153720316503"><a name="p1153720316503"></a><a name="p1153720316503"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1553715335013"><a name="p1553715335013"></a><a name="p1553715335013"></a><strong id="b1898128253"><a name="b1898128253"></a><a name="b1898128253"></a>const faiss::IndexIDMap *index</strong>：CPU侧index资源。</p>
</td>
</tr>
<tr id="row1253716318502"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p115377325010"><a name="p115377325010"></a><a name="p115377325010"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p153712355015"><a name="p153712355015"></a><a name="p153712355015"></a>无</p>
</td>
</tr>
<tr id="row9537203125019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p553711365019"><a name="p553711365019"></a><a name="p553711365019"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p553743135020"><a name="p553743135020"></a><a name="p553743135020"></a>无</p>
</td>
</tr>
<tr id="row55373320504"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1537143115010"><a name="p1537143115010"></a><a name="p1537143115010"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p12537193155012"><a name="p12537193155012"></a><a name="p12537193155012"></a><span class="parmname" id="parmname1290641255"><a name="parmname1290641255"></a><a name="parmname1290641255"></a>“index”</span>需要为合法有效的IndexIDMap指针，index的成员索引的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}，sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001456695084"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1019716501395"><a name="p1019716501395"></a><a name="p1019716501395"></a>void copyTo(faiss::IndexScalarQuantizer* index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>将AscendIndexSQ的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b217675214518"><a name="b217675214518"></a><a name="b217675214518"></a>faiss::IndexScalarQuantizer* index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname74101355456"><a name="parmname74101355456"></a><a name="parmname74101355456"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

<a name="table817201512500"></a>
<table><tbody><tr id="row1517171595016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9171015145014"><a name="p9171015145014"></a><a name="p9171015145014"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexIDMap* index) const;</p>
</td>
</tr>
<tr id="row5171115145019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3177155503"><a name="p3177155503"></a><a name="p3177155503"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p51761519504"><a name="p51761519504"></a><a name="p51761519504"></a>将AscendIndexSQ的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row101711535017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p717151512506"><a name="p717151512506"></a><a name="p717151512506"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12170155508"><a name="p12170155508"></a><a name="p12170155508"></a><strong id="b18757155564"><a name="b18757155564"></a><a name="b18757155564"></a>faiss::IndexIDMap *index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="row61781514507"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p417181516501"><a name="p417181516501"></a><a name="p417181516501"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1617181515015"><a name="p1617181515015"></a><a name="p1617181515015"></a>无</p>
</td>
</tr>
<tr id="row917171512503"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p11172156506"><a name="p11172156506"></a><a name="p11172156506"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p317215145014"><a name="p317215145014"></a><a name="p317215145014"></a>无</p>
</td>
</tr>
<tr id="row6179153503"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p20171415125010"><a name="p20171415125010"></a><a name="p20171415125010"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p191791513507"><a name="p191791513507"></a><a name="p191791513507"></a><span class="parmname" id="parmname169852091361"><a name="parmname169852091361"></a><a name="parmname169852091361"></a>“index”</span>需要为合法有效的IndexIDMap指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

## getBase接口<a name="ZH-CN_TOPIC_0000001456694928"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getBase(int deviceId, char* xb) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexSQ在特定<span class="parmname" id="parmname1731761464915"><a name="parmname1731761464915"></a><a name="parmname1731761464915"></a>“deviceId”</span>上管理的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b0928171820610"><a name="b0928171820610"></a><a name="b0928171820610"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b1959712112614"><a name="b1959712112614"></a><a name="b1959712112614"></a>char* xb</strong>：AscendIndexSQ在<span class="parmname" id="parmname132844390614"><a name="parmname132844390614"></a><a name="parmname132844390614"></a>“deviceId”</span>上存储的底库特征向量。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul126112716618"></a><a name="ul126112716618"></a><ul id="ul126112716618"><li><span class="parmname" id="parmname12140629560"><a name="parmname12140629560"></a><a name="parmname12140629560"></a>“deviceId”</span>需要为合法的设备ID。</li><li><span class="parmname" id="parmname14391592819"><a name="parmname14391592819"></a><a name="parmname14391592819"></a>“xb”</span>需要为非空指针，且长度应该为dims * BaseSize * sizeof(uint8_t)字节，否则可能出现越界读写错误并引起程序崩溃，其中BaseSize为getBaseSize函数的返回值。</li></ul>
</td>
</tr>
</tbody>
</table>

## getBaseSize接口<a name="ZH-CN_TOPIC_0000001456854788"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>size_t getBaseSize(int deviceId) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexSQ在特定<span class="parmname" id="parmname17430117154914"><a name="parmname17430117154914"></a><a name="parmname17430117154914"></a>“deviceId”</span>上管理的特征向量数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b10691175516203"><a name="b10691175516203"></a><a name="b10691175516203"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>在特定<span class="parmname" id="parmname9112805213"><a name="parmname9112805213"></a><a name="parmname9112805213"></a>“deviceId”</span>上的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname57601417219"><a name="parmname57601417219"></a><a name="parmname57601417219"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## getIdxMap接口<a name="ZH-CN_TOPIC_0000001456375152"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt;&amp; idxMap) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexSQ在特定<span class="parmname" id="parmname12367174413480"><a name="parmname12367174413480"></a><a name="parmname12367174413480"></a>“deviceId”</span>上管理的特征向量ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b156818551961"><a name="b156818551961"></a><a name="b156818551961"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b69151357768"><a name="b69151357768"></a><a name="b69151357768"></a>std::vector&lt;idx_t&gt; &amp;idxMap</strong>：AscendIndexSQ在<span class="parmname" id="parmname101352016714"><a name="parmname101352016714"></a><a name="parmname101352016714"></a>“deviceId”</span>上存储的底库特征向量ID 。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname47241030712"><a name="parmname47241030712"></a><a name="parmname47241030712"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001456375300"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7779180105218"><a name="p7779180105218"></a><a name="p7779180105218"></a>AscendIndexSQ&amp; operator=(const AscendIndexSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b16267637049"><a name="b16267637049"></a><a name="b16267637049"></a>const AscendIndexSQ&amp;</strong>：AscendIndexSQ对象。</p>
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

## search\_with\_filter接口<a name="ZH-CN_TOPIC_0000001810589742"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p676092910161"><a name="p676092910161"></a><a name="p676092910161"></a>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10290157145418"><a name="p10290157145418"></a><a name="p10290157145418"></a>AscendIndexSQ的特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname1390811816443"><a name="parmname1390811816443"></a><a name="parmname1390811816443"></a>“k”</span>条特征的ID。提供基于CID过滤的功能，“filters”为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b1976572871110"><a name="b1976572871110"></a><a name="b1976572871110"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b12370123112117"><a name="b12370123112117"></a><a name="b12370123112117"></a>const float *x</strong>：特征向量数据。</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b3869633101113"><a name="b3869633101113"></a><a name="b3869633101113"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p13978130112613"><a name="p13978130112613"></a><a name="p13978130112613"></a><strong id="b157981335181116"><a name="b157981335181116"></a><a name="b157981335181116"></a>const void *filters</strong>：过滤条件。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b7967538161117"><a name="b7967538161117"></a><a name="b7967538161117"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname20628131394417"><a name="parmname20628131394417"></a><a name="parmname20628131394417"></a>“k”</span>个向量间的距离值。</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b34371240191113"><a name="b34371240191113"></a><a name="b34371240191113"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname14669121564412"><a name="parmname14669121564412"></a><a name="parmname14669121564412"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul6584134771119"></a><a name="ul6584134771119"></a><ul id="ul6584134771119"><li>此处<span class="parmname" id="parmname1251142183215"><a name="parmname1251142183215"></a><a name="parmname1251142183215"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处<span class="parmname" id="parmname166023416327"><a name="parmname166023416327"></a><a name="parmname166023416327"></a>“k”</span>通常不允许超过4096。</li><li>此处指针<span class="parmname" id="parmname1281764124"><a name="parmname1281764124"></a><a name="parmname1281764124"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname10844578120"><a name="parmname10844578120"></a><a name="parmname10844578120"></a>“distances”</span>/<span class="parmname" id="parmname1668917981220"><a name="parmname1668917981220"></a><a name="parmname1668917981220"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b1939083417308"><a name="b1939083417308"></a><a name="b1939083417308"></a>k</strong> * <strong id="b8896138133019"><a name="b8896138133019"></a><a name="b8896138133019"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname18187812201214"><a name="parmname18187812201214"></a><a name="parmname18187812201214"></a>“filters”</span>需要为非空指针，且长度为n * 6的uint32_t的数组，否则可能出现越界读取的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001456694932"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11977033135012"><a name="p11977033135012"></a><a name="p11977033135012"></a>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7584153011582"><a name="p7584153011582"></a><a name="p7584153011582"></a>AscendIndexSQ的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为<strong id="b177781744812"><a name="b177781744812"></a><a name="b177781744812"></a>0</strong>、<strong id="b61111618184814"><a name="b61111618184814"></a><a name="b61111618184814"></a>1</strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong id="b18384163024810"><a name="b18384163024810"></a><a name="b18384163024810"></a>1</strong>参与，<strong id="b2070119316483"><a name="b2070119316483"></a><a name="b2070119316483"></a>0</strong>不参与。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b8509184711813"><a name="b8509184711813"></a><a name="b8509184711813"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b2086114494814"><a name="b2086114494814"></a><a name="b2086114494814"></a>const float *x</strong>：特征向量数据。</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b2484155112813"><a name="b2484155112813"></a><a name="b2484155112813"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p841235065815"><a name="p841235065815"></a><a name="p841235065815"></a><strong id="b77116531187"><a name="b77116531187"></a><a name="b77116531187"></a>const void *mask：</strong>特征底库掩码。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b676467798"><a name="b676467798"></a><a name="b676467798"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname1152618119407"><a name="parmname1152618119407"></a><a name="parmname1152618119407"></a>“k”</span>个向量间的距离值。</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b5824159193"><a name="b5824159193"></a><a name="b5824159193"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname145221464014"><a name="parmname145221464014"></a><a name="parmname145221464014"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul184581435495"></a><a name="ul184581435495"></a><ul id="ul184581435495"><li>此处<span class="parmname" id="parmname11945191043317"><a name="parmname11945191043317"></a><a name="parmname11945191043317"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处<span class="parmname" id="parmname118167143338"><a name="parmname118167143338"></a><a name="parmname118167143338"></a>“k”</span>通常不允许超过4096。</li><li>此处指针<span class="parmname" id="parmname2847104218106"><a name="parmname2847104218106"></a><a name="parmname2847104218106"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname142691946111020"><a name="parmname142691946111020"></a><a name="parmname142691946111020"></a>“distances”</span>/<span class="parmname" id="parmname2586184851015"><a name="parmname2586184851015"></a><a name="parmname2586184851015"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b101511878307"><a name="b101511878307"></a><a name="b101511878307"></a>k</strong> * <strong id="b717661318306"><a name="b717661318306"></a><a name="b717661318306"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname199479521105"><a name="parmname199479521105"></a><a name="parmname199479521105"></a>“mask”</span>需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>

## train接口<a name="ZH-CN_TOPIC_0000001506414905"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void train(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对AscendIndexSQ执行训练量化器，继承AscendFaiss中的接口，并提供具体的实现。<strong id="b103001027278"><a name="b103001027278"></a><a name="b103001027278"></a>注意，执行add之前必须对Index进行train。</strong></p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b986713077"><a name="b986713077"></a><a name="b986713077"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b11961715876"><a name="b11961715876"></a><a name="b11961715876"></a>const float *x</strong>：特征向量数据。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1796193986"></a><a name="ul1796193986"></a><ul id="ul1796193986"><li>此处<span class="parmname" id="parmname4258436173317"><a name="parmname4258436173317"></a><a name="parmname4258436173317"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname14391592819"><a name="parmname14391592819"></a><a name="parmname14391592819"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>训练会统计的数据的分布，训练集比较小可能会影响查询精度。</li></ul>
</td>
</tr>
</tbody>
</table>
