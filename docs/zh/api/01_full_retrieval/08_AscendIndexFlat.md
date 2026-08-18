# AscendIndexFlat<a id="ZH-CN_TOPIC_0000001506334757"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506334829"></a>

AscendIndexFlat是最基础的特征检索，存储FP16浮点数类型特征向量并执行暴力搜索。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

> [!NOTE]
>AscendIndexFlat算法L2和IP距离支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，使用在线算子需要用户在应用程序的最后显式调用 \(void\)aclFinalize\(\) （需要包含头文件：\#include "acl/acl.h"）。

## AscendIndexFlat接口<a name="ZH-CN_TOPIC_0000001456375308"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a>AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexFlat的构造函数，基于一个已有的<span class="parmname" id="parmname186437475368"><a name="parmname186437475368"></a><a name="parmname186437475368"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b1735185317361"><a name="b1735185317361"></a><a name="b1735185317361"></a>const faiss::IndexFlat *index</strong>：CPU侧index资源。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1693555513368"><a name="b1693555513368"></a><a name="b1693555513368"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6239525111613"><a name="p6239525111613"></a><a name="p6239525111613"></a><span class="parmname" id="parmname721815874613"><a name="parmname721815874613"></a><a name="parmname721815874613"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1735274911381"></a>
<table><tbody><tr id="row163522495386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p113526492389"><a name="p113526492389"></a><a name="p113526492389"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p935216491386"><a name="p935216491386"></a><a name="p935216491386"></a>AscendIndexFlat(const faiss::IndexIDMap *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="row14352124915385"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p2352749113819"><a name="p2352749113819"></a><a name="p2352749113819"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13528493381"><a name="p13528493381"></a><a name="p13528493381"></a>AscendIndexFlat的构造函数，基于一个已有的<span class="parmname" id="parmname1212316485395"><a name="parmname1212316485395"></a><a name="parmname1212316485395"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row13352184923815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p535219495381"><a name="p535219495381"></a><a name="p535219495381"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p11352164923819"><a name="p11352164923819"></a><a name="p11352164923819"></a><strong id="b37691251203911"><a name="b37691251203911"></a><a name="b37691251203911"></a>const faiss::IndexIDMap *index</strong>：CPU侧Index资源。</p>
<p id="p14352104915380"><a name="p14352104915380"></a><a name="p14352104915380"></a><strong id="b20797653143915"><a name="b20797653143915"></a><a name="b20797653143915"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row5352154943813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p12352114933817"><a name="p12352114933817"></a><a name="p12352114933817"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16352194953814"><a name="p16352194953814"></a><a name="p16352194953814"></a>无</p>
</td>
</tr>
<tr id="row1735214943812"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1535284912383"><a name="p1535284912383"></a><a name="p1535284912383"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p535264943818"><a name="p535264943818"></a><a name="p535264943818"></a>无</p>
</td>
</tr>
<tr id="row235216491381"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1035274933810"><a name="p1035274933810"></a><a name="p1035274933810"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname96759552468"><a name="parmname96759552468"></a><a name="parmname96759552468"></a>“index”</span>需要为合法有效的IndexIDMap指针，该Index的成员索引维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table142416323911"></a>
<table><tbody><tr id="row1257343916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p19251332393"><a name="p19251332393"></a><a name="p19251332393"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11207257504"><a name="p11207257504"></a><a name="p11207257504"></a>AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="row2258310398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p9252383918"><a name="p9252383918"></a><a name="p9252383918"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p102553123917"><a name="p102553123917"></a><a name="p102553123917"></a>AscendIndexFlat的构造函数，生成维度为dims的AscendIndexFlat（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="parmname1614731717426"><a name="parmname1614731717426"></a><a name="parmname1614731717426"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row1525633399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p16259393917"><a name="p16259393917"></a><a name="p16259393917"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10256318392"><a name="p10256318392"></a><a name="p10256318392"></a><strong id="b13406132154210"><a name="b13406132154210"></a><a name="b13406132154210"></a>int dims</strong>：AscendIndex管理的一组特征向量的维度。</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1658993044220"><a name="b1658993044220"></a><a name="b1658993044220"></a>faiss::MetricType metric</strong>：AscendIndexFlat在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b159198328421"><a name="b159198328421"></a><a name="b159198328421"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row102514316397"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7254343920"><a name="p7254343920"></a><a name="p7254343920"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p22553153917"><a name="p22553153917"></a><a name="p22553153917"></a>无</p>
</td>
</tr>
<tr id="row22516313918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p62511393920"><a name="p62511393920"></a><a name="p62511393920"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p92519310395"><a name="p92519310395"></a><a name="p92519310395"></a>无</p>
</td>
</tr>
<tr id="row6251237398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p18251437396"><a name="p18251437396"></a><a name="p18251437396"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1988754314420"></a><a name="ul1988754314420"></a><ul id="ul1988754314420"><li>dims ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table5169814143913"></a>
<table><tbody><tr id="row1116961423914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p4169121473916"><a name="p4169121473916"></a><a name="p4169121473916"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7112274471"><a name="p7112274471"></a><a name="p7112274471"></a>AscendIndexFlat(const AscendIndexFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row1416991413916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p121699143394"><a name="p121699143394"></a><a name="p121699143394"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p191699142396"><a name="p191699142396"></a><a name="p191699142396"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row61691614163913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5169814123916"><a name="p5169814123916"></a><a name="p5169814123916"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b8275365435"><a name="b8275365435"></a><a name="b8275365435"></a>const AscendIndexFlat&amp;</strong>：常量AscendIndexFlat。</p>
</td>
</tr>
<tr id="row151691414153917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1016941403913"><a name="p1016941403913"></a><a name="p1016941403913"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p21691714113915"><a name="p21691714113915"></a><a name="p21691714113915"></a>无</p>
</td>
</tr>
<tr id="row181697141391"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3169161413395"><a name="p3169161413395"></a><a name="p3169161413395"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p01691714163917"><a name="p01691714163917"></a><a name="p01691714163917"></a>无</p>
</td>
</tr>
<tr id="row1416991443915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p01691014143911"><a name="p01691014143911"></a><a name="p01691014143911"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p161695143397"><a name="p161695143397"></a><a name="p161695143397"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table04891725153918"></a>
<table><tbody><tr id="row194894256391"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p13489525203913"><a name="p13489525203913"></a><a name="p13489525203913"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexFlat();</p>
</td>
</tr>
<tr id="row1248962513399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p94891225163914"><a name="p94891225163914"></a><a name="p94891225163914"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1148902511397"><a name="p1148902511397"></a><a name="p1148902511397"></a>AscendIndexFlat的析构函数，销毁AscendIndexFlat对象，释放资源。</p>
</td>
</tr>
<tr id="row15489182583911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p10489142503915"><a name="p10489142503915"></a><a name="p10489142503915"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row6489525163919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p10489152514391"><a name="p10489152514391"></a><a name="p10489152514391"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p13489425133914"><a name="p13489425133914"></a><a name="p13489425133914"></a>无</p>
</td>
</tr>
<tr id="row1248992503912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p548912515395"><a name="p548912515395"></a><a name="p548912515395"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16489182512392"><a name="p16489182512392"></a><a name="p16489182512392"></a>无</p>
</td>
</tr>
<tr id="row16489725193918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p134891125183917"><a name="p134891125183917"></a><a name="p134891125183917"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15489102517393"><a name="p15489102517393"></a><a name="p15489102517393"></a>无</p>
</td>
</tr>
</tbody>
</table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456535180"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyFrom(const faiss::IndexFlat *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexFlat基于一个已有的Index拷贝到Ascend，清空当前的AscendIndexFlat底库，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b166777494441"><a name="b166777494441"></a><a name="b166777494441"></a>const faiss::IndexFlat *index</strong>：CPU侧index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname16174215239"><a name="parmname16174215239"></a><a name="parmname16174215239"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table525914213409"></a>
<table><tbody><tr id="row16259174214406"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1025994215407"><a name="p1025994215407"></a><a name="p1025994215407"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6259442124016"><a name="p6259442124016"></a><a name="p6259442124016"></a>void copyFrom(const faiss::IndexIDMap *index);</p>
</td>
</tr>
<tr id="row1925914423401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1025984212403"><a name="p1025984212403"></a><a name="p1025984212403"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p92592422406"><a name="p92592422406"></a><a name="p92592422406"></a>AscendIndexFlat基于一个已有的<span class="parmname" id="parmname063811221710"><a name="parmname063811221710"></a><a name="parmname063811221710"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexFlat底库，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row5259842124019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7259174224012"><a name="p7259174224012"></a><a name="p7259174224012"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p925984284010"><a name="p925984284010"></a><a name="p925984284010"></a><strong id="b95691227174516"><a name="b95691227174516"></a><a name="b95691227174516"></a>const faiss::IndexIDMap *index</strong>：CPU侧index资源。</p>
</td>
</tr>
<tr id="row14259042124016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1125954204015"><a name="p1125954204015"></a><a name="p1125954204015"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14260174234011"><a name="p14260174234011"></a><a name="p14260174234011"></a>无</p>
</td>
</tr>
<tr id="row20260174294018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12260842114015"><a name="p12260842114015"></a><a name="p12260842114015"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p15260242114012"><a name="p15260242114012"></a><a name="p15260242114012"></a>无</p>
</td>
</tr>
<tr id="row1626015428401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p152607429407"><a name="p152607429407"></a><a name="p152607429407"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1726012424402"><a name="p1726012424402"></a><a name="p1726012424402"></a>index需要为合法有效的IndexIDMap指针，否则可能造成程序崩溃或功能不可用，该Index的成员索引维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001456535148"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexFlat *index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>将AscendIndexFlat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b194718510463"><a name="b194718510463"></a><a name="b194718510463"></a>faiss::IndexFlat *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname773416518233"><a name="parmname773416518233"></a><a name="parmname773416518233"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

<a name="table154531752144016"></a>
<table><tbody><tr id="row12453652124015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1245375234010"><a name="p1245375234010"></a><a name="p1245375234010"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p194531452144019"><a name="p194531452144019"></a><a name="p194531452144019"></a>void copyTo(faiss::IndexIDMap *index) const;</p>
</td>
</tr>
<tr id="row74535524403"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7453752174010"><a name="p7453752174010"></a><a name="p7453752174010"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p045325210409"><a name="p045325210409"></a><a name="p045325210409"></a>将AscendIndexFlat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row11453135211406"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p94531252184014"><a name="p94531252184014"></a><a name="p94531252184014"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16453175224020"><a name="p16453175224020"></a><a name="p16453175224020"></a><strong id="b172171419114612"><a name="b172171419114612"></a><a name="b172171419114612"></a>faiss::IndexIDMap *index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="row345495215407"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p8454135218406"><a name="p8454135218406"></a><a name="p8454135218406"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p74541752174010"><a name="p74541752174010"></a><a name="p74541752174010"></a>无</p>
</td>
</tr>
<tr id="row19454852184017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3454752144017"><a name="p3454752144017"></a><a name="p3454752144017"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p145465264013"><a name="p145465264013"></a><a name="p145465264013"></a>无</p>
</td>
</tr>
<tr id="row845415211403"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p14541452144011"><a name="p14541452144011"></a><a name="p14541452144011"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p24541452194012"><a name="p24541452194012"></a><a name="p24541452194012"></a><span class="parmname" id="parmname13154121312411"><a name="parmname13154121312411"></a><a name="parmname13154121312411"></a>“index”</span>需要为合法有效的IndexIDMap指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

## getBase接口<a name="ZH-CN_TOPIC_0000001456375236"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getBase(int deviceId, char* xb) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexFlat在特定<span class="parmname" id="parmname156459015472"><a name="parmname156459015472"></a><a name="parmname156459015472"></a>“deviceId”</span>上管理的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b10126123144712"><a name="b10126123144712"></a><a name="b10126123144712"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b1957812517475"><a name="b1957812517475"></a><a name="b1957812517475"></a>char* xb</strong>：AscendIndexFlat在<span class="parmname" id="parmname12584191319472"><a name="parmname12584191319472"></a><a name="parmname12584191319472"></a>“deviceId”</span>上存储的底库特征向量。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname645613115475"><a name="parmname645613115475"></a><a name="parmname645613115475"></a>“deviceId”</span>需要为合法值的设备ID。</p>
<p id="p835381215917"><a name="p835381215917"></a><a name="p835381215917"></a><span class="parmname" id="parmname14391592819"><a name="parmname14391592819"></a><a name="parmname14391592819"></a>“xb”</span>需要为非空指针，且长度应该为dims * BaseSize * sizeof(float32)字节，否则可能出现越界读写错误并引起程序崩溃，其中BaseSize为getBaseSize函数的返回值。</p>
</td>
</tr>
</tbody>
</table>

## getBaseSize接口<a name="ZH-CN_TOPIC_0000001456854956"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>size_t getBaseSize(int deviceId) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexFlat在特定<span class="parmname" id="parmname5505194594611"><a name="parmname5505194594611"></a><a name="parmname5505194594611"></a>“deviceId”</span>上管理的特征向量数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b152596352465"><a name="b152596352465"></a><a name="b152596352465"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>在特定<span class="parmname" id="parmname362764114465"><a name="parmname362764114465"></a><a name="parmname362764114465"></a>“deviceId”</span>上的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname13287173918465"><a name="parmname13287173918465"></a><a name="parmname13287173918465"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## getIdxMap接口<a name="ZH-CN_TOPIC_0000001506334785"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexFlat在特定<span class="parmname" id="parmname141894416475"><a name="parmname141894416475"></a><a name="parmname141894416475"></a>“deviceId”</span>上管理的特征向量ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b18839174713474"><a name="b18839174713474"></a><a name="b18839174713474"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b2587150104719"><a name="b2587150104719"></a><a name="b2587150104719"></a>std::vector&lt;idx_t&gt; &amp;idxMap</strong>：AscendIndexFlat在<span class="parmname" id="parmname27675634719"><a name="parmname27675634719"></a><a name="parmname27675634719"></a>“deviceId”</span>上存储的底库特征向量ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname886865894720"><a name="parmname886865894720"></a><a name="parmname886865894720"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506495701"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexFlat&amp; operator=(const AscendIndexFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b1210811552436"><a name="b1210811552436"></a><a name="b1210811552436"></a>const AscendIndexFlat&amp;</strong>：常量AscendIndexFlat。</p>
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

## search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001810529650"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11977033135012"><a name="p11977033135012"></a><a name="p11977033135012"></a>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7584153011582"><a name="p7584153011582"></a><a name="p7584153011582"></a>AscendIndexFlat的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为<strong id="b177781744812"><a name="b177781744812"></a><a name="b177781744812"></a>0</strong>、<strong id="b61111618184814"><a name="b61111618184814"></a><a name="b61111618184814"></a>1</strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong id="b18384163024810"><a name="b18384163024810"></a><a name="b18384163024810"></a>1</strong>参与，<strong id="b2070119316483"><a name="b2070119316483"></a><a name="b2070119316483"></a>0</strong>不参与。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b8509184711813"><a name="b8509184711813"></a><a name="b8509184711813"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b2086114494814"><a name="b2086114494814"></a><a name="b2086114494814"></a>const float *x</strong>：特征向量数据。</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b2484155112813"><a name="b2484155112813"></a><a name="b2484155112813"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p841235065815"><a name="p841235065815"></a><a name="p841235065815"></a><strong id="b77116531187"><a name="b77116531187"></a><a name="b77116531187"></a>const void *mask</strong>：特征底库掩码。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul184581435495"></a><a name="ul184581435495"></a><ul id="ul184581435495"><li>此处<span class="parmname" id="parmname11945191043317"><a name="parmname11945191043317"></a><a name="parmname11945191043317"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处<span class="parmname" id="parmname118167143338"><a name="parmname118167143338"></a><a name="parmname118167143338"></a>“k”</span>通常不允许超过4096。</li><li>此处指针<span class="parmname" id="parmname2847104218106"><a name="parmname2847104218106"></a><a name="parmname2847104218106"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname142691946111020"><a name="parmname142691946111020"></a><a name="parmname142691946111020"></a>“distances”</span>/<span class="parmname" id="parmname2586184851015"><a name="parmname2586184851015"></a><a name="parmname2586184851015"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b101511878307"><a name="b101511878307"></a><a name="b101511878307"></a>k</strong> * <strong id="b717661318306"><a name="b717661318306"></a><a name="b717661318306"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname199479521105"><a name="parmname199479521105"></a><a name="parmname199479521105"></a>“mask”</span>需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table0628133121511"></a>
<table><tbody><tr id="row5682739155"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.1.1"><p id="p156823331511"><a name="p156823331511"></a><a name="p156823331511"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.1.1 "><p id="p1868293131514"><a name="p1868293131514"></a><a name="p1868293131514"></a>void search_with_masks(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row1368233181518"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.2.1"><p id="p36821633157"><a name="p36821633157"></a><a name="p36821633157"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p8682432157"><a name="p8682432157"></a><a name="p8682432157"></a>AscendIndexFlat的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为0、1比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，1参与，0不参与。</p>
</td>
</tr>
<tr id="row196837312153"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.3.1"><p id="p10683233157"><a name="p10683233157"></a><a name="p10683233157"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.3.1 "><p id="p1768318381520"><a name="p1768318381520"></a><a name="p1768318381520"></a><strong id="b11627624191518"><a name="b11627624191518"></a><a name="b11627624191518"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p2683123121519"><a name="p2683123121519"></a><a name="p2683123121519"></a><strong id="b151811427191518"><a name="b151811427191518"></a><a name="b151811427191518"></a>const uint16_t *x</strong>：特征向量数据。</p>
<p id="p146831137157"><a name="p146831137157"></a><a name="p146831137157"></a><strong id="b16567292155"><a name="b16567292155"></a><a name="b16567292155"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p768320371520"><a name="p768320371520"></a><a name="p768320371520"></a><strong id="b136051632191510"><a name="b136051632191510"></a><a name="b136051632191510"></a>const void *mask</strong>：特征底库掩码。</p>
</td>
</tr>
<tr id="row1868310301516"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.4.1"><p id="p16683193111511"><a name="p16683193111511"></a><a name="p16683193111511"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.4.1 "><p id="p1168315312152"><a name="p1168315312152"></a><a name="p1168315312152"></a><strong id="b1930365216156"><a name="b1930365216156"></a><a name="b1930365216156"></a>float *distances</strong>：查询向量与距离最近的前“k”个向量间的距离值。</p>
<p id="p10683133158"><a name="p10683133158"></a><a name="p10683133158"></a><strong id="b952215610159"><a name="b952215610159"></a><a name="b952215610159"></a>idx_t *labels</strong>：查询的距离最近的前“k”个向量的ID。</p>
</td>
</tr>
<tr id="row1668310317152"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.5.1"><p id="p19683203141517"><a name="p19683203141517"></a><a name="p19683203141517"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.5.1 "><p id="p468393131518"><a name="p468393131518"></a><a name="p468393131518"></a>无</p>
</td>
</tr>
<tr id="row1768312318157"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.6.1"><p id="p1368343181514"><a name="p1368343181514"></a><a name="p1368343181514"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><a name="ul16154204141611"></a><a name="ul16154204141611"></a><ul id="ul16154204141611"><li>此处“n”的取值范围：0 &lt; n &lt; 1e9。</li><li>此处“k”通常不允许超过4096。</li><li>此处指针“x”需要为非空指针，且长度应该为dim * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针“distances”/“labels”需要为非空指针，且长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针“mask”需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>
