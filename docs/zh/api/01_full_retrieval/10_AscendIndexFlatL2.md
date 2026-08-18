# AscendIndexFlatL2<a name="ZH-CN_TOPIC_0000001456375424"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001877955534"></a>

AscendIndexFlatL2是存储FP16浮点数类型并使用L2距离的特征暴力检索算法。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

> [!NOTE]
>AscendIndexFlatL2算法支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，使用在线算子需要用户在应用程序的最后显式调用 \(void\)aclFinalize\(\) （需要包含头文件：\#include "acl/acl.h"）。

## AscendIndexFlatL2接口<a name="ZH-CN_TOPIC_0000001506495761"></a>

<a name="zh-cn_topic_0000001294312541_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294312541_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294312541_p12559123810"><a name="zh-cn_topic_0000001294312541_p12559123810"></a><a name="zh-cn_topic_0000001294312541_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294312541_p2425655144613"><a name="zh-cn_topic_0000001294312541_p2425655144613"></a><a name="zh-cn_topic_0000001294312541_p2425655144613"></a>AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294312541_p1212599383"><a name="zh-cn_topic_0000001294312541_p1212599383"></a><a name="zh-cn_topic_0000001294312541_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294312541_p131714208358"><a name="zh-cn_topic_0000001294312541_p131714208358"></a><a name="zh-cn_topic_0000001294312541_p131714208358"></a>AscendIndexFlatL2的构造函数，基于一个已有的<span class="parmname" id="zh-cn_topic_0000001294312541_parmname69451751507"><a name="zh-cn_topic_0000001294312541_parmname69451751507"></a><a name="zh-cn_topic_0000001294312541_parmname69451751507"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294312541_p112195910383"><a name="zh-cn_topic_0000001294312541_p112195910383"></a><a name="zh-cn_topic_0000001294312541_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294312541_p874812810555"><a name="zh-cn_topic_0000001294312541_p874812810555"></a><a name="zh-cn_topic_0000001294312541_p874812810555"></a><strong id="zh-cn_topic_0000001294312541_b2688145217499"><a name="zh-cn_topic_0000001294312541_b2688145217499"></a><a name="zh-cn_topic_0000001294312541_b2688145217499"></a>faiss::IndexFlatL2 *index</strong>：CPU侧Index资源。</p>
<p id="zh-cn_topic_0000001294312541_p661314244382"><a name="zh-cn_topic_0000001294312541_p661314244382"></a><a name="zh-cn_topic_0000001294312541_p661314244382"></a><strong id="zh-cn_topic_0000001294312541_b278625404911"><a name="zh-cn_topic_0000001294312541_b278625404911"></a><a name="zh-cn_topic_0000001294312541_b278625404911"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294312541_p17235973820"><a name="zh-cn_topic_0000001294312541_p17235973820"></a><a name="zh-cn_topic_0000001294312541_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294312541_p973225082318"><a name="zh-cn_topic_0000001294312541_p973225082318"></a><a name="zh-cn_topic_0000001294312541_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294312541_p182459113812"><a name="zh-cn_topic_0000001294312541_p182459113812"></a><a name="zh-cn_topic_0000001294312541_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294312541_p132314362521"><a name="zh-cn_topic_0000001294312541_p132314362521"></a><a name="zh-cn_topic_0000001294312541_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294312541_p423590386"><a name="zh-cn_topic_0000001294312541_p423590386"></a><a name="zh-cn_topic_0000001294312541_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294312541_p182559163813"><a name="zh-cn_topic_0000001294312541_p182559163813"></a><a name="zh-cn_topic_0000001294312541_p182559163813"></a><span class="parmname" id="zh-cn_topic_0000001294312541_parmname6385211185011"><a name="zh-cn_topic_0000001294312541_parmname6385211185011"></a><a name="zh-cn_topic_0000001294312541_parmname6385211185011"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为faiss::MetricType::METRIC_L2。</p>
</td>
</tr>
</tbody>
</table>

<a name="zh-cn_topic_0000001294591937_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294591937_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294591937_p12559123810"><a name="zh-cn_topic_0000001294591937_p12559123810"></a><a name="zh-cn_topic_0000001294591937_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294591937_p144102184422"><a name="zh-cn_topic_0000001294591937_p144102184422"></a><a name="zh-cn_topic_0000001294591937_p144102184422"></a>AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294591937_p1212599383"><a name="zh-cn_topic_0000001294591937_p1212599383"></a><a name="zh-cn_topic_0000001294591937_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294591937_p94401440428"><a name="zh-cn_topic_0000001294591937_p94401440428"></a><a name="zh-cn_topic_0000001294591937_p94401440428"></a>AscendIndexFlatL2的构造函数，生成维度为dims的AscendIndexFlatL2（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="zh-cn_topic_0000001294591937_parmname18694103215115"><a name="zh-cn_topic_0000001294591937_parmname18694103215115"></a><a name="zh-cn_topic_0000001294591937_parmname18694103215115"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294591937_p112195910383"><a name="zh-cn_topic_0000001294591937_p112195910383"></a><a name="zh-cn_topic_0000001294591937_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294591937_p874812810555"><a name="zh-cn_topic_0000001294591937_p874812810555"></a><a name="zh-cn_topic_0000001294591937_p874812810555"></a><strong id="zh-cn_topic_0000001294591937_b8667143775117"><a name="zh-cn_topic_0000001294591937_b8667143775117"></a><a name="zh-cn_topic_0000001294591937_b8667143775117"></a>int dims</strong>：AscendIndexFlatL2管理的一组特征向量的维度。</p>
<p id="zh-cn_topic_0000001294591937_p1220621175115"><a name="zh-cn_topic_0000001294591937_p1220621175115"></a><a name="zh-cn_topic_0000001294591937_p1220621175115"></a><strong id="zh-cn_topic_0000001294591937_b6244340115115"><a name="zh-cn_topic_0000001294591937_b6244340115115"></a><a name="zh-cn_topic_0000001294591937_b6244340115115"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294591937_p17235973820"><a name="zh-cn_topic_0000001294591937_p17235973820"></a><a name="zh-cn_topic_0000001294591937_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294591937_p973225082318"><a name="zh-cn_topic_0000001294591937_p973225082318"></a><a name="zh-cn_topic_0000001294591937_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294591937_p182459113812"><a name="zh-cn_topic_0000001294591937_p182459113812"></a><a name="zh-cn_topic_0000001294591937_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294591937_p132314362521"><a name="zh-cn_topic_0000001294591937_p132314362521"></a><a name="zh-cn_topic_0000001294591937_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294591937_p423590386"><a name="zh-cn_topic_0000001294591937_p423590386"></a><a name="zh-cn_topic_0000001294591937_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294591937_p1229447954"><a name="zh-cn_topic_0000001294591937_p1229447954"></a><a name="zh-cn_topic_0000001294591937_p1229447954"></a>dims ∈ {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 4096, 3584}</p>
</td>
</tr>
</tbody>
</table>

<a name="zh-cn_topic_0000001247793230_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001247793230_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001247793230_p12559123810"><a name="zh-cn_topic_0000001247793230_p12559123810"></a><a name="zh-cn_topic_0000001247793230_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001247793230_p7112274471"><a name="zh-cn_topic_0000001247793230_p7112274471"></a><a name="zh-cn_topic_0000001247793230_p7112274471"></a>AscendIndexFlatL2(const AscendIndexFlatL2&amp;) = delete;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001247793230_p1212599383"><a name="zh-cn_topic_0000001247793230_p1212599383"></a><a name="zh-cn_topic_0000001247793230_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001247793230_p131714208358"><a name="zh-cn_topic_0000001247793230_p131714208358"></a><a name="zh-cn_topic_0000001247793230_p131714208358"></a>声明此Index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001247793230_p112195910383"><a name="zh-cn_topic_0000001247793230_p112195910383"></a><a name="zh-cn_topic_0000001247793230_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001247793230_p867213174418"><a name="zh-cn_topic_0000001247793230_p867213174418"></a><a name="zh-cn_topic_0000001247793230_p867213174418"></a><strong id="zh-cn_topic_0000001247793230_b322283735213"><a name="zh-cn_topic_0000001247793230_b322283735213"></a><a name="zh-cn_topic_0000001247793230_b322283735213"></a>const AscendIndexFlatL2&amp;</strong>：常量AscendIndexFlatL2。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001247793230_p17235973820"><a name="zh-cn_topic_0000001247793230_p17235973820"></a><a name="zh-cn_topic_0000001247793230_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001247793230_p973225082318"><a name="zh-cn_topic_0000001247793230_p973225082318"></a><a name="zh-cn_topic_0000001247793230_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001247793230_p182459113812"><a name="zh-cn_topic_0000001247793230_p182459113812"></a><a name="zh-cn_topic_0000001247793230_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001247793230_p132314362521"><a name="zh-cn_topic_0000001247793230_p132314362521"></a><a name="zh-cn_topic_0000001247793230_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001247793230_p423590386"><a name="zh-cn_topic_0000001247793230_p423590386"></a><a name="zh-cn_topic_0000001247793230_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001247793230_p182559163813"><a name="zh-cn_topic_0000001247793230_p182559163813"></a><a name="zh-cn_topic_0000001247793230_p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="zh-cn_topic_0000001294312453_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294312453_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294312453_p12559123810"><a name="zh-cn_topic_0000001294312453_p12559123810"></a><a name="zh-cn_topic_0000001294312453_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294312453_p132681218211"><a name="zh-cn_topic_0000001294312453_p132681218211"></a><a name="zh-cn_topic_0000001294312453_p132681218211"></a>virtual ~AscendIndexFlatL2()</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294312453_p1212599383"><a name="zh-cn_topic_0000001294312453_p1212599383"></a><a name="zh-cn_topic_0000001294312453_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294312453_p131714208358"><a name="zh-cn_topic_0000001294312453_p131714208358"></a><a name="zh-cn_topic_0000001294312453_p131714208358"></a>AscendIndexFlatL2的析构函数，销毁AscendIndexFlatL2对象，释放资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294312453_p112195910383"><a name="zh-cn_topic_0000001294312453_p112195910383"></a><a name="zh-cn_topic_0000001294312453_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294312453_p8451184515218"><a name="zh-cn_topic_0000001294312453_p8451184515218"></a><a name="zh-cn_topic_0000001294312453_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294312453_p17235973820"><a name="zh-cn_topic_0000001294312453_p17235973820"></a><a name="zh-cn_topic_0000001294312453_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294312453_p973225082318"><a name="zh-cn_topic_0000001294312453_p973225082318"></a><a name="zh-cn_topic_0000001294312453_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294312453_p182459113812"><a name="zh-cn_topic_0000001294312453_p182459113812"></a><a name="zh-cn_topic_0000001294312453_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294312453_p132314362521"><a name="zh-cn_topic_0000001294312453_p132314362521"></a><a name="zh-cn_topic_0000001294312453_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294312453_p423590386"><a name="zh-cn_topic_0000001294312453_p423590386"></a><a name="zh-cn_topic_0000001294312453_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294312453_p182559163813"><a name="zh-cn_topic_0000001294312453_p182559163813"></a><a name="zh-cn_topic_0000001294312453_p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456375400"></a>

<a name="zh-cn_topic_0000001248112146_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001248112146_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001248112146_p12559123810"><a name="zh-cn_topic_0000001248112146_p12559123810"></a><a name="zh-cn_topic_0000001248112146_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001248112146_p1531315343445"><a name="zh-cn_topic_0000001248112146_p1531315343445"></a><a name="zh-cn_topic_0000001248112146_p1531315343445"></a>void copyFrom(faiss::IndexFlat *index);</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001248112146_p1212599383"><a name="zh-cn_topic_0000001248112146_p1212599383"></a><a name="zh-cn_topic_0000001248112146_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001248112146_p131714208358"><a name="zh-cn_topic_0000001248112146_p131714208358"></a><a name="zh-cn_topic_0000001248112146_p131714208358"></a>AscendIndexFlat基于一个已有的<span class="parmname" id="parmname1804125953520"><a name="parmname1804125953520"></a><a name="parmname1804125953520"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexFlatL2底库，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001248112146_p112195910383"><a name="zh-cn_topic_0000001248112146_p112195910383"></a><a name="zh-cn_topic_0000001248112146_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001248112146_p874812810555"><a name="zh-cn_topic_0000001248112146_p874812810555"></a><a name="zh-cn_topic_0000001248112146_p874812810555"></a><strong id="zh-cn_topic_0000001248112146_b976174655318"><a name="zh-cn_topic_0000001248112146_b976174655318"></a><a name="zh-cn_topic_0000001248112146_b976174655318"></a>const faiss::IndexFlat *index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001248112146_p17235973820"><a name="zh-cn_topic_0000001248112146_p17235973820"></a><a name="zh-cn_topic_0000001248112146_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001248112146_p973225082318"><a name="zh-cn_topic_0000001248112146_p973225082318"></a><a name="zh-cn_topic_0000001248112146_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001248112146_p182459113812"><a name="zh-cn_topic_0000001248112146_p182459113812"></a><a name="zh-cn_topic_0000001248112146_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001248112146_p132314362521"><a name="zh-cn_topic_0000001248112146_p132314362521"></a><a name="zh-cn_topic_0000001248112146_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001248112146_p423590386"><a name="zh-cn_topic_0000001248112146_p423590386"></a><a name="zh-cn_topic_0000001248112146_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001248112146_p182559163813"><a name="zh-cn_topic_0000001248112146_p182559163813"></a><a name="zh-cn_topic_0000001248112146_p182559163813"></a><span class="parmname" id="zh-cn_topic_0000001248112146_parmname159121156135315"><a name="zh-cn_topic_0000001248112146_parmname159121156135315"></a><a name="zh-cn_topic_0000001248112146_parmname159121156135315"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3584}，底库向量总数的取值范围：0 &lt;= n &lt; 1e9，metric_type参数取值为faiss::MetricType::METRIC_L2。</p>
</td>
</tr>
</tbody>
</table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001456535052"></a>

<a name="zh-cn_topic_0000001247793178_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001247793178_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001247793178_p12559123810"><a name="zh-cn_topic_0000001247793178_p12559123810"></a><a name="zh-cn_topic_0000001247793178_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001247793178_p10713954155218"><a name="zh-cn_topic_0000001247793178_p10713954155218"></a><a name="zh-cn_topic_0000001247793178_p10713954155218"></a>void copyTo(faiss::IndexFlat *index);</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001247793178_p1212599383"><a name="zh-cn_topic_0000001247793178_p1212599383"></a><a name="zh-cn_topic_0000001247793178_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001247793178_p131714208358"><a name="zh-cn_topic_0000001247793178_p131714208358"></a><a name="zh-cn_topic_0000001247793178_p131714208358"></a>将AscendIndexFlatL2的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001247793178_p112195910383"><a name="zh-cn_topic_0000001247793178_p112195910383"></a><a name="zh-cn_topic_0000001247793178_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001247793178_p874812810555"><a name="zh-cn_topic_0000001247793178_p874812810555"></a><a name="zh-cn_topic_0000001247793178_p874812810555"></a><strong id="zh-cn_topic_0000001247793178_b2644689548"><a name="zh-cn_topic_0000001247793178_b2644689548"></a><a name="zh-cn_topic_0000001247793178_b2644689548"></a>faiss::IndexFlat *index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001247793178_p17235973820"><a name="zh-cn_topic_0000001247793178_p17235973820"></a><a name="zh-cn_topic_0000001247793178_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001247793178_p973225082318"><a name="zh-cn_topic_0000001247793178_p973225082318"></a><a name="zh-cn_topic_0000001247793178_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001247793178_p182459113812"><a name="zh-cn_topic_0000001247793178_p182459113812"></a><a name="zh-cn_topic_0000001247793178_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001247793178_p132314362521"><a name="zh-cn_topic_0000001247793178_p132314362521"></a><a name="zh-cn_topic_0000001247793178_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001247793178_p423590386"><a name="zh-cn_topic_0000001247793178_p423590386"></a><a name="zh-cn_topic_0000001247793178_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001247793178_p182559163813"><a name="zh-cn_topic_0000001247793178_p182559163813"></a><a name="zh-cn_topic_0000001247793178_p182559163813"></a><span class="parmname" id="zh-cn_topic_0000001247793178_parmname683216143547"><a name="zh-cn_topic_0000001247793178_parmname683216143547"></a><a name="zh-cn_topic_0000001247793178_parmname683216143547"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001456695116"></a>

<a name="zh-cn_topic_0000001294432513_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294432513_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294432513_p12559123810"><a name="zh-cn_topic_0000001294432513_p12559123810"></a><a name="zh-cn_topic_0000001294432513_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294432513_p1213215268503"><a name="zh-cn_topic_0000001294432513_p1213215268503"></a><a name="zh-cn_topic_0000001294432513_p1213215268503"></a>AscendIndexFlatL2&amp; operator=(const AscendIndexFlatL2&amp;) = delete;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294432513_p1212599383"><a name="zh-cn_topic_0000001294432513_p1212599383"></a><a name="zh-cn_topic_0000001294432513_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294432513_p131714208358"><a name="zh-cn_topic_0000001294432513_p131714208358"></a><a name="zh-cn_topic_0000001294432513_p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294432513_p112195910383"><a name="zh-cn_topic_0000001294432513_p112195910383"></a><a name="zh-cn_topic_0000001294432513_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294432513_p867213174418"><a name="zh-cn_topic_0000001294432513_p867213174418"></a><a name="zh-cn_topic_0000001294432513_p867213174418"></a><strong id="zh-cn_topic_0000001294432513_b12571191511538"><a name="zh-cn_topic_0000001294432513_b12571191511538"></a><a name="zh-cn_topic_0000001294432513_b12571191511538"></a>const AscendIndexFlatL2&amp;</strong>：常量AscendIndexFlatL2。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294432513_p17235973820"><a name="zh-cn_topic_0000001294432513_p17235973820"></a><a name="zh-cn_topic_0000001294432513_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294432513_p973225082318"><a name="zh-cn_topic_0000001294432513_p973225082318"></a><a name="zh-cn_topic_0000001294432513_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294432513_p182459113812"><a name="zh-cn_topic_0000001294432513_p182459113812"></a><a name="zh-cn_topic_0000001294432513_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294432513_p132314362521"><a name="zh-cn_topic_0000001294432513_p132314362521"></a><a name="zh-cn_topic_0000001294432513_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294432513_p423590386"><a name="zh-cn_topic_0000001294432513_p423590386"></a><a name="zh-cn_topic_0000001294432513_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294432513_p182559163813"><a name="zh-cn_topic_0000001294432513_p182559163813"></a><a name="zh-cn_topic_0000001294432513_p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>
