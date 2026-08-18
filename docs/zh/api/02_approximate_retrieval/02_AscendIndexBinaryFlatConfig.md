# AscendIndexBinaryFlatConfig<a name="ZH-CN_TOPIC_0000001506495777"></a>

AscendIndexBinaryFlat需要使用对应的AscendIndexBinaryFlatConfig执行对应资源的初始化，配置执行检索过程中的硬件资源“devices”和预置的内存池大小“resources”。

- AscendIndexBinaryFlat仅支持单个昇腾AI处理器的<term>Atlas 推理系列产品</term>，依赖AICPU算子和BinaryFlat算子，请参考[自定义算子介绍](../../05_user_guide.md#自定义算子介绍)生成对应算子。
- AscendIndexBinaryFlat仅支持标准态部署方式。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|deviceList|std::vector\<int>|Device侧设备ID。AscendIndexBinaryFlat类仅支持单个<term>Atlas 推理系列产品</term>的加速卡。|
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>AscendIndexBinaryFlat仅支持单个<span id="ph13714132410186"><a name="ph13714132410186"></a><a name="ph13714132410186"></a>昇腾AI处理器</span>的<span id="ph87140243189"><a name="ph87140243189"></a><a name="ph87140243189"></a><term>Atlas 推理系列产品</term></span>，如果第0个<span id="ph112871028144417"><a name="ph112871028144417"></a><a name="ph112871028144417"></a>昇腾AI处理器</span>不可用则无法使用默认构造。</p>
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
