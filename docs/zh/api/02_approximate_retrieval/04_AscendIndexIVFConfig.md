# AscendIndexIVFConfig<a name="ZH-CN_TOPIC_0000001456535024"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456695128"></a>

AscendIndexIVF需要使用对应的AscendIndexIVFConfig执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|flatConfig|AscendIndexConfig|参数配置对象。|
|useKmeansPP|bool|是否使用NPU加速IVF聚类过程。|
|cp|ClusteringParameters|聚类相关参数，具体可以参见Faiss相关接口说明。不建议修改此参数，其中训练迭代次数参数默认为16。迭代次数设置过大，会显著增加训练时长。|

> [!NOTE]
>
> AscendIndexIVFSQConfig继承于[AscendIndexConfig](../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig)。

## AscendIndexIVFConfig接口<a name="ZH-CN_TOPIC_0000001506334629"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5591115071213"></a><a name="ul5591115071213"></a><ul id="ul5591115071213"><li><span class="parmname" id="parmname14872125571216"><a name="parmname14872125571216"></a><a name="parmname14872125571216"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname15293195861219"><a name="parmname15293195861219"></a><a name="parmname15293195861219"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue51241072364"><a name="parmvalue51241072364"></a><a name="parmvalue51241072364"></a>“-1”</span>时，Device侧<span id="ph8157732103"><a name="ph8157732103"></a><a name="ph8157732103"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9168241111516"></a><a name="ul9168241111516"></a><ul id="ul9168241111516"><li><span class="parmname" id="parmname975454381516"><a name="parmname975454381516"></a><a name="parmname975454381516"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname8790174512156"><a name="parmname8790174512156"></a><a name="parmname8790174512156"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue8480143663618"><a name="parmvalue8480143663618"></a><a name="parmvalue8480143663618"></a>“-1”</span>时，Device侧<span id="ph245516181062"><a name="ph245516181062"></a><a name="ph245516181062"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>

## SetDefaultClusteringConfig接口<a name="ZH-CN_TOPIC_0000001506495669"></a>

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
