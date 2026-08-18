# AscendIndexIVFSQTConfig<a name="ZH-CN_TOPIC_0000001506495881"></a>

AscendIndexIVFSQT需要使用对应的AscendIndexIVFSQTConfig执行对应资源的初始化。

**AscendIndexIVFSQTConfig<a name="section6579185362314"></a>**

> [!NOTE]
>AscendIndexIVFSQTConfig继承于[AscendIndexIVFSQConfig](./08_AscendIndexIVFSQConfig.md#ascendindexivfsqconfig)。

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1626263017587"></a><a name="ul1626263017587"></a><ul id="ul1626263017587"><li><span class="parmname" id="parmname512943295811"><a name="parmname512943295811"></a><a name="parmname512943295811"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname162541419202"><a name="parmname162541419202"></a><a name="parmname162541419202"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1092012916596"></a><a name="ul1092012916596"></a><ul id="ul1092012916596"><li><span class="parmname" id="parmname169381516145920"><a name="parmname169381516145920"></a><a name="parmname169381516145920"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname98182212319"><a name="parmname98182212319"></a><a name="parmname98182212319"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
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
