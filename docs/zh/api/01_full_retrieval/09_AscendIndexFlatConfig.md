# AscendIndexFlatConfig<a name="ZH-CN_TOPIC_0000001456375216"></a>

AscendIndexFlat需要使用对应的AscendIndexFlatConfig执行对应资源的初始化。

**接口说明<a name="section140920164419"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>inline AscendIndexFlatConfig()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexFlatConfig的默认构造函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table46951722104415"></a>
<table><tbody><tr id="row186961822204410"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1069662218442"><a name="p1069662218442"></a><a name="p1069662218442"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1792514755010"><a name="p1792514755010"></a><a name="p1792514755010"></a>inline AscendIndexFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row169692210443"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p969622274410"><a name="p969622274410"></a><a name="p969622274410"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7696172274411"><a name="p7696172274411"></a><a name="p7696172274411"></a>AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据<span class="parmname" id="parmname468135223212"><a name="parmname468135223212"></a><a name="parmname468135223212"></a>“devices”</span>中配置的值设置Device侧<span id="ph10238132515618"><a name="ph10238132515618"></a><a name="ph10238132515618"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row136963220449"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p146961622204418"><a name="p146961622204418"></a><a name="p146961622204418"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p5696122214448"><a name="p5696122214448"></a><a name="p5696122214448"></a><strong id="b1274764723010"><a name="b1274764723010"></a><a name="b1274764723010"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p17641112731910"><a name="p17641112731910"></a><a name="p17641112731910"></a><strong id="b13831175053011"><a name="b13831175053011"></a><a name="b13831175053011"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname1331442416116"><a name="parmname1331442416116"></a><a name="parmname1331442416116"></a>“FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。</p>
<a name="ul1067904281918"></a><a name="ul1067904281918"></a><ul id="ul1067904281918"><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue17434174411812"><a name="parmvalue17434174411812"></a><a name="parmvalue17434174411812"></a>“faiss::METRIC_L2”</span>时建议设置1024MB。</li><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue634861211200"><a name="parmvalue634861211200"></a><a name="parmvalue634861211200"></a>“faiss::METRIC_INNER_PRODUCT”</span>时建议设置1280MB。</li></ul>
</td>
</tr>
<tr id="row16696172214415"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1069610226440"><a name="p1069610226440"></a><a name="p1069610226440"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p669620222449"><a name="p669620222449"></a><a name="p669620222449"></a>无</p>
</td>
</tr>
<tr id="row16696192264412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3696922114411"><a name="p3696922114411"></a><a name="p3696922114411"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p56962225445"><a name="p56962225445"></a><a name="p56962225445"></a>无</p>
</td>
</tr>
<tr id="row169602211448"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1469617229447"><a name="p1469617229447"></a><a name="p1469617229447"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul44541139318"></a><a name="ul44541139318"></a><ul id="ul44541139318"><li><span class="parmname" id="parmname1793182919311"><a name="parmname1793182919311"></a><a name="parmname1793182919311"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname15293195861219"><a name="parmname15293195861219"></a><a name="parmname15293195861219"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue13120151692414"><a name="parmvalue13120151692414"></a><a name="parmvalue13120151692414"></a>“-1”</span>时，Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table842319354444"></a>
<table><tbody><tr id="row1142318355442"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1842393554413"><a name="p1842393554413"></a><a name="p1842393554413"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p593315225213"><a name="p593315225213"></a><a name="p593315225213"></a>inline AscendIndexFlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row1242323524413"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p84231035164418"><a name="p84231035164418"></a><a name="p84231035164418"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1142333517449"><a name="p1142333517449"></a><a name="p1142333517449"></a>AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据<span class="parmname" id="parmname2062863616353"><a name="parmname2062863616353"></a><a name="parmname2062863616353"></a>“devices”</span>中配置的值设置Device侧<span id="ph37713595611"><a name="ph37713595611"></a><a name="ph37713595611"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row94235350446"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p742383516443"><a name="p742383516443"></a><a name="p742383516443"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14423173514449"><a name="p14423173514449"></a><a name="p14423173514449"></a><strong id="b10613041173517"><a name="b10613041173517"></a><a name="b10613041173517"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b58045433352"><a name="b58045433352"></a><a name="b58045433352"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname13423133524413"><a name="parmname13423133524413"></a><a name="parmname13423133524413"></a>“FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。</p>
<a name="ul3423163517446"></a><a name="ul3423163517446"></a><ul id="ul3423163517446"><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue34231835124411"><a name="parmvalue34231835124411"></a><a name="parmvalue34231835124411"></a>“faiss::METRIC_L2”</span>时建议设置1024MB。</li><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue124230354440"><a name="parmvalue124230354440"></a><a name="parmvalue124230354440"></a>“faiss::METRIC_INNER_PRODUCT”</span>时建议设置1280MB。</li></ul>
</td>
</tr>
<tr id="row1842343514447"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p642343513448"><a name="p642343513448"></a><a name="p642343513448"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p4424183564416"><a name="p4424183564416"></a><a name="p4424183564416"></a>无</p>
</td>
</tr>
<tr id="row11424135174412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p24244358441"><a name="p24244358441"></a><a name="p24244358441"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16424135124411"><a name="p16424135124411"></a><a name="p16424135124411"></a>无</p>
</td>
</tr>
<tr id="row14424183594412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p17424133524410"><a name="p17424133524410"></a><a name="p17424133524410"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1811714915354"></a><a name="ul1811714915354"></a><ul id="ul1811714915354"><li><span class="parmname" id="parmname06427236368"><a name="parmname06427236368"></a><a name="parmname06427236368"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname442411352446"><a name="parmname442411352446"></a><a name="parmname442411352446"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue31828359242"><a name="parmvalue31828359242"></a><a name="parmvalue31828359242"></a>“-1”</span>时，Device侧<span id="ph1942410353449"><a name="ph1942410353449"></a><a name="ph1942410353449"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>
