# AscendIndexSQConfig<a name="ZH-CN_TOPIC_0000001456375392"></a>

AscendIndexSQ需要使用对应的AscendIndexSQConfig执行对应资源的初始化。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>inline AscendIndexSQConfig()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexSQConfig的默认构造函数，默认指定的deviceList为0（即指定NPU的第0个<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>作为AscendFaiss执行检索的异构计算平台），采用默认的资源池大小。</p>
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

<a name="table108621239568"></a>
<table><tbody><tr id="row1686242395610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p78621239565"><a name="p78621239565"></a><a name="p78621239565"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p133718164310"><a name="p133718164310"></a><a name="p133718164310"></a>inline AscendIndexSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t  blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row178624230566"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7862192305612"><a name="p7862192305612"></a><a name="p7862192305612"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1686232319567"><a name="p1686232319567"></a><a name="p1686232319567"></a>AscendIndexSQConfig的构造函数，生成AscendIndexSQConfig，此时根据<span class="parmname" id="parmname113412141400"><a name="parmname113412141400"></a><a name="parmname113412141400"></a>“devices”</span>中配置的值设置Device侧<span id="ph126659211576"><a name="ph126659211576"></a><a name="ph126659211576"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row886222375617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2862723165612"><a name="p2862723165612"></a><a name="p2862723165612"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p128621723155611"><a name="p128621723155611"></a><a name="p128621723155611"></a><strong id="b18990511018"><a name="b18990511018"></a><a name="b18990511018"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b64851871304"><a name="b64851871304"></a><a name="b64851871304"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的<span class="parmname" id="parmname1539165725917"><a name="parmname1539165725917"></a><a name="parmname1539165725917"></a>“SQ_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p10918131681313"><a name="p10918131681313"></a><a name="p10918131681313"></a><strong id="b31638114817"><a name="b31638114817"></a><a name="b31638114817"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size，默认值为16384 * 16 = 262144，该值会影响最大可创建Index的数量与检索的性能。</p>
</td>
</tr>
<tr id="row986352311564"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7863823135618"><a name="p7863823135618"></a><a name="p7863823135618"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1286342314562"><a name="p1286342314562"></a><a name="p1286342314562"></a>无</p>
</td>
</tr>
<tr id="row0863723185611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p208632023155619"><a name="p208632023155619"></a><a name="p208632023155619"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1386313230564"><a name="p1386313230564"></a><a name="p1386313230564"></a>无</p>
</td>
</tr>
<tr id="row486382311561"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15863423115615"><a name="p15863423115615"></a><a name="p15863423115615"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul17466113116595"></a><a name="ul17466113116595"></a><ul id="ul17466113116595"><li><span class="parmname" id="parmname58171617204"><a name="parmname58171617204"></a><a name="parmname58171617204"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname15293195861219"><a name="parmname15293195861219"></a><a name="parmname15293195861219"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue1117872319298"><a name="parmvalue1117872319298"></a><a name="parmvalue1117872319298"></a>“-1”</span>时，Device侧<span id="ph18863142385614"><a name="ph18863142385614"></a><a name="ph18863142385614"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li><li><span class="parmname" id="parmname10816837131615"><a name="parmname10816837131615"></a><a name="parmname10816837131615"></a>“blockSize”</span>可配置的值的集合为{16384 * 8，16384 * 16，16384 * 32，16384 * 64}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1735412445711"></a>
<table><tbody><tr id="row19354134175714"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1635417413572"><a name="p1635417413572"></a><a name="p1635417413572"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19780437164418"><a name="p19780437164418"></a><a name="p19780437164418"></a>inline AscendIndexSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t  blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row93540419578"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1635420425713"><a name="p1635420425713"></a><a name="p1635420425713"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13546445718"><a name="p13546445718"></a><a name="p13546445718"></a>AscendIndexSQConfig的构造函数，生成AscendIndexSQConfig，此时根据<span class="parmname" id="parmname387713394011"><a name="parmname387713394011"></a><a name="parmname387713394011"></a>“devices”</span>中配置的值设置Device侧<span id="ph32441233165718"><a name="ph32441233165718"></a><a name="ph32441233165718"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row33541741571"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1135414185711"><a name="p1135414185711"></a><a name="p1135414185711"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1135410418574"><a name="p1135410418574"></a><a name="p1135410418574"></a><strong id="b3303144712017"><a name="b3303144712017"></a><a name="b3303144712017"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p13541545573"><a name="p13541545573"></a><a name="p13541545573"></a><strong id="b197022051409"><a name="b197022051409"></a><a name="b197022051409"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的<span class="parmname" id="parmname439611564019"><a name="parmname439611564019"></a><a name="parmname439611564019"></a>“SQ_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p1035454195716"><a name="p1035454195716"></a><a name="p1035454195716"></a><strong id="b12664181414917"><a name="b12664181414917"></a><a name="b12664181414917"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size，默认值为16384 * 16 = 262144，该值会影响最大可创建Index的数量与检索的性能。</p>
</td>
</tr>
<tr id="row2354104115713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14354184105712"><a name="p14354184105712"></a><a name="p14354184105712"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p43540455713"><a name="p43540455713"></a><a name="p43540455713"></a>无</p>
</td>
</tr>
<tr id="row1354442570"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20354749572"><a name="p20354749572"></a><a name="p20354749572"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p163541948576"><a name="p163541948576"></a><a name="p163541948576"></a>无</p>
</td>
</tr>
<tr id="row2354174135711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p153545418579"><a name="p153545418579"></a><a name="p153545418579"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1931270712"></a><a name="ul1931270712"></a><ul id="ul1931270712"><li><span class="parmname" id="parmname639724115"><a name="parmname639724115"></a><a name="parmname639724115"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname14354144195712"><a name="parmname14354144195712"></a><a name="parmname14354144195712"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue1294113452910"><a name="parmvalue1294113452910"></a><a name="parmvalue1294113452910"></a>“-1”</span>时，Device侧<span id="ph1135454195716"><a name="ph1135454195716"></a><a name="ph1135454195716"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li><li><span class="parmname" id="parmname635415445713"><a name="parmname635415445713"></a><a name="parmname635415445713"></a>“blockSize”</span>可配置的值的集合为{16384 * 8，16384 * 16，16384 * 32，16384 * 64}。</li></ul>
</td>
</tr>
</tbody>
</table>
