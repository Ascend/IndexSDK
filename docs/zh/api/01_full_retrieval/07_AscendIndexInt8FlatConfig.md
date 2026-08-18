# AscendIndexInt8FlatConfig<a name="ZH-CN_TOPIC_0000001456535040"></a>

AscendIndexInt8Flat需要使用对应的AscendIndexInt8FlatConfig执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|dIndexMode|Int8IndexMode|配置Index int8检索模式。|
|dBlockSize|uint32_t|配置Device侧的blockSize。|

**接口说明<a name="section136272015172914"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexInt8FlatConfig(uint32_t blockSize =BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，配置Device侧blockSize，配置int8的检索模式。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b12302102216503"><a name="b12302102216503"></a><a name="b12302102216503"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname241445820209"><a name="parmname241445820209"></a><a name="parmname241445820209"></a>“BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
<p id="p196421349716"><a name="p196421349716"></a><a name="p196421349716"></a><strong id="b49637218593"><a name="b49637218593"></a><a name="b49637218593"></a>Int8IndexMode indexMode</strong>：配置Index int8检索模式。默认值为<strong id="b13424175612711"><a name="b13424175612711"></a><a name="b13424175612711"></a>DEFAULT_MODE</strong>。</p>
<a name="ul112020291075"></a><a name="ul112020291075"></a><ul id="ul112020291075"><li><strong id="b8674833612"><a name="b8674833612"></a><a name="b8674833612"></a>DEFAULT_MODE</strong>模式，默认模式。</li><li><strong id="b197655512712"><a name="b197655512712"></a><a name="b197655512712"></a>PIPE_SEARCH_MODE</strong>模式，该模式针对batch大于或等于<strong id="b925325991216"><a name="b925325991216"></a><a name="b925325991216"></a>128</strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB<strong id="b142221114490"><a name="b142221114490"></a><a name="b142221114490"></a>。</strong></li><li><strong id="b193311527712"><a name="b193311527712"></a><a name="b193311527712"></a>WITHOUT_NORM_MODE</strong>模式，暂时不支持本模式。</li></ul>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul72731322134614"></a><a name="ul72731322134614"></a><ul id="ul72731322134614"><li><span class="parmname" id="parmname567764215019"><a name="parmname567764215019"></a><a name="parmname567764215019"></a>“blockSize”</span>可配置的值的集合为{16384， 32768， 65536， 131072， 262144}</li></ul>
<a name="ul17184174914612"></a><a name="ul17184174914612"></a><ul id="ul17184174914612"><li><strong id="b535836183515"><a name="b535836183515"></a><a name="b535836183515"></a><span class="parmname" id="parmname346219387426"><a name="parmname346219387426"></a><a name="parmname346219387426"></a>“indexMode”</span></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1258103643012"></a>
<table><tbody><tr id="row95803619306"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p15853617303"><a name="p15853617303"></a><a name="p15853617303"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p558236163018"><a name="p558236163018"></a><a name="p558236163018"></a>AscendIndexInt8FlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</p>
</td>
</tr>
<tr id="row10580363301"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1058436123019"><a name="p1058436123019"></a><a name="p1058436123019"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p459183617305"><a name="p459183617305"></a><a name="p459183617305"></a>AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，此时根据Devices中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源，配置资源池大小。配置Device侧blockSize，配置int8的检索模式。</p>
</td>
</tr>
<tr id="row9592036113014"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p145953643012"><a name="p145953643012"></a><a name="p145953643012"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p165943663010"><a name="p165943663010"></a><a name="p165943663010"></a><strong id="b56134644720"><a name="b56134644720"></a><a name="b56134644720"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b10147012114711"><a name="b10147012114711"></a><a name="b10147012114711"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname143161514154715"><a name="parmname143161514154715"></a><a name="parmname143161514154715"></a>“INT8_FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置1024MB。</p>
<p id="p266920816252"><a name="p266920816252"></a><a name="p266920816252"></a><strong id="b14261165535013"><a name="b14261165535013"></a><a name="b14261165535013"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname13591036163017"><a name="parmname13591036163017"></a><a name="parmname13591036163017"></a>“BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
<p id="p920025214811"><a name="p920025214811"></a><a name="p920025214811"></a><strong id="b1359036103010"><a name="b1359036103010"></a><a name="b1359036103010"></a>Int8IndexMode indexMode</strong>：配置Index int8检索模式。默认值为<strong id="b109732583816"><a name="b109732583816"></a><a name="b109732583816"></a>DEFAULT_MODE</strong>。</p>
<a name="ul19317031497"></a><a name="ul19317031497"></a><ul id="ul19317031497"><li><strong id="b1220019526818"><a name="b1220019526818"></a><a name="b1220019526818"></a>DEFAULT_MODE</strong>模式，默认模式。</li><li><strong id="b162925151916"><a name="b162925151916"></a><a name="b162925151916"></a>PIPE_SEARCH_MODE</strong>模式，该模式针对batch大于或等于<strong id="b459736143011"><a name="b459736143011"></a><a name="b459736143011"></a>128</strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB。</li><li><strong id="b1772318287911"><a name="b1772318287911"></a><a name="b1772318287911"></a>WITHOUT_NORM_MODE</strong>模式，暂时不支持本模式。</li></ul>
</td>
</tr>
<tr id="row859836143020"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p359536193015"><a name="p359536193015"></a><a name="p359536193015"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p659336153018"><a name="p659336153018"></a><a name="p659336153018"></a>无</p>
</td>
</tr>
<tr id="row105953623010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p759536193011"><a name="p759536193011"></a><a name="p759536193011"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p195910363302"><a name="p195910363302"></a><a name="p195910363302"></a>无</p>
</td>
</tr>
<tr id="row1259173611301"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p1059173683011"><a name="p1059173683011"></a><a name="p1059173683011"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul07292414471"></a><a name="ul07292414471"></a><ul id="ul07292414471"><li><span class="parmname" id="parmname63243288473"><a name="parmname63243288473"></a><a name="parmname63243288473"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname9781626112312"><a name="parmname9781626112312"></a><a name="parmname9781626112312"></a>“resourceSize”</span>配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。当batch大于等于96时，为提升算法性能，建议<span class="parmname" id="parmname8997192411519"><a name="parmname8997192411519"></a><a name="parmname8997192411519"></a>“resourceSize”</span>设置为大于等于2 * 1024MB。</li><li><span class="parmname" id="parmname2376185113505"><a name="parmname2376185113505"></a><a name="parmname2376185113505"></a>“blockSize”</span>可配置的值的集合为{16384， 32768， 65536， 131072， 262144}</li><li><strong id="b259143623014"><a name="b259143623014"></a><a name="b259143623014"></a><span class="parmname" id="parmname759936143015"><a name="parmname759936143015"></a><a name="parmname759936143015"></a>“indexMode”</span></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table8629135217302"></a>
<table><tbody><tr id="row6630115223010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p126301152183015"><a name="p126301152183015"></a><a name="p126301152183015"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1363005243010"><a name="p1363005243010"></a><a name="p1363005243010"></a>AscendIndexInt8FlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)</p>
</td>
</tr>
<tr id="row1630175243019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p16630752193012"><a name="p16630752193012"></a><a name="p16630752193012"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p263012528307"><a name="p263012528307"></a><a name="p263012528307"></a>AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，此时根据<span class="parmname" id="parmname1810443014014"><a name="parmname1810443014014"></a><a name="parmname1810443014014"></a>“devices”</span>中配置的值设置Device侧<span id="ph763075283015"><a name="ph763075283015"></a><a name="ph763075283015"></a>昇腾AI处理器</span>资源，配置资源池大小。配置Device侧blockSize，配置int8的检索模式。</p>
</td>
</tr>
<tr id="row363011522300"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p196301452183019"><a name="p196301452183019"></a><a name="p196301452183019"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p4630205293012"><a name="p4630205293012"></a><a name="p4630205293012"></a><strong id="b658511585471"><a name="b658511585471"></a><a name="b658511585471"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p2063075223014"><a name="p2063075223014"></a><a name="p2063075223014"></a><strong id="b1075315534819"><a name="b1075315534819"></a><a name="b1075315534819"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname13771892480"><a name="parmname13771892480"></a><a name="parmname13771892480"></a>“INT8_FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置1024MB。</p>
<p id="p5467172216269"><a name="p5467172216269"></a><a name="p5467172216269"></a><strong id="b134751453101011"><a name="b134751453101011"></a><a name="b134751453101011"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname4630052103015"><a name="parmname4630052103015"></a><a name="parmname4630052103015"></a>“BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
<p id="p38771541295"><a name="p38771541295"></a><a name="p38771541295"></a><strong id="b6630352153010"><a name="b6630352153010"></a><a name="b6630352153010"></a>Int8IndexMode indexMode</strong>：配置Index int8检索模式。默认值为<span class="parmvalue" id="parmvalue651015459212"><a name="parmvalue651015459212"></a><a name="parmvalue651015459212"></a>“DEFAULT_MODE”</span>。</p>
<a name="ul57151159696"></a><a name="ul57151159696"></a><ul id="ul57151159696"><li><strong id="b963075218304"><a name="b963075218304"></a><a name="b963075218304"></a>DEFAULT_MODE</strong>模式，默认模式。</li><li><strong id="b1936318711107"><a name="b1936318711107"></a><a name="b1936318711107"></a>PIPE_SEARCH_MODE</strong>模式，该模式针对batch大于或等于<strong id="b663015210304"><a name="b663015210304"></a><a name="b663015210304"></a>128</strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB<strong id="b1763015217308"><a name="b1763015217308"></a><a name="b1763015217308"></a>。</strong></li><li><strong id="b11411131411019"><a name="b11411131411019"></a><a name="b11411131411019"></a>WITHOUT_NORM_MODE</strong>模式，暂时不支持本模式。</li></ul>
</td>
</tr>
<tr id="row20630135293014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p116301652203017"><a name="p116301652203017"></a><a name="p116301652203017"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15630952103017"><a name="p15630952103017"></a><a name="p15630952103017"></a>无</p>
</td>
</tr>
<tr id="row2630852173011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20630185213016"><a name="p20630185213016"></a><a name="p20630185213016"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p176301752133010"><a name="p176301752133010"></a><a name="p176301752133010"></a>无</p>
</td>
</tr>
<tr id="row156301552103017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9630105253015"><a name="p9630105253015"></a><a name="p9630105253015"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul115901014144816"></a><a name="ul115901014144816"></a><ul id="ul115901014144816"><li><span class="parmname" id="parmname34910164489"><a name="parmname34910164489"></a><a name="parmname34910164489"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname13630135263011"><a name="parmname13630135263011"></a><a name="parmname13630135263011"></a>“resourceSize”</span>配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。当batch大于等于96时，为提升算法性能，建议<span class="parmname" id="parmname96301052143011"><a name="parmname96301052143011"></a><a name="parmname96301052143011"></a>“resourceSize”</span>设置为大于等于2 * 1024MB。</li><li><span class="parmname" id="parmname3456949185114"><a name="parmname3456949185114"></a><a name="parmname3456949185114"></a>“blockSize”</span>可配置的值的集合为{16384， 32768， 65536， 131072， 262144}。</li><li><strong id="b151606422049"><a name="b151606422049"></a><a name="b151606422049"></a><span class="parmname" id="parmname206301752163017"><a name="parmname206301752163017"></a><a name="parmname206301752163017"></a>“indexMode”</span></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</li></ul>
</td>
</tr>
</tbody>
</table>
