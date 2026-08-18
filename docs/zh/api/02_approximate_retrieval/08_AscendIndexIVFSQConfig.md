# AscendIndexIVFSQConfig<a name="ZH-CN_TOPIC_0000001456375204"></a>

AscendIndexIVFSQ需要使用对应的AscendIndexIVFSQConfig执行对应资源的初始化。

**AscendIndexIVFSQConfig<a name="section015013311183"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexIVFSQConfig();</p>
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

<a name="table19736185071817"></a>
<table><tbody><tr id="row673665061814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p13736350131811"><a name="p13736350131811"></a><a name="p13736350131811"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14172141316118"><a name="p14172141316118"></a><a name="p14172141316118"></a>inline AscendIndexIVFSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</p>
</td>
</tr>
<tr id="row1773645071818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1373675014185"><a name="p1373675014185"></a><a name="p1373675014185"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQConfig的构造函数，生成AscendIndexIVFSQConfig，此时根据<span class="parmname" id="parmname880211685812"><a name="parmname880211685812"></a><a name="parmname880211685812"></a>“devices”</span>中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认的初始化。</p>
</td>
</tr>
<tr id="row37368508181"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p127361650191819"><a name="p127361650191819"></a><a name="p127361650191819"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p127360501187"><a name="p127360501187"></a><a name="p127360501187"></a><strong id="b97961612194319"><a name="b97961612194319"></a><a name="b97961612194319"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b52891554312"><a name="b52891554312"></a><a name="b52891554312"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname5831817164320"><a name="parmname5831817164320"></a><a name="parmname5831817164320"></a>“IVFSQ_DEFAULT_TEMP_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row573613503187"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p11736135011819"><a name="p11736135011819"></a><a name="p11736135011819"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1773695013184"><a name="p1773695013184"></a><a name="p1773695013184"></a>无</p>
</td>
</tr>
<tr id="row173619506182"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1273618502186"><a name="p1273618502186"></a><a name="p1273618502186"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p57361450111811"><a name="p57361450111811"></a><a name="p57361450111811"></a>无</p>
</td>
</tr>
<tr id="row8736205051816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p147360502189"><a name="p147360502189"></a><a name="p147360502189"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul107261247431"></a><a name="ul107261247431"></a><ul id="ul107261247431"><li><span class="parmname" id="parmname9379202674310"><a name="parmname9379202674310"></a><a name="parmname9379202674310"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname238112819435"><a name="parmname238112819435"></a><a name="parmname238112819435"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1056711401917"></a>
<table><tbody><tr id="row1956720419193"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18567124191912"><a name="p18567124191912"></a><a name="p18567124191912"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p656714416197"><a name="p656714416197"></a><a name="p656714416197"></a>inline AscendIndexIVFSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</p>
</td>
</tr>
<tr id="row25671541197"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p105673420197"><a name="p105673420197"></a><a name="p105673420197"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p115671411920"><a name="p115671411920"></a><a name="p115671411920"></a>AscendIndexIVFSQConfig的构造函数，生成AscendIndexIVFSQConfig，此时根据<span class="parmname" id="parmname07048285588"><a name="parmname07048285588"></a><a name="parmname07048285588"></a>“devices”</span>中配置的值设置Device侧<span id="ph185671543194"><a name="ph185671543194"></a><a name="ph185671543194"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认的初始化。</p>
</td>
</tr>
<tr id="row556720415197"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p956718420193"><a name="p956718420193"></a><a name="p956718420193"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p165676418192"><a name="p165676418192"></a><a name="p165676418192"></a><strong id="b2157122616458"><a name="b2157122616458"></a><a name="b2157122616458"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p15678461910"><a name="p15678461910"></a><a name="p15678461910"></a><strong id="b12875133214511"><a name="b12875133214511"></a><a name="b12875133214511"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname3581123624515"><a name="parmname3581123624515"></a><a name="parmname3581123624515"></a>“IVFSQ_DEFAULT_TEMP_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row256744171918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1856714121912"><a name="p1856714121912"></a><a name="p1856714121912"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p17567114101912"><a name="p17567114101912"></a><a name="p17567114101912"></a>无</p>
</td>
</tr>
<tr id="row15567164161919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3567144111913"><a name="p3567144111913"></a><a name="p3567144111913"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p95677416193"><a name="p95677416193"></a><a name="p95677416193"></a>无</p>
</td>
</tr>
<tr id="row13567194101919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11567449197"><a name="p11567449197"></a><a name="p11567449197"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9973203054911"></a><a name="ul9973203054911"></a><ul id="ul9973203054911"><li><span class="parmname" id="parmname2614369494"><a name="parmname2614369494"></a><a name="parmname2614369494"></a>“devices”</span>需要为合法有效不重复的设备ID。</li><li><span class="parmname" id="parmname1549143813490"><a name="parmname1549143813490"></a><a name="parmname1549143813490"></a>“resourceSize”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

**SetDefaultIVFSQConfig<a name="section039015215286"></a>**

<a name="table1185313082915"></a>
<table><tbody><tr id="row18531107298"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p128531018292"><a name="p128531018292"></a><a name="p128531018292"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58281361827"><a name="p58281361827"></a><a name="p58281361827"></a>inline void SetDefaultIVFSQConfig();</p>
</td>
</tr>
<tr id="row198530002911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p8853200102915"><a name="p8853200102915"></a><a name="p8853200102915"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p685316016291"><a name="p685316016291"></a><a name="p685316016291"></a>执行默认的初始化，设置迭代数为16，每个centroids最多设置512个点。</p>
</td>
</tr>
<tr id="row68536022919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p13853804297"><a name="p13853804297"></a><a name="p13853804297"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p158531705291"><a name="p158531705291"></a><a name="p158531705291"></a>无</p>
</td>
</tr>
<tr id="row785340142913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18538016297"><a name="p18538016297"></a><a name="p18538016297"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p198531703295"><a name="p198531703295"></a><a name="p198531703295"></a>无</p>
</td>
</tr>
<tr id="row1885319062919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1853120102913"><a name="p1853120102913"></a><a name="p1853120102913"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p485319062912"><a name="p485319062912"></a><a name="p485319062912"></a>无</p>
</td>
</tr>
<tr id="row188538018295"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p78535032910"><a name="p78535032910"></a><a name="p78535032910"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p128538010296"><a name="p128538010296"></a><a name="p128538010296"></a>无</p>
</td>
</tr>
</tbody>
</table>
