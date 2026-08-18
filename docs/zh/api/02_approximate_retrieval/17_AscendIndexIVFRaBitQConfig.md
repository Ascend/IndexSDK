# AscendIndexIVFRaBitQConfig<a name="ZH-CN_TOPIC_0000002544944511"></a>

AscendIndexIVFRaBitQ需要使用对应的AscendIndexIVFRaBitQConfig执行对应资源的初始化。

## 成员介绍<a name="section4211138173219"></a>

<a name="table388535175015"></a>
<table><thead align="left"><tr id="row11881435135015"><th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.1.4.1.1"><p id="p688635145015"><a name="p688635145015"></a><a name="p688635145015"></a>成员</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.1.4.1.2"><p id="p208815352501"><a name="p208815352501"></a><a name="p208815352501"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.1.4.1.3"><p id="p5891535145012"><a name="p5891535145012"></a><a name="p5891535145012"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2890354502"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p1561239193314"><a name="p1561239193314"></a><a name="p1561239193314"></a>useRandomOrthogonalMatrix</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p1589135125017"><a name="p1589135125017"></a><a name="p1589135125017"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p1789103575010"><a name="p1789103575010"></a><a name="p1789103575010"></a>是否使用随机正交矩阵，默认为true。</p>
</td>
</tr>
<tr id="row78912359503"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p10523201623317"><a name="p10523201623317"></a><a name="p10523201623317"></a>needRefine</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p18993510505"><a name="p18993510505"></a><a name="p18993510505"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p689113555010"><a name="p689113555010"></a><a name="p689113555010"></a>是否需要精排，默认为false。</p>
</td>
</tr>
<tr id="row188933513506"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p9321123113316"><a name="p9321123113316"></a><a name="p9321123113316"></a>matrixSeed</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p98919354505"><a name="p98919354505"></a><a name="p98919354505"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p2089235135016"><a name="p2089235135016"></a><a name="p2089235135016"></a>生成随机正交矩阵的随机种子，默认为12345。</p>
</td>
</tr>
<tr id="row192882773310"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p8877163119331"><a name="p8877163119331"></a><a name="p8877163119331"></a>refineAlpha</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p12928122719330"><a name="p12928122719330"></a><a name="p12928122719330"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p3928142753311"><a name="p3928142753311"></a><a name="p3928142753311"></a>精排相关参数，检索时原本需要检索前k个，需要精排则检索前k * refineAlpha个，再从中取topk。</p>
<p id="p169972614290"><a name="p169972614290"></a><a name="p169972614290"></a>该值默认为2，设置得越大，召回率越高，检索效率越低。</p>
</td>
</tr>
</tbody>
</table>

## AscendIndexIVFRaBitQConfig<a name="section6579185362314"></a>

>**说明：**
>AscendIndexIVFRaBitQConfig继承于[AscendIndexIVFConfig](../02_approximate_retrieval/04_AscendIndexIVFConfig.md#ascendindexivfconfig)。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172966052218"><a name="p172966052218"></a><a name="p172966052218"></a>inline AscendIndexIVFRaBitQConfig();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>默认构造函数，默认devices为{0}，使用第0个<span id="ph79732210444"><a name="ph79732210444"></a><a name="ph79732210444"></a>昇腾AI处理器</span>进行计算，默认resource为128MB。</p>
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

<a name="table3725347611"></a>
<table><tbody><tr id="row137251141265"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1372544561"><a name="p1372544561"></a><a name="p1372544561"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4706533988"><a name="p4706533988"></a><a name="p4706533988"></a>inline AscendIndexIVFRaBitQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row0725941369"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p87251143611"><a name="p87251143611"></a><a name="p87251143611"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p099917815338"><a name="p099917815338"></a><a name="p099917815338"></a>AscendIndexIVFRaBitQConfig的构造函数，生成AscendIndexIVFRaBitQConfig，此时根据<span class="parmname" id="parmname18510024575"><a name="parmname18510024575"></a><a name="parmname18510024575"></a>“devices”</span>中配置的值设置Device侧<span id="ph1099958133314"><a name="ph1099958133314"></a><a name="ph1099958133314"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认初始化。</p>
</td>
</tr>
<tr id="row872516411614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p157251441762"><a name="p157251441762"></a><a name="p157251441762"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1172515411612"><a name="p1172515411612"></a><a name="p1172515411612"></a><strong id="b74801235171213"><a name="b74801235171213"></a><a name="b74801235171213"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p999908173313"><a name="p999908173313"></a><a name="p999908173313"></a><strong id="b851894117126"><a name="b851894117126"></a><a name="b851894117126"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname15799101092219"><a name="parmname15799101092219"></a><a name="parmname15799101092219"></a>“IVF_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5591115071213"></a><a name="ul5591115071213"></a><ul id="ul5591115071213"><li><span class="parmname" id="parmname14872125571216"><a name="parmname14872125571216"></a><a name="parmname14872125571216"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname15293195861219"><a name="parmname15293195861219"></a><a name="parmname15293195861219"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue51241072364"><a name="parmvalue51241072364"></a><a name="parmvalue51241072364"></a>“-1”</span>时，Device侧<span id="ph8157732103"><a name="ph8157732103"></a><a name="ph8157732103"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table745471811619"></a>
<table><tbody><tr id="row445418187618"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p145417181561"><a name="p145417181561"></a><a name="p145417181561"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172151146898"><a name="p172151146898"></a><a name="p172151146898"></a>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row845519181169"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p164551418362"><a name="p164551418362"></a><a name="p164551418362"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1445513182614"><a name="p1445513182614"></a><a name="p1445513182614"></a>AscendIndexIVFRaBitQConfig的构造函数，生成AscendIndexIVFRaBitQConfig，此时根据<span class="parmname" id="parmname15821822101518"><a name="parmname15821822101518"></a><a name="parmname15821822101518"></a>“devices”</span>中配置的值设置Device侧<span id="ph663911476576"><a name="ph663911476576"></a><a name="ph663911476576"></a>昇腾AI处理器</span>资源，配置资源池大小并执行默认初始化。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9168241111516"></a><a name="ul9168241111516"></a><ul id="ul9168241111516"><li><span class="parmname" id="parmname975454381516"><a name="parmname975454381516"></a><a name="parmname975454381516"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname8790174512156"><a name="parmname8790174512156"></a><a name="parmname8790174512156"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue8480143663618"><a name="parmvalue8480143663618"></a><a name="parmvalue8480143663618"></a>“-1”</span>时，Device侧<span id="ph245516181062"><a name="ph245516181062"></a><a name="ph245516181062"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1037111614358"></a>
<table><tbody><tr id="row837916103513"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p237101615359"><a name="p237101615359"></a><a name="p237101615359"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1237151683519"><a name="p1237151683519"></a><a name="p1237151683519"></a>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, bool useRandomOrthogonalMatrix_, bool needRefine_, int matrixSeed_, float alpha_, int64_t resourceSize = IVF_DEFAULT_MEM);</p>
</td>
</tr>
<tr id="row173761693511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p19379164358"><a name="p19379164358"></a><a name="p19379164358"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p14374169350"><a name="p14374169350"></a><a name="p14374169350"></a>AscendIndexIVFRaBitQConfig的构造函数，生成AscendIndexIVFRaBitQConfig，此时根据输入参数执行初始化。</p>
</td>
</tr>
<tr id="row163791683511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14371816183518"><a name="p14371816183518"></a><a name="p14371816183518"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p837151615356"><a name="p837151615356"></a><a name="p837151615356"></a><strong id="b163741683510"><a name="b163741683510"></a><a name="b163741683510"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p173258481363"><a name="p173258481363"></a><a name="p173258481363"></a><strong id="b87628480367"><a name="b87628480367"></a><a name="b87628480367"></a>bool useRandomOrthogonalMatrix_</strong>：是否使用随机正交矩阵。</p>
<p id="p131691249163620"><a name="p131691249163620"></a><a name="p131691249163620"></a><strong id="b9570114914362"><a name="b9570114914362"></a><a name="b9570114914362"></a>bool needRefine_</strong>：是否需要精排。</p>
<p id="p1692910498362"><a name="p1692910498362"></a><a name="p1692910498362"></a><strong id="b227917501366"><a name="b227917501366"></a><a name="b227917501366"></a>int matrixSeed_</strong>：生成随机正交矩阵的随机种子。</p>
<p id="p655111502366"><a name="p655111502366"></a><a name="p655111502366"></a><strong id="b208741450183610"><a name="b208741450183610"></a><a name="b208741450183610"></a>float alpha_</strong>：精排相关参数。</p>
<p id="p1237101653516"><a name="p1237101653516"></a><a name="p1237101653516"></a><strong id="b153781620352"><a name="b153781620352"></a><a name="b153781620352"></a>int resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname19371816113519"><a name="parmname19371816113519"></a><a name="parmname19371816113519"></a>“IVF_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row3379162354"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p03711623518"><a name="p03711623518"></a><a name="p03711623518"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2037121633519"><a name="p2037121633519"></a><a name="p2037121633519"></a>无</p>
</td>
</tr>
<tr id="row1237716203511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20371816183513"><a name="p20371816183513"></a><a name="p20371816183513"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8371316133513"><a name="p8371316133513"></a><a name="p8371316133513"></a>无</p>
</td>
</tr>
<tr id="row193701643510"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p10371816193518"><a name="p10371816193518"></a><a name="p10371816193518"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul193714166355"></a><a name="ul193714166355"></a><ul id="ul193714166355"><li><span class="parmname" id="parmname83781610351"><a name="parmname83781610351"></a><a name="parmname83781610351"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname7372016103511"><a name="parmname7372016103511"></a><a name="parmname7372016103511"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue1037716153513"><a name="parmvalue1037716153513"></a><a name="parmvalue1037716153513"></a>“-1”</span>时，Device侧<span id="ph4372160356"><a name="ph4372160356"></a><a name="ph4372160356"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>
