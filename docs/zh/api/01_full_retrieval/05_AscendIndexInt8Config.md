# AscendIndexInt8Config<a id="ZH-CN_TOPIC_0000001456854968"></a>

AscendIndexInt8需要使用对应的AscendIndexInt8Config执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|deviceList|std::vector\<int>|Device侧设备ID。|
|resourceSize|int64_t|设备侧预置的内存池大小，单位为字节。|

**接口说明<a name="section135441937164218"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexInt8Config()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8Config的默认构造函数，默认指定的deviceList为0（即指定NPU的第0个<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>作为AscendFaiss执行检索的异构计算平台），采用默认的资源池大小。</p>
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

<a name="table012165162914"></a>
<table><tbody><tr id="row71210582913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p31219512297"><a name="p31219512297"></a><a name="p31219512297"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132201932191815"><a name="p132201932191815"></a><a name="p132201932191815"></a>AscendIndexInt8Config(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row212554294"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1012185102912"><a name="p1012185102912"></a><a name="p1012185102912"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p71214510295"><a name="p71214510295"></a><a name="p71214510295"></a>AscendIndexInt8Config的构造函数，生成AscendIndexInt8Config，此时根据<span class="parmname" id="parmname131034538596"><a name="parmname131034538596"></a><a name="parmname131034538596"></a>“devices”</span>中配置的值设置Device侧<span id="ph151213517298"><a name="ph151213517298"></a><a name="ph151213517298"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row101210562912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1612856291"><a name="p1612856291"></a><a name="p1612856291"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p13185103613195"><a name="p13185103613195"></a><a name="p13185103613195"></a><strong id="b17802134542320"><a name="b17802134542320"></a><a name="b17802134542320"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b139931451162315"><a name="b139931451162315"></a><a name="b139931451162315"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname19157165610231"><a name="parmname19157165610231"></a><a name="parmname19157165610231"></a>“INDEX_INT8_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row201311582910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7139542914"><a name="p7139542914"></a><a name="p7139542914"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p913145182911"><a name="p913145182911"></a><a name="p913145182911"></a>无</p>
</td>
</tr>
<tr id="row61312510298"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6137572910"><a name="p6137572910"></a><a name="p6137572910"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7131256297"><a name="p7131256297"></a><a name="p7131256297"></a>无</p>
</td>
</tr>
<tr id="row81316519296"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p41316552918"><a name="p41316552918"></a><a name="p41316552918"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul12323102516242"></a><a name="ul12323102516242"></a><ul id="ul12323102516242"><li><span class="parmname" id="parmname1876622712243"><a name="parmname1876622712243"></a><a name="parmname1876622712243"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname460517421216"><a name="parmname460517421216"></a><a name="parmname460517421216"></a>“resources”</span>配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table9202719152913"></a>
<table><tbody><tr id="row620221922910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p152021219162918"><a name="p152021219162918"></a><a name="p152021219162918"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p620271952915"><a name="p620271952915"></a><a name="p620271952915"></a>AscendIndexInt8Config(std::vector&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row720217191294"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7202151915293"><a name="p7202151915293"></a><a name="p7202151915293"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8202419152912"><a name="p8202419152912"></a><a name="p8202419152912"></a>AscendIndexInt8Config的构造函数，生成AscendIndexInt8Config，此时根据<span class="parmname" id="parmname55384581598"><a name="parmname55384581598"></a><a name="parmname55384581598"></a>“devices”</span>中配置的值设置Device侧<span id="ph182026195293"><a name="ph182026195293"></a><a name="ph182026195293"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row7202101919297"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p02021219172918"><a name="p02021219172918"></a><a name="p02021219172918"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p17202319192913"><a name="p17202319192913"></a><a name="p17202319192913"></a><strong id="b3189458112410"><a name="b3189458112410"></a><a name="b3189458112410"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p32021619132913"><a name="p32021619132913"></a><a name="p32021619132913"></a><strong id="b18933954250"><a name="b18933954250"></a><a name="b18933954250"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname514412106256"><a name="parmname514412106256"></a><a name="parmname514412106256"></a>“INDEX_INT8_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row22021519142913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p202027199296"><a name="p202027199296"></a><a name="p202027199296"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p202021619112918"><a name="p202021619112918"></a><a name="p202021619112918"></a>无</p>
</td>
</tr>
<tr id="row120218193297"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2202171902914"><a name="p2202171902914"></a><a name="p2202171902914"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1720211192299"><a name="p1720211192299"></a><a name="p1720211192299"></a>无</p>
</td>
</tr>
<tr id="row520291962913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1720213191294"><a name="p1720213191294"></a><a name="p1720213191294"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul373416141258"></a><a name="ul373416141258"></a><ul id="ul373416141258"><li><span class="parmname" id="parmname67839163255"><a name="parmname67839163255"></a><a name="parmname67839163255"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname112023193297"><a name="parmname112023193297"></a><a name="parmname112023193297"></a>“resources”</span>配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>
