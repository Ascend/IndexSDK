# AscendIndexIVFSPConfig<a id="ZH-CN_TOPIC_0000001635696057"></a>

AscendIndexIVFSP需要使用对应的AscendIndexIVFSPConfig执行对应资源的初始化。

**公共参数<a name="section17656114673616"></a>**

|参数名|数据类型|参数说明|
|--|--|--|
|handleBatch|int|检索时每次下发计算的候选桶数量，默认值为64。|
|nprobe|int|检索时总的候选桶数量，默认值为64。|
|searchListSize|int|检索时每次下发计算的每个桶的最大样本数量，默认值为32768。若桶太大，程序会自动根据searchListSize将桶拆成多次算子下发计算距离。|

**接口说明<a name="section74781713710"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>inline AscendIndexIVFSPConfig();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>默认构造函数，默认devices为{0}，使用第0个<span id="ph79732210444"><a name="ph79732210444"></a><a name="ph79732210444"></a>昇腾AI处理器</span>进行计算，默认resources为128MB。</p>
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

<a name="table121971648373"></a>
<table><tbody><tr id="row13197134820716"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p51977481976"><a name="p51977481976"></a><a name="p51977481976"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161970481773"><a name="p161970481773"></a><a name="p161970481773"></a>inline explicit AscendIndexIVFSPConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</p>
</td>
</tr>
<tr id="row141971748972"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1419717481876"><a name="p1419717481876"></a><a name="p1419717481876"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSPConfig构造函数，生成AscendIndexIVFSPConfig，指定Device侧设备ID和资源池大小。</p>
</td>
</tr>
<tr id="row191973486716"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1019719485712"><a name="p1019719485712"></a><a name="p1019719485712"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p59096365498"><a name="p59096365498"></a><a name="p59096365498"></a><strong id="b2851659184912"><a name="b2851659184912"></a><a name="b2851659184912"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p9909173617494"><a name="p9909173617494"></a><a name="p9909173617494"></a><strong id="b61011425012"><a name="b61011425012"></a><a name="b61011425012"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmvalue" id="parmvalue134388571724"><a name="parmvalue134388571724"></a><a name="parmvalue134388571724"></a>“IVF_SP_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p1990912367496"><a name="p1990912367496"></a><a name="p1990912367496"></a><strong id="b19708106145014"><a name="b19708106145014"></a><a name="b19708106145014"></a>uint32_t blockSize</strong>：预置的内存块大小，单位为Byte。默认参数为头文件中的<span class="parmvalue" id="parmvalue380144918213"><a name="parmvalue380144918213"></a><a name="parmvalue380144918213"></a>“DEFAULT_BLOCK_SIZE”</span>。</p>
</td>
</tr>
<tr id="row61979480720"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p91979481711"><a name="p91979481711"></a><a name="p91979481711"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1719734811717"><a name="p1719734811717"></a><a name="p1719734811717"></a>无</p>
</td>
</tr>
<tr id="row1919711482718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p181974481777"><a name="p181974481777"></a><a name="p181974481777"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p919717486712"><a name="p919717486712"></a><a name="p919717486712"></a>无</p>
</td>
</tr>
<tr id="row1719719481072"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p181973481672"><a name="p181973481672"></a><a name="p181973481672"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20548163415505"></a><a name="ul20548163415505"></a><ul id="ul20548163415505"><li><span class="parmname" id="parmname956713818220"><a name="parmname956713818220"></a><a name="parmname956713818220"></a>“devices”</span>需要为合法有效不重复的设备ID，当前仅支持1个NPU设备。</li><li><span class="parmname" id="parmname798183519217"><a name="parmname798183519217"></a><a name="parmname798183519217"></a>“resources”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table56061252785"></a>
<table><tbody><tr id="row6606552282"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p06062521781"><a name="p06062521781"></a><a name="p06062521781"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p108121410115113"><a name="p108121410115113"></a><a name="p108121410115113"></a>inline explicit AscendIndexIVFSPConfig(std::vector&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</p>
</td>
</tr>
<tr id="row156061352486"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p460610521381"><a name="p460610521381"></a><a name="p460610521381"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8606152685"><a name="p8606152685"></a><a name="p8606152685"></a>AscendIndexIVFSPConfig构造函数，生成AscendIndexIVFSPConfig，指定Device侧设备ID和资源池大小。</p>
</td>
</tr>
<tr id="row146067521289"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p46064521488"><a name="p46064521488"></a><a name="p46064521488"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p15921231125111"><a name="p15921231125111"></a><a name="p15921231125111"></a><strong id="b18248195515616"><a name="b18248195515616"></a><a name="b18248195515616"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p09211631145120"><a name="p09211631145120"></a><a name="p09211631145120"></a><strong id="b11958155810618"><a name="b11958155810618"></a><a name="b11958155810618"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmvalue" id="parmvalue3723441276"><a name="parmvalue3723441276"></a><a name="parmvalue3723441276"></a>“IVF_SP_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p139219313517"><a name="p139219313517"></a><a name="p139219313517"></a><strong id="b159531621471"><a name="b159531621471"></a><a name="b159531621471"></a>uint32_t blockSize</strong>：预置的内存块大小，单位为Byte。默认参数为头文件中的<span class="parmvalue" id="parmvalue232315188714"><a name="parmvalue232315188714"></a><a name="parmvalue232315188714"></a>“DEFAULT_BLOCK_SIZE”</span>。</p>
</td>
</tr>
<tr id="row1160718521816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9607252586"><a name="p9607252586"></a><a name="p9607252586"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p106076525815"><a name="p106076525815"></a><a name="p106076525815"></a>无</p>
</td>
</tr>
<tr id="row36075522089"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6607125210819"><a name="p6607125210819"></a><a name="p6607125210819"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p26071521080"><a name="p26071521080"></a><a name="p26071521080"></a>无</p>
</td>
</tr>
<tr id="row8607152585"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p160714523812"><a name="p160714523812"></a><a name="p160714523812"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul899544135119"></a><a name="ul899544135119"></a><ul id="ul899544135119"><li><span class="parmname" id="parmname18598358477"><a name="parmname18598358477"></a><a name="parmname18598358477"></a>“devices”</span>需要为合法有效不重复的设备ID，当前仅支持1个NPU设备。</li><li><span class="parmname" id="parmname1019313551472"><a name="parmname1019313551472"></a><a name="parmname1019313551472"></a>“resources”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>
