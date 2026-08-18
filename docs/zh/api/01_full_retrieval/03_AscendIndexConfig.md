
# AscendIndexConfig<a id="ZH-CN_TOPIC_0000001506414705"></a>

AscendIndex需要使用对应的AscendIndexConfig执行对应资源的初始化，AscendIndexConfig中需要配置执行检索过程中的硬件资源和内存池大小等。

> [!NOTE]
>内存池大小单位为**Byte**，此参数用于指定Device侧预留的内存池大小。内存池用于存储昇腾硬件上进行距离计算的结果，底库规模较大时，建议预留更大的内存池大小。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|deviceList|std::vector\<int>|Device侧设备ID。|
|resourceSize|int64_t|Device侧内存池大小，单位为字节，默认参数为头文件中的**INDEX_DEFAULT_MEM**。|
|slim|bool|AscendIndexConfig成员变量，是否动态增加内存。|
|filterable|bool|AscendIndexConfig成员变量，是否按照id进行过滤。|
|dBlockSize|uint32_t|配置Device侧的blockSize。|

**接口说明<a name="section1197816229504"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexConfig()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexConfig默认构造函数，默认指定的deviceList为0（即指定NPU的第0个<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>作为AscendFaiss执行检索的异构计算平台），默认的资源池大小为32MB（32*1024*1024字节）。</p>
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

<a name="table0786126165110"></a>
<table><tbody><tr id="row2787106115110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16787196145110"><a name="p16787196145110"></a><a name="p16787196145110"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1370313297588"><a name="p1370313297588"></a><a name="p1370313297588"></a>AscendIndexConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row378710616519"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p278776125114"><a name="p278776125114"></a><a name="p278776125114"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p978718675120"><a name="p978718675120"></a><a name="p978718675120"></a>AscendIndexConfig的构造函数，生成AscendIndexConfig，此时根据<span class="parmname" id="parmname16341331367"><a name="parmname16341331367"></a><a name="parmname16341331367"></a>“devices”</span>中配置的值设置Device侧<span id="ph8787176125114"><a name="ph8787176125114"></a><a name="ph8787176125114"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row167879675117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p6787136185119"><a name="p6787136185119"></a><a name="p6787136185119"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p578796165119"><a name="p578796165119"></a><a name="p578796165119"></a><strong id="b7745577018"><a name="b7745577018"></a><a name="b7745577018"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b120518131001"><a name="b120518131001"></a><a name="b120518131001"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname1331442416116"><a name="parmname1331442416116"></a><a name="parmname1331442416116"></a>“INDEX_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p45336291716"><a name="p45336291716"></a><a name="p45336291716"></a><strong id="b138053362110"><a name="b138053362110"></a><a name="b138053362110"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname12815818101613"><a name="parmname12815818101613"></a><a name="parmname12815818101613"></a>“DEFAULT_BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
</td>
</tr>
<tr id="row6787166165117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p77879695115"><a name="p77879695115"></a><a name="p77879695115"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1278716612512"><a name="p1278716612512"></a><a name="p1278716612512"></a>无</p>
</td>
</tr>
<tr id="row1787116135119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1278711610519"><a name="p1278711610519"></a><a name="p1278711610519"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p4787166125116"><a name="p4787166125116"></a><a name="p4787166125116"></a>无</p>
</td>
</tr>
<tr id="row87873611514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2787126195113"><a name="p2787126195113"></a><a name="p2787126195113"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5609290011"></a><a name="ul5609290011"></a><ul id="ul5609290011"><li><span class="parmname" id="parmname869315371603"><a name="parmname869315371603"></a><a name="parmname869315371603"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname460517421216"><a name="parmname460517421216"></a><a name="parmname460517421216"></a>“resources”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table23967285518"></a>
<table><tbody><tr id="row17396102845111"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.1.1"><p id="p83961128145117"><a name="p83961128145117"></a><a name="p83961128145117"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.1.1 "><p id="p894518112592"><a name="p894518112592"></a><a name="p894518112592"></a>AscendIndexConfig(std::vector&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row03962028165110"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.2.1"><p id="p439632811517"><a name="p439632811517"></a><a name="p439632811517"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.2.1 "><p id="p163968283515"><a name="p163968283515"></a><a name="p163968283515"></a>AscendIndexConfig的构造函数，生成AscendIndexConfig，此时根据<span class="parmname" id="parmname1778119110614"><a name="parmname1778119110614"></a><a name="parmname1778119110614"></a>“devices”</span>中配置的值设置Device侧<span id="ph1039613285519"><a name="ph1039613285519"></a><a name="ph1039613285519"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row2396172875119"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.3.1"><p id="p12396228175120"><a name="p12396228175120"></a><a name="p12396228175120"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.3.1 "><p id="p43961828195110"><a name="p43961828195110"></a><a name="p43961828195110"></a><strong id="b183551822226"><a name="b183551822226"></a><a name="b183551822226"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p839622813512"><a name="p839622813512"></a><a name="p839622813512"></a><strong id="b1160625821"><a name="b1160625821"></a><a name="b1160625821"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname14396928125116"><a name="parmname14396928125116"></a><a name="parmname14396928125116"></a>“INDEX_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p192819288016"><a name="p192819288016"></a><a name="p192819288016"></a><strong id="b07647394111"><a name="b07647394111"></a><a name="b07647394111"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname9394163317166"><a name="parmname9394163317166"></a><a name="parmname9394163317166"></a>“DEFAULT_BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
</td>
</tr>
<tr id="row9396182816510"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.4.1"><p id="p18396152865113"><a name="p18396152865113"></a><a name="p18396152865113"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.4.1 "><p id="p1539616283512"><a name="p1539616283512"></a><a name="p1539616283512"></a>无</p>
</td>
</tr>
<tr id="row1439642825118"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.5.1"><p id="p2396928115112"><a name="p2396928115112"></a><a name="p2396928115112"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.5.1 "><p id="p1939617284511"><a name="p1939617284511"></a><a name="p1939617284511"></a>无</p>
</td>
</tr>
<tr id="row12396182811516"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.6.1"><p id="p1439682811514"><a name="p1439682811514"></a><a name="p1439682811514"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.6.1 "><a name="ul133383413220"></a><a name="ul133383413220"></a><ul id="ul133383413220"><li><span class="parmname" id="parmname1593203817210"><a name="parmname1593203817210"></a><a name="parmname1593203817210"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname11396162818517"><a name="parmname11396162818517"></a><a name="parmname11396162818517"></a>“resources”</span>配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>
