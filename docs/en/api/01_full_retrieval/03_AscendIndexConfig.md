# `AscendIndexConfig`<a id="en-us_TOPIC_0000001506414705"></a>

AscendIndex requires the corresponding `AscendIndexConfig` to initialize the relevant resources. `AscendIndexConfig` must configure the hardware resources and memory pool size used during retrieval.

> [!NOTE]
> The memory pool size unit is **Byte**. This parameter specifies the size of the preallocated memory pool on the device side. The memory pool stores the results of distance calculations on Ascend hardware. When the base library is large, you are advised to reserve a larger memory pool.

**Members<a name="section1372191465013"></a>**

| Member | Type | Description |
| ------ | ---- | ----------- |
| deviceList | std::vector\<int> | Device-side device IDs. |
| resourceSize | int64_t | Device-side memory pool size, in bytes. The default parameter is `INDEX_DEFAULT_MEM` in the header file. |
| slim | bool | Member variable of `AscendIndexConfig`. Indicates whether to increase memory dynamically. |
| filterable | bool | Member variable of `AscendIndexConfig`. Indicates whether to filter by ID. |
| dBlockSize | uint32_t | Device-side block size configuration. |

**API Description<a name="section1197816229504"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>AscendIndexConfig()</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Default constructor of <code>AscendIndexConfig</code>. The default <span class="parmname" id="parmname16341331367"><a name="parmname16341331367"></a><a name="parmname16341331367"></a>deviceList</span> is <code>0</code>, which means that the Ascend AI Processor with ID <code>0</code> on the NPU is used as the heterogeneous computing platform for AscendFaiss retrieval. The default resource-pool size is <code>32 MB</code> (<code>32*1024*1024</code> bytes).</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table0786126165110"></a>
<table><tbody><tr id="row2787106115110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16787196145110"><a name="p16787196145110"></a><a name="p16787196145110"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1370313297588"><a name="p1370313297588"></a><a name="p1370313297588"></a><code>AscendIndexConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></p>
</td>
</tr>
<tr id="row378710616519"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p278776125114"><a name="p278776125114"></a><a name="p278776125114"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p978718675120"><a name="p978718675120"></a><a name="p978718675120"></a>Constructor of <code>AscendIndexConfig</code>. It creates an <code>AscendIndexConfig</code> and sets device-side Ascend AI Processor resources according to the values configured in <span class="parmname" id="parmname16341331367"><a name="parmname16341331367"></a><a name="parmname16341331367"></a><code>devices</code></span>, while also configuring the resource-pool size.</p>
</td>
</tr>
<tr id="row167879675117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p6787136185119"><a name="p6787136185119"></a><a name="p6787136185119"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p578796165119"><a name="p578796165119"></a><a name="p578796165119"></a><strong id="b7745577018"><a name="b7745577018"></a><a name="b7745577018"></a>std::initializer_list&lt;int&gt; devices</strong>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b120518131001"><a name="b120518131001"></a><a name="b120518131001"></a>int64_t resources</strong>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <span class="parmname" id="parmname1331442416116"><a name="parmname1331442416116"></a><a name="parmname1331442416116"></a><code>INDEX_DEFAULT_MEM</code></span> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</p>
<p id="p45336291716"><a name="p45336291716"></a><a name="p45336291716"></a><strong id="b138053362110"><a name="b138053362110"></a><a name="b138053362110"></a>uint32_t blockSize</strong>: Device-side block size configuration. It constrains the amount of data processed in one tik operator call and the size of vectors stored in each partition of the base-library shard. The default value of <span class="parmname" id="parmname12815818101613"><a name="parmname12815818101613"></a><a name="parmname12815818101613"></a><code>DEFAULT_BLOCK_SIZE</code></span> is <code>16384 * 16 = 262144</code>.</p>
</td>
</tr>
<tr id="row6787166165117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p77879695115"><a name="p77879695115"></a><a name="p77879695115"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1278716612512"><a name="p1278716612512"></a><a name="p1278716612512"></a>None</p>
</td>
</tr>
<tr id="row1787116135119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1278711610519"><a name="p1278711610519"></a><a name="p1278711610519"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p4787166125116"><a name="p4787166125116"></a><a name="p4787166125116"></a>None</p>
</td>
</tr>
<tr id="row87873611514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2787126195113"><a name="p2787126195113"></a><a name="p2787126195113"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5609290011"></a><a name="ul5609290011"></a><ul id="ul5609290011"><li><span class="parmname" id="parmname869315371603"><a name="parmname869315371603"></a><a name="parmname869315371603"></a><code>devices</code></span> must be valid, unique device IDs. The maximum number is 64.</li><li>The configured value of <span class="parmname" id="parmname460517421216"><a name="parmname460517421216"></a><a name="parmname460517421216"></a>resources</span> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table23967285518"></a>
<table><tbody><tr id="row17396102845111"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.1.1"><p id="p83961128145117"><a name="p83961128145117"></a><a name="p83961128145117"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.1.1 "><p id="p894518112592"><a name="p894518112592"></a><a name="p894518112592"></a><code>AscendIndexConfig(std::vector&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</code></p>
</td>
</tr>
<tr id="row03962028165110"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.2.1"><p id="p439632811517"><a name="p439632811517"></a><a name="p439632811517"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.2.1 "><p id="p163968283515"><a name="p163968283515"></a><a name="p163968283515"></a>Constructor of <code>AscendIndexConfig</code>. It creates an <code>AscendIndexConfig</code> and sets device-side Ascend AI Processor resources according to the values configured in <span class="parmname" id="parmname1778119110614"><a name="parmname1778119110614"></a><a name="parmname1778119110614"></a><code>devices</code></span>, while also configuring the resource-pool size.</p>
</td>
</tr>
<tr id="row2396172875119"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.3.1"><p id="p12396228175120"><a name="p12396228175120"></a><a name="p12396228175120"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.3.1 "><p id="p43961828195110"><a name="p43961828195110"></a><a name="p43961828195110"></a><strong id="b183551822226"><a name="b183551822226"></a><a name="b183551822226"></a>std::vector&lt;int&gt; devices</strong>: Device-side device IDs.</p>
<p id="p839622813512"><a name="p839622813512"></a><a name="p839622813512"></a><strong id="b1160625821"><a name="b1160625821"></a><a name="b1160625821"></a>int64_t resources</strong>: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <span class="parmname" id="parmname14396928125116"><a name="parmname14396928125116"></a><a name="parmname14396928125116"></a><code>INDEX_DEFAULT_MEM</code></span> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</p>
<p id="p192819288016"><a name="p192819288016"></a><a name="p192819288016"></a><strong id="b07647394111"><a name="b07647394111"></a><a name="b07647394111"></a>uint32_t blockSize</strong>: Device-side block size configuration. It constrains the amount of data processed in one tik operator call and the size of vectors stored in each partition of the base-library shard. The default value of <span class="parmname" id="parmname9394163317166"><a name="parmname9394163317166"></a><a name="parmname9394163317166"></a><code>DEFAULT_BLOCK_SIZE</code></span> is <code>16384 * 16 = 262144</code>.</p>
</td>
</tr>
<tr id="row9396182816510"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.4.1"><p id="p18396152865113"><a name="p18396152865113"></a><a name="p18396152865113"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.4.1 "><p id="p1539616283512"><a name="p1539616283512"></a><a name="p1539616283512"></a>None</p>
</td>
</tr>
<tr id="row1439642825118"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.5.1"><p id="p2396928115112"><a name="p2396928115112"></a><a name="p2396928115112"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.5.1 "><p id="p1939617284511"><a name="p1939617284511"></a><a name="p1939617284511"></a>None</p>
</td>
</tr>
<tr id="row12396182811516"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.6.1"><p id="p1439682811514"><a name="p1439682811514"></a><a name="p1439682811514"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.6.1 "><a name="ul133383413220"></a><a name="ul133383413220"></a><ul id="ul133383413220"><li><span class="parmname" id="parmname1593203817210"><a name="parmname1593203817210"></a><a name="parmname1593203817210"></a><code>devices</code></span> must be valid, unique device IDs. The maximum number is 64.</li><li><span class="parmname" id="parmname11396162818517"><a name="parmname11396162818517"></a><a name="parmname11396162818517"></a><code>resources</code></span> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>
