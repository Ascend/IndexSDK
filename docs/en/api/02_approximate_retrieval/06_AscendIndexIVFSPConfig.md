# `AscendIndexIVFSPConfig`<a id="en-us_TOPIC_0000001635696057"></a>

`AscendIndexIVFSP` requires the corresponding `AscendIndexIVFSPConfig` to initialize the corresponding resources.

**Common Parameters<a name="section17656114673616"></a>**

| Parameter | Data Type | Description |
|--|--|--|
| handleBatch | `int` | Number of candidate buckets submitted for computation each time during search. The default value is `64`. |
| nprobe | `int` | Total number of candidate buckets used during search. The default value is `64`. |
| searchListSize | `int` | Maximum number of samples in each bucket submitted for computation each time during search. The default value is `32768`. If a bucket is too large, the program automatically splits the bucket into multiple operator submissions according to `searchListSize` to compute distances. |

**API Description<a name="section74781713710"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>inline AscendIndexIVFSPConfig();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>Default constructor. The default <code>devices</code> value is <code>{0}</code>, so the 0th Ascend AI Processor is used for computation. The default <code>resources</code> value is <code>128 MB</code>.</p>
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

<a name="table121971648373"></a>
<table><tbody><tr id="row13197134820716"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p51977481976"><a name="p51977481976"></a><a name="p51977481976"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161970481773"><a name="p161970481773"></a><a name="p161970481773"></a><code>inline explicit AscendIndexIVFSPConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></p>
</td>
</tr>
<tr id="row141971748972"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1419717481876"><a name="p1419717481876"></a><a name="p1419717481876"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor for <code>AscendIndexIVFSPConfig</code>. It creates an <code>AscendIndexIVFSPConfig</code> and specifies the Device-side device IDs and the resource pool size.</p>
</td>
</tr>
<tr id="row191973486716"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1019719485712"><a name="p1019719485712"></a><a name="p1019719485712"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p59096365498"><a name="p59096365498"></a><a name="p59096365498"></a><strong id="b2851659184912"><a name="b2851659184912"></a><a name="b2851659184912"></a><code>std::initializer_list&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p9909173617494"><a name="p9909173617494"></a><a name="p9909173617494"></a><strong id="b61011425012"><a name="b61011425012"></a><a name="b61011425012"></a><code>int64_t resources</code></strong>: Preallocated memory pool size on the Device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVF_SP_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch count. When the base library size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</p>
<p id="p1990912367496"><a name="p1990912367496"></a><a name="p1990912367496"></a><strong id="b19708106145014"><a name="b19708106145014"></a><a name="b19708106145014"></a><code>uint32_t blockSize</code></strong>: Preallocated memory block size, in bytes. The default value is <code>DEFAULT_BLOCK_SIZE</code> in the header file.</p>
</td>
</tr>
<tr id="row61979480720"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p91979481711"><a name="p91979481711"></a><a name="p91979481711"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1719734811717"><a name="p1719734811717"></a><a name="p1719734811717"></a>None</p>
</td>
</tr>
<tr id="row1919711482718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p181974481777"><a name="p181974481777"></a><a name="p181974481777"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p919717486712"><a name="p919717486712"></a><a name="p919717486712"></a>None</p>
</td>
</tr>
<tr id="row1719719481072"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p181973481672"><a name="p181973481672"></a><a name="p181973481672"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20548163415505"></a><a name="ul20548163415505"></a><ul id="ul20548163415505"><li><code>devices</code> must be valid, unique device IDs. Currently, only one NPU device is supported.</li><li>The configured <code>resources</code> value must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table56061252785"></a>
<table><tbody><tr id="row6606552282"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p06062521781"><a name="p06062521781"></a><a name="p06062521781"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p108121410115113"><a name="p108121410115113"></a><a name="p108121410115113"></a><code>inline explicit AscendIndexIVFSPConfig(std::vector&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></p>
</td>
</tr>
<tr id="row156061352486"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p460610521381"><a name="p460610521381"></a><a name="p460610521381"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8606152685"><a name="p8606152685"></a><a name="p8606152685"></a>Constructor for <code>AscendIndexIVFSPConfig</code>. It creates an <code>AscendIndexIVFSPConfig</code> and specifies the Device-side device IDs and the resource pool size.</p>
</td>
</tr>
<tr id="row146067521289"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p46064521488"><a name="p46064521488"></a><a name="p46064521488"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p15921231125111"><a name="p15921231125111"></a><a name="p15921231125111"></a><strong id="b18248195515616"><a name="b18248195515616"></a><a name="b18248195515616"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p09211631145120"><a name="p09211631145120"></a><a name="p09211631145120"></a><strong id="b11958155810618"><a name="b11958155810618"></a><a name="b11958155810618"></a><code>int64_t resources</code></strong>: Preallocated memory pool size on the Device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVF_SP_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch count. When the base library size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</p>
<p id="p139219313517"><a name="p139219313517"></a><a name="p139219313517"></a><strong id="b159531621471"><a name="b159531621471"></a><a name="b159531621471"></a><code>uint32_t blockSize</code></strong>: Preallocated memory block size, in bytes. The default value is <code>DEFAULT_BLOCK_SIZE</code> in the header file.</p>
</td>
</tr>
<tr id="row1160718521816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9607252586"><a name="p9607252586"></a><a name="p9607252586"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p106076525815"><a name="p106076525815"></a><a name="p106076525815"></a>None</p>
</td>
</tr>
<tr id="row36075522089"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6607125210819"><a name="p6607125210819"></a><a name="p6607125210819"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p26071521080"><a name="p26071521080"></a><a name="p26071521080"></a>None</p>
</td>
</tr>
<tr id="row8607152585"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p160714523812"><a name="p160714523812"></a><a name="p160714523812"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul899544135119"></a><a name="ul899544135119"></a><ul id="ul899544135119"><li><code>devices</code> must be valid, unique device IDs. Currently, only one NPU device is supported.</li><li>The configured <code>resources</code> value must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>
