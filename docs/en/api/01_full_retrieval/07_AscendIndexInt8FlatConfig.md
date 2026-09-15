# `AscendIndexInt8FlatConfig`<a name="en-us_TOPIC_0000001456535040"></a>

<code>AscendIndexInt8Flat</code> requires the corresponding <code>AscendIndexInt8FlatConfig</code> to initialize the associated resources.

**Members<a name="section1372191465013"></a>**

| Member | Type | Description |
| ------ | ---- | ----------- |
| dIndexMode | Int8IndexMode | Configures the INT8 retrieval mode for the <code>Index</code>. |
| dBlockSize | uint32_t | Configures the device-side <code>blockSize</code>. |

**API Description<a name="section136272015172914"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>AscendIndexInt8FlatConfig(uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndexInt8FlatConfig</code>. It creates an <code>AscendIndexInt8FlatConfig</code>, configures the device-side <code>blockSize</code>, and configures the INT8 retrieval mode.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><code>uint32_t blockSize</code>: Configures the device-side <code>blockSize</code>. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</p>
<p id="p196421349716"><a name="p196421349716"></a><a name="p196421349716"></a><code>Int8IndexMode indexMode</code>: Configures the INT8 retrieval mode for the <code>Index</code>. The default value is <code>DEFAULT_MODE</code>.</p>
<a name="ul112020291075"></a><a name="ul112020291075"></a><ul id="ul112020291075"><li><code>DEFAULT_MODE</code>: Default mode.</li><li><code>PIPE_SEARCH_MODE</code>: This mode is optimized for scenarios where the batch is greater than or equal to <code>128</code>. When you use this mode, you are advised to set <code>resourceSize</code> to at least <code>1324 MB</code>.</li><li><code>WITHOUT_NORM_MODE</code>: This mode is not supported at this time.</li></ul>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul72731322134614"></a><a name="ul72731322134614"></a><ul id="ul72731322134614"><li>The set of valid <code>blockSize</code> values is {<code>16384</code>, <code>32768</code>, <code>65536</code>, <code>131072</code>, <code>262144</code>}.</li></ul>
<a name="ul17184174914612"></a><a name="ul17184174914612"></a><ul id="ul17184174914612"><li>In <code>PIPE_SEARCH_MODE</code>, <code>AscendIndexInt8Flat</code> supports only <code>METRIC_L2</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1258103643012"></a>
<table><tbody><tr id="row95803619306"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p15853617303"><a name="p15853617303"></a><a name="p15853617303"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p558236163018"><a name="p558236163018"></a><a name="p558236163018"></a><code>AscendIndexInt8FlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</code></p>
</td>
</tr>
<tr id="row10580363301"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1058436123019"><a name="p1058436123019"></a><a name="p1058436123019"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p459183617305"><a name="p459183617305"></a><a name="p459183617305"></a>Constructor of <code>AscendIndexInt8FlatConfig</code>. It creates an <code>AscendIndexInt8FlatConfig</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>. It also configures the device-side <code>blockSize</code> and the INT8 retrieval mode.</p>
</td>
</tr>
<tr id="row9592036113014"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p145953643012"><a name="p145953643012"></a><a name="p145953643012"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p165943663010"><a name="p165943663010"></a><a name="p165943663010"></a><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>int64_t resourceSize</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INT8_FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</p>
<p id="p266920816252"><a name="p266920816252"></a><a name="p266920816252"></a><code>uint32_t blockSize</code>: Configures the device-side <code>blockSize</code>. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</p>
<p id="p920025214811"><a name="p920025214811"></a><a name="p920025214811"></a><code>Int8IndexMode indexMode</code>: Configures the INT8 retrieval mode for the <code>Index</code>. The default value is <code>DEFAULT_MODE</code>.</p>
<a name="ul19317031497"></a><a name="ul19317031497"></a><ul id="ul19317031497"><li><code>DEFAULT_MODE</code>: Default mode.</li><li><code>PIPE_SEARCH_MODE</code>: This mode is optimized for scenarios where the batch is greater than or equal to <code>128</code>. When you use this mode, you are advised to set <code>resourceSize</code> to at least <code>1324 MB</code>.</li><li><code>WITHOUT_NORM_MODE</code>: This mode is not supported at this time.</li></ul>
</td>
</tr>
<tr id="row859836143020"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p359536193015"><a name="p359536193015"></a><a name="p359536193015"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p659336153018"><a name="p659336153018"></a><a name="p659336153018"></a>None</p>
</td>
</tr>
<tr id="row105953623010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p759536193011"><a name="p759536193011"></a><a name="p759536193011"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p195910363302"><a name="p195910363302"></a><a name="p195910363302"></a>None</p>
</td>
</tr>
<tr id="row1259173611301"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p1059173683011"><a name="p1059173683011"></a><a name="p1059173683011"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul07292414471"></a><a name="ul07292414471"></a><ul id="ul07292414471"><li><code>devices</code> must be valid, unique device IDs, and the maximum number is 64.</li><li>The configured <code>resourceSize</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes). When the batch is greater than or equal to <code>96</code>, you are advised to set <code>resourceSize</code> to at least <code>2 * 1024 MB</code> to improve algorithm performance.</li><li>The set of valid <code>blockSize</code> values is {<code>16384</code>, <code>32768</code>, <code>65536</code>, <code>131072</code>, <code>262144</code>}.</li><li>In <code>PIPE_SEARCH_MODE</code>, <code>AscendIndexInt8Flat</code> supports only <code>METRIC_L2</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table8629135217302"></a>
<table><tbody><tr id="row6630115223010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p126301152183015"><a name="p126301152183015"></a><a name="p126301152183015"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1363005243010"><a name="p1363005243010"></a><a name="p1363005243010"></a><code>AscendIndexInt8FlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)</code></p>
</td>
</tr>
<tr id="row1630175243019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p16630752193012"><a name="p16630752193012"></a><a name="p16630752193012"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p263012528307"><a name="p263012528307"></a><a name="p263012528307"></a>Constructor of <code>AscendIndexInt8FlatConfig</code>. It creates an <code>AscendIndexInt8FlatConfig</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>. It also configures the device-side <code>blockSize</code> and the INT8 retrieval mode.</p>
</td>
</tr>
<tr id="row363011522300"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p196301452183019"><a name="p196301452183019"></a><a name="p196301452183019"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p4630205293012"><a name="p4630205293012"></a><a name="p4630205293012"></a><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.</p>
<p id="p2063075223014"><a name="p2063075223014"></a><a name="p2063075223014"></a><code>int64_t resourceSize</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INT8_FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</p>
<p id="p5467172216269"><a name="p5467172216269"></a><a name="p5467172216269"></a><code>uint32_t blockSize</code>: Configures the device-side <code>blockSize</code>. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</p>
<p id="p38771541295"><a name="p38771541295"></a><a name="p38771541295"></a><code>Int8IndexMode indexMode</code>: Configures the INT8 retrieval mode for the <code>Index</code>. The default value is <code>DEFAULT_MODE</code>.</p>
<a name="ul57151159696"></a><a name="ul57151159696"></a><ul id="ul57151159696"><li><code>DEFAULT_MODE</code>: Default mode.</li><li><code>PIPE_SEARCH_MODE</code>: This mode is optimized for scenarios where the batch is greater than or equal to <code>128</code>. When you use this mode, you are advised to set <code>resourceSize</code> to at least <code>1324 MB</code>.</li><li><code>WITHOUT_NORM_MODE</code>: This mode is not supported at this time.</li></ul>
</td>
</tr>
<tr id="row20630135293014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p116301652203017"><a name="p116301652203017"></a><a name="p116301652203017"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15630952103017"><a name="p15630952103017"></a><a name="p15630952103017"></a>None</p>
</td>
</tr>
<tr id="row2630852173011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20630185213016"><a name="p20630185213016"></a><a name="p20630185213016"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p176301752133010"><a name="p176301752133010"></a><a name="p176301752133010"></a>None</p>
</td>
</tr>
<tr id="row156301552103017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9630105253015"><a name="p9630105253015"></a><a name="p9630105253015"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul115901014144816"></a><a name="ul115901014144816"></a><ul id="ul115901014144816"><li><code>devices</code> must be valid, unique device IDs, and the maximum number is 64.</li><li>The configured <code>resourceSize</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes). When the batch is greater than or equal to <code>96</code>, you are advised to set <code>resourceSize</code> to at least <code>2 * 1024 MB</code> to improve algorithm performance.</li><li>The set of valid <code>blockSize</code> values is {<code>16384</code>, <code>32768</code>, <code>65536</code>, <code>131072</code>, <code>262144</code>}.</li><li>In <code>PIPE_SEARCH_MODE</code>, <code>AscendIndexInt8Flat</code> supports only <code>METRIC_L2</code>.</li></ul>
</td>
</tr>
</tbody>
</table>
