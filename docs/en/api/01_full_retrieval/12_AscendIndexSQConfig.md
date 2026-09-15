# `AscendIndexSQConfig`<a name="en-us_TOPIC_0000001456375392"></a>

`AscendIndexSQ` requires the corresponding `AscendIndexSQConfig` to initialize its resources.

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>inline AscendIndexSQConfig()</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>The default constructor of <code>AscendIndexSQConfig</code>. The default <code>deviceList</code> is <code>0</code>, which means the first Ascend AI Processor of the NPU is selected as the heterogeneous computing platform for <code>AscendFaiss</code> retrieval. The default resource pool size is used.</p>
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

<a name="table108621239568"></a>
<table><tbody><tr id="row1686242395610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p78621239565"><a name="p78621239565"></a><a name="p78621239565"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p133718164310"><a name="p133718164310"></a><a name="p133718164310"></a><code>inline AscendIndexSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t  blockSize = DEFAULT_BLOCK_SIZE)</code></p>
</td>
</tr>
<tr id="row178624230566"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7862192305612"><a name="p7862192305612"></a><a name="p7862192305612"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1686232319567"><a name="p1686232319567"></a><a name="p1686232319567"></a>The constructor of <code>AscendIndexSQConfig</code>. It creates an <code>AscendIndexSQConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</p>
</td>
</tr>
<tr id="row886222375617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2862723165612"><a name="p2862723165612"></a><a name="p2862723165612"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p128621723155611"><a name="p128621723155611"></a><a name="p128621723155611"></a><strong id="b18990511018"><a name="b18990511018"></a><a name="b18990511018"></a><code>std::initializer_list&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b64851871304"><a name="b64851871304"></a><a name="b64851871304"></a><code>int64_t resourceSize</code></strong>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>SQ_DEFAULT_MEM</code> defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>10,000,000</code> and the batch size is greater than or equal to <code>16</code>, you are advised to set it to <code>1024 MB</code>.</p>
<p id="p10918131681313"><a name="p10918131681313"></a><a name="p10918131681313"></a><strong id="b31638114817"><a name="b31638114817"></a><a name="b31638114817"></a><code>uint32_t blockSize</code></strong>: Configures the <code>blockSize</code> on the Device side. It constrains the amount of data processed in a single <code>tik</code> operator execution and the size of vectors stored in each shard of the base library. The default value is <code>16384 * 16 = 262144</code>. This value affects the maximum number of <code>Index</code> objects that can be created and retrieval performance.</p>
</td>
</tr>
<tr id="row986352311564"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7863823135618"><a name="p7863823135618"></a><a name="p7863823135618"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1286342314562"><a name="p1286342314562"></a><a name="p1286342314562"></a>None</p>
</td>
</tr>
<tr id="row0863723185611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p208632023155619"><a name="p208632023155619"></a><a name="p208632023155619"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1386313230564"><a name="p1386313230564"></a><a name="p1386313230564"></a>None</p>
</td>
</tr>
<tr id="row486382311561"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15863423115615"><a name="p15863423115615"></a><a name="p15863423115615"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul17466113116595"></a><a name="ul17466113116595"></a><ul id="ul17466113116595"><li><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>.</li><li>The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</li><li>The valid values of <code>blockSize</code> are <code>{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1735412445711"></a>
<table><tbody><tr id="row19354134175714"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1635417413572"><a name="p1635417413572"></a><a name="p1635417413572"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19780437164418"><a name="p19780437164418"></a><a name="p19780437164418"></a><code>inline AscendIndexSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t  blockSize = DEFAULT_BLOCK_SIZE)</code></p>
</td>
</tr>
<tr id="row93540419578"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1635420425713"><a name="p1635420425713"></a><a name="p1635420425713"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13546445718"><a name="p13546445718"></a><a name="p13546445718"></a>The constructor of <code>AscendIndexSQConfig</code>. It creates an <code>AscendIndexSQConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</p>
</td>
</tr>
<tr id="row33541741571"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1135414185711"><a name="p1135414185711"></a><a name="p1135414185711"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1135410418574"><a name="p1135410418574"></a><a name="p1135410418574"></a><strong id="b3303144712017"><a name="b3303144712017"></a><a name="b3303144712017"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p13541545573"><a name="p13541545573"></a><a name="p13541545573"></a><strong id="b197022051409"><a name="b197022051409"></a><a name="b197022051409"></a><code>int64_t resourceSize</code></strong>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>SQ_DEFAULT_MEM</code> defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>10,000,000</code> and the batch size is greater than or equal to <code>16</code>, you are advised to set it to <code>1024 MB</code>.</p>
<p id="p1035454195716"><a name="p1035454195716"></a><a name="p1035454195716"></a><strong id="b12664181414917"><a name="b12664181414917"></a><a name="b12664181414917"></a><code>uint32_t blockSize</code></strong>: Configures the <code>blockSize</code> on the Device side. It constrains the amount of data processed in a single <code>tik</code> operator execution and the size of vectors stored in each shard of the base library. The default value is <code>16384 * 16 = 262144</code>. This value affects the maximum number of <code>Index</code> objects that can be created and retrieval performance.</p>
</td>
</tr>
<tr id="row2354104115713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14354184105712"><a name="p14354184105712"></a><a name="p14354184105712"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p13540455713"><a name="p13540455713"></a><a name="p13540455713"></a>None</p>
</td>
</tr>
<tr id="row1354442570"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20354749572"><a name="p20354749572"></a><a name="p20354749572"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p163541948576"><a name="p163541948576"></a><a name="p163541948576"></a>None</p>
</td>
</tr>
<tr id="row2354174135711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p153545418579"><a name="p153545418579"></a><a name="p153545418579"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1931270712"></a><a name="ul1931270712"></a><ul id="ul1931270712"><li><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>.</li><li>The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</li><li>The valid values of <code>blockSize</code> are <code>{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}</code>.</li></ul>
</td>
</tr>
</tbody>
</table>
