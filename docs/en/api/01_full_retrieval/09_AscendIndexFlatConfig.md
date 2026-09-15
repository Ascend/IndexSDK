# `AscendIndexFlatConfig`<a name="en-us_TOPIC_0000001456375216"></a>

<code>AscendIndexFlat</code> requires the corresponding <code>AscendIndexFlatConfig</code> to initialize the corresponding resources.

**API Description<a name="section140920164419"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>inline AscendIndexFlatConfig()</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>The default constructor of <code>AscendIndexFlatConfig</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table46951722104415"></a>
<table><tbody><tr id="row186961822204410"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1069662218442"><a name="p1069662218442"></a><a name="p1069662218442"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1792514755010"><a name="p1792514755010"></a><a name="p1792514755010"></a><code>inline AscendIndexFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></p>
</td>
</tr>
<tr id="row169692210443"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p969622274410"><a name="p969622274410"></a><a name="p969622274410"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7696172274411"><a name="p7696172274411"></a><a name="p7696172274411"></a>The constructor of <code>AscendIndexFlatConfig</code>. It creates an <code>AscendIndexFlatConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</p>
</td>
</tr>
<tr id="row136963220449"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p146961622204418"><a name="p146961622204418"></a><a name="p146961622204418"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p5696122214448"><a name="p5696122214448"></a><a name="p5696122214448"></a><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.</p>
<p id="p17641112731910"><a name="p17641112731910"></a><a name="p17641112731910"></a><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>4194304</code> and the batch size is greater than or equal to <code>16</code>, use the following recommendations.</p>
<a name="ul1067904281918"></a><a name="ul1067904281918"></a><ul id="ul1067904281918"><li>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_L2</code>, the recommended value is <code>1024 MB</code>.</li><li>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_INNER_PRODUCT</code>, the recommended value is <code>1280 MB</code>.</li></ul>
</td>
</tr>
<tr id="row16696172214415"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1069610226440"><a name="p1069610226440"></a><a name="p1069610226440"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p669620222449"><a name="p669620222449"></a><a name="p669620222449"></a>None</p>
</td>
</tr>
<tr id="row16696192264412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3696922114411"><a name="p3696922114411"></a><a name="p3696922114411"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p56962225445"><a name="p56962225445"></a><a name="p56962225445"></a>None</p>
</td>
</tr>
<tr id="row169602211448"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1469617229447"><a name="p1469617229447"></a><a name="p1469617229447"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul44541139318"></a><a name="ul44541139318"></a><ul id="ul44541139318"><li><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>.</li><li>The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table842319354444"></a>
<table><tbody><tr id="row1142318355442"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1842393554413"><a name="p1842393554413"></a><a name="p1842393554413"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p593315225213"><a name="p593315225213"></a><a name="p593315225213"></a><code>inline AscendIndexFlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</code></p>
</td>
</tr>
<tr id="row1242323524413"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p84231035164418"><a name="p84231035164418"></a><a name="p84231035164418"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1142333517449"><a name="p1142333517449"></a><a name="p1142333517449"></a>The constructor of <code>AscendIndexFlatConfig</code>. It creates an <code>AscendIndexFlatConfig</code> and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in <code>devices</code>.</p>
</td>
</tr>
<tr id="row94235350446"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p742383516443"><a name="p742383516443"></a><a name="p742383516443"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14423173514449"><a name="p14423173514449"></a><a name="p14423173514449"></a><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>int64_t resourceSize</code>: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>FLAT_DEFAULT_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to <code>4194304</code> and the batch size is greater than or equal to <code>16</code>, use the following recommendations.</p>
<a name="ul3423163517446"></a><a name="ul3423163517446"></a><ul id="ul3423163517446"><li>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_L2</code>, the recommended value is <code>1024 MB</code>.</li><li>When the distance type of <code>AscendIndexFlat</code> is <code>faiss::METRIC_INNER_PRODUCT</code>, the recommended value is <code>1280 MB</code>.</li></ul>
</td>
</tr>
<tr id="row1842343514447"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p642343513448"><a name="p642343513448"></a><a name="p642343513448"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p4424183564416"><a name="p4424183564416"></a><a name="p4424183564416"></a>None</p>
</td>
</tr>
<tr id="row11424135174412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p24244358441"><a name="p24244358441"></a><a name="p24244358441"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16424135124411"><a name="p16424135124411"></a><a name="p16424135124411"></a>None</p>
</td>
</tr>
<tr id="row14424183594412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p17424133524410"><a name="p17424133524410"></a><a name="p17424133524410"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1811714915354"></a><a name="ul1811714915354"></a><ul id="ul1811714915354"><li><code>devices</code> must contain valid, unique device IDs. The maximum number is <code>64</code>.</li><li>The value configured for <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When this value is set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value <code>128 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>
