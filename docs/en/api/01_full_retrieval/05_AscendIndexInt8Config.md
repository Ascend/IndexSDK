# `AscendIndexInt8Config`<a id="en-us_TOPIC_0000001456854968"></a>

<code>AscendIndexInt8</code> requires the corresponding <code>AscendIndexInt8Config</code> to initialize the associated resources.

**Members<a name="section1372191465013"></a>**

| Member | Type | Description |
| ------ | ---- | ----------- |
| deviceList | std::vector\<int> | Device-side device IDs. |
| resourceSize | int64_t | Preallocated memory pool size on the device side, in bytes. |

**API Description<a name="section135441937164218"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>AscendIndexInt8Config()</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Default constructor of <code>AscendIndexInt8Config</code>. The default <code>deviceList</code> is <code>0</code>, which means Ascend AI Processor <code>0</code> on the NPU is used as the heterogeneous computing platform for AscendFaiss retrieval. The default resource pool size is used.</p>
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

<a name="table012165162914"></a>
<table><tbody><tr id="row71210582913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p31219512297"><a name="p31219512297"></a><a name="p31219512297"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132201932191815"><a name="p132201932191815"></a><a name="p132201932191815"></a><code>AscendIndexInt8Config(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></p>
</td>
</tr>
<tr id="row212554294"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1012185102912"><a name="p1012185102912"></a><a name="p1012185102912"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p71214510295"><a name="p71214510295"></a><a name="p71214510295"></a>Constructor of <code>AscendIndexInt8Config</code>. It creates an <code>AscendIndexInt8Config</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>.</p>
</td>
</tr>
<tr id="row101210562912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1612856291"><a name="p1612856291"></a><a name="p1612856291"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p13185103613195"><a name="p13185103613195"></a><a name="p13185103613195"></a><code>std::initializer_list&lt;int&gt; devices</code>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><code>int64_t resources</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INDEX_INT8_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</p>
</td>
</tr>
<tr id="row201311582910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7139542914"><a name="p7139542914"></a><a name="p7139542914"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p913145182911"><a name="p913145182911"></a><a name="p913145182911"></a>None</p>
</td>
</tr>
<tr id="row61312510298"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6137572910"><a name="p6137572910"></a><a name="p6137572910"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7131256297"><a name="p7131256297"></a><a name="p7131256297"></a>None</p>
</td>
</tr>
<tr id="row81316519296"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p41316552918"><a name="p41316552918"></a><a name="p41316552918"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul12323102516242"></a><a name="ul12323102516242"></a><ul id="ul12323102516242"><li><code>devices</code> must be valid, unique device IDs, and the maximum number is 64.</li><li>The configured <code>resources</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table9202719152913"></a>
<table><tbody><tr id="row620221922910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p152021219162918"><a name="p152021219162918"></a><a name="p152021219162918"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p620271952915"><a name="p620271952915"></a><a name="p620271952915"></a><code>AscendIndexInt8Config(std::vector&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</code></p>
</td>
</tr>
<tr id="row720217191294"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7202151915293"><a name="p7202151915293"></a><a name="p7202151915293"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8202419152912"><a name="p8202419152912"></a><a name="p8202419152912"></a>Constructor of <code>AscendIndexInt8Config</code>. It creates an <code>AscendIndexInt8Config</code> and configures device-side Ascend AI Processor resources and the resource pool size according to the values in <code>devices</code>.</p>
</td>
</tr>
<tr id="row7202101919297"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p02021219172918"><a name="p02021219172918"></a><a name="p02021219172918"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p17202319192913"><a name="p17202319192913"></a><a name="p17202319192913"></a><code>std::vector&lt;int&gt; devices</code>: Device-side device IDs.</p>
<p id="p32021619132913"><a name="p32021619132913"></a><a name="p32021619132913"></a><code>int64_t resources</code>: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>INDEX_INT8_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to <code>1024 MB</code>.</p>
</td>
</tr>
<tr id="row22021519142913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p202027199296"><a name="p202027199296"></a><a name="p202027199296"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p202021619112918"><a name="p202021619112918"></a><a name="p202021619112918"></a>None</p>
</td>
</tr>
<tr id="row120218193297"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2202171902914"><a name="p2202171902914"></a><a name="p2202171902914"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1720211192299"><a name="p1720211192299"></a><a name="p1720211192299"></a>None</p>
</td>
</tr>
<tr id="row520291962913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1720213191294"><a name="p1720213191294"></a><a name="p1720213191294"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul373416141258"></a><a name="ul373416141258"></a><ul id="ul373416141258"><li><code>devices</code> must be valid, unique device IDs, and the maximum number is 64.</li><li>The configured <code>resources</code> value must not exceed <code>16 * 1024 MB</code> (<code>16 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>
