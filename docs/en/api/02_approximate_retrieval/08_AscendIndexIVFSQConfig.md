# `AscendIndexIVFSQConfig`<a name="en-us_TOPIC_0000001456375204"></a>

`AscendIndexIVFSQ` requires the corresponding `AscendIndexIVFSQConfig` to initialize the corresponding resources.

**`AscendIndexIVFSQConfig`<a name="section015013311183"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>AscendIndexIVFSQConfig();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>Default constructor. The default <code>devices</code> value is <code>{0}</code>, so the 0th Ascend AI Processor is used for computation. The default <code>resource</code> value is <code>384 MB</code>.</p>
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

<a name="table19736185071817"></a>
<table><tbody><tr id="row673665061814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p13736350131811"><a name="p13736350131811"></a><a name="p13736350131811"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14172141316118"><a name="p14172141316118"></a><a name="p14172141316118"></a><code>inline AscendIndexIVFSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</code></p>
</td>
</tr>
<tr id="row1773645071818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1373675014185"><a name="p1373675014185"></a><a name="p1373675014185"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor for <code>AscendIndexIVFSQConfig</code>. It creates an <code>AscendIndexIVFSQConfig</code>, sets the Ascend AI Processor resources on the Device side according to the values configured in <code>devices</code>, configures the resource pool size, and performs default initialization.</p>
</td>
</tr>
<tr id="row37368508181"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p127361650191819"><a name="p127361650191819"></a><a name="p127361650191819"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p127360501187"><a name="p127360501187"></a><a name="p127360501187"></a><strong id="b97961612194319"><a name="b97961612194319"></a><a name="b97961612194319"></a><code>std::initializer_list&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b52891554312"><a name="b52891554312"></a><a name="b52891554312"></a><code>int64_t resourceSize</code></strong>: Preallocated memory pool size on the Device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVFSQ_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch count. When the base library size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</p>
</td>
</tr>
<tr id="row573613503187"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p11736135011819"><a name="p11736135011819"></a><a name="p11736135011819"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1773695013184"><a name="p1773695013184"></a><a name="p1773695013184"></a>None</p>
</td>
</tr>
<tr id="row173619506182"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1273618502186"><a name="p1273618502186"></a><a name="p1273618502186"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p57361450111811"><a name="p57361450111811"></a><a name="p57361450111811"></a>None</p>
</td>
</tr>
<tr id="row8736205051816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p147360502189"><a name="p147360502189"></a><a name="p147360502189"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul107261247431"></a><a name="ul107261247431"></a><ul id="ul107261247431"><li><code>devices</code> must be valid, unique device IDs.</li><li>The configured <code>resourceSize</code> value must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1056711401917"></a>
<table><tbody><tr id="row1956720419193"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18567124191912"><a name="p18567124191912"></a><a name="p18567124191912"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p656714416197"><a name="p656714416197"></a><a name="p656714416197"></a><code>inline AscendIndexIVFSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);</code></p>
</td>
</tr>
<tr id="row25671541197"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p105673420197"><a name="p105673420197"></a><a name="p105673420197"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p115671411920"><a name="p115671411920"></a><a name="p115671411920"></a>Constructor for <code>AscendIndexIVFSQConfig</code>. It creates an <code>AscendIndexIVFSQConfig</code>, sets the Ascend AI Processor resources on the Device side according to the values configured in <code>devices</code>, configures the resource pool size, and performs default initialization.</p>
</td>
</tr>
<tr id="row556720415197"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p956718420193"><a name="p956718420193"></a><a name="p956718420193"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p165676418192"><a name="p165676418192"></a><a name="p165676418192"></a><strong id="b2157122616458"><a name="b2157122616458"></a><a name="b2157122616458"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p15678461910"><a name="p15678461910"></a><a name="p15678461910"></a><strong id="b12875133214511"><a name="b12875133214511"></a><a name="b12875133214511"></a><code>int64_t resourceSize</code></strong>: Preallocated memory pool size on the Device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVFSQ_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch count. When the base library size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.</p>
</td>
</tr>
<tr id="row256744171918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1856714121912"><a name="p1856714121912"></a><a name="p1856714121912"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p17567114101912"><a name="p17567114101912"></a><a name="p17567114101912"></a>None</p>
</td>
</tr>
<tr id="row15567164161919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3567144111913"><a name="p3567144111913"></a><a name="p3567144111913"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p95677416193"><a name="p95677416193"></a><a name="p95677416193"></a>None</p>
</td>
</tr>
<tr id="row13567194101919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11567449197"><a name="p11567449197"></a><a name="p11567449197"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9973203054911"></a><a name="ul9973203054911"></a><ul id="ul9973203054911"><li><code>devices</code> must be valid, unique device IDs.</li><li>The configured <code>resourceSize</code> value must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>

**`SetDefaultIVFSQConfig`<a name="section039015215286"></a>**

<a name="table1185313082915"></a>
<table><tbody><tr id="row18531107298"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p128531018292"><a name="p128531018292"></a><a name="p128531018292"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58281361827"><a name="p58281361827"></a><a name="p58281361827"></a><code>inline void SetDefaultIVFSQConfig();</code></p>
</td>
</tr>
<tr id="row198530002911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p8853200102915"><a name="p8853200102915"></a><a name="p8853200102915"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p685316016291"><a name="p685316016291"></a><a name="p685316016291"></a>Performs default initialization. Sets the number of iterations to <code>16</code> and sets a maximum of <code>512</code> points for each centroid.</p>
</td>
</tr>
<tr id="row68536022919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p13853804297"><a name="p13853804297"></a><a name="p13853804297"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p158531705291"><a name="p158531705291"></a><a name="p158531705291"></a>None</p>
</td>
</tr>
<tr id="row785340142913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18538016297"><a name="p18538016297"></a><a name="p18538016297"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p198531703295"><a name="p198531703295"></a><a name="p198531703295"></a>None</p>
</td>
</tr>
<tr id="row1885319062919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1853120102913"><a name="p1853120102913"></a><a name="p1853120102913"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p485319062912"><a name="p485319062912"></a><a name="p485319062912"></a>None</p>
</td>
</tr>
<tr id="row188538018295"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p78535032910"><a name="p78535032910"></a><a name="p78535032910"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p128538010296"><a name="p128538010296"></a><a name="p128538010296"></a>None</p>
</td>
</tr>
</tbody>
</table>
