# `AscendIndexIVFSQTConfig`<a name="en-us_TOPIC_0000001506495881"></a>

`AscendIndexIVFSQT` uses the corresponding `AscendIndexIVFSQTConfig` to initialize the required resources.

**AscendIndexIVFSQTConfig<a name="section6579185362314"></a>**

> [!NOTE]
> `AscendIndexIVFSQTConfig` inherits from <a href="./08_AscendIndexIVFSQConfig.md#ascendindexivfsqconfig">`AscendIndexIVFSQConfig`</a>.

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>inline AscendIndexIVFSQTConfig();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>Default constructor. The default <code>devices</code> value is <code>{0}</code>, which uses the 0th Ascend AI Processor for computation. The default <code>resource</code> value is <code>384 MB</code>.</p>
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

<a name="table42413462115"></a>
<table><tbody><tr id="row1524133414212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1224153422117"><a name="p1224153422117"></a><a name="p1224153422117"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p14172141316118"><a name="p14172141316118"></a><a name="p14172141316118"></a><code>inline AscendIndexIVFSQTConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></p>
</td>
</tr>
<tr id="row72433412120"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p324153419217"><a name="p324153419217"></a><a name="p324153419217"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor for <code>AscendIndexIVFSQTConfig</code>. It creates an <code>AscendIndexIVFSQTConfig</code> instance and, based on the values configured in <code>devices</code>, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization.</p>
</td>
</tr>
<tr id="row124103412219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p72414340215"><a name="p72414340215"></a><a name="p72414340215"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1624123417213"><a name="p1624123417213"></a><a name="p1624123417213"></a><strong id="b20622101413581"><a name="b20622101413581"></a><a name="b20622101413581"></a><code>std::initializer_list&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b2777720145818"><a name="b2777720145818"></a><a name="b2777720145818"></a><code>int64_t resourceSize</code></strong>: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVFSQT_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to <code>1024 MB</code> when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16.</p>
</td>
</tr>
<tr id="row62417348212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p142463482114"><a name="p142463482114"></a><a name="p142463482114"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p624134112112"><a name="p624134112112"></a><a name="p624134112112"></a>None</p>
</td>
</tr>
<tr id="row202443411211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p225193492114"><a name="p225193492114"></a><a name="p225193492114"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16255347215"><a name="p16255347215"></a><a name="p16255347215"></a>None</p>
</td>
</tr>
<tr id="row9251334172114"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11251534172114"><a name="p11251534172114"></a><a name="p11251534172114"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1626263017587"></a><a name="ul1626263017587"></a><ul id="ul1626263017587"><li><code>devices</code> must contain valid, non-duplicate device IDs.</li><li>The configured value of <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table0812225238"></a>
<table><tbody><tr id="row681152292314"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p181722132314"><a name="p181722132314"></a><a name="p181722132314"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15811522152312"><a name="p15811522152312"></a><a name="p15811522152312"></a><code>inline AscendIndexIVFSQTConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);</code></p>
</td>
</tr>
<tr id="row681722142311"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p12811722132310"><a name="p12811722132310"></a><a name="p12811722132310"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1981132202311"><a name="p1981132202311"></a><a name="p1981132202311"></a>Constructor for <code>AscendIndexIVFSQTConfig</code>. It creates an <code>AscendIndexIVFSQTConfig</code> instance and, based on the values configured in <code>devices</code>, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization.</p>
</td>
</tr>
<tr id="row158132218234"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1481522122316"><a name="p1481522122316"></a><a name="p1481522122316"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19811822132311"><a name="p19811822132311"></a><a name="p19811822132311"></a><strong id="b1316015619589"><a name="b1316015619589"></a><a name="b1316015619589"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p1081182213237"><a name="p1081182213237"></a><a name="p1081182213237"></a><strong id="b9205192205911"><a name="b9205192205911"></a><a name="b9205192205911"></a><code>int64_t resourceSize</code></strong>: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVFSQT_DEFAULT_TEMP_MEM</code> in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to <code>1024 MB</code> when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16.</p>
</td>
</tr>
<tr id="row281522152313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18811922152314"><a name="p18811922152314"></a><a name="p18811922152314"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1381192212233"><a name="p1381192212233"></a><a name="p1381192212233"></a>None</p>
</td>
</tr>
<tr id="row4811922172317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p28113227234"><a name="p28113227234"></a><a name="p28113227234"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1181322132318"><a name="p1181322132318"></a><a name="p1181322132318"></a>None</p>
</td>
</tr>
<tr id="row081622142316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p18811622102319"><a name="p18811622102319"></a><a name="p18811622102319"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1092012916596"></a><a name="ul1092012916596"></a><ul id="ul1092012916596"><li><code>devices</code> must contain valid, non-duplicate device IDs.</li><li>The configured value of <code>resourceSize</code> must not exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes).</li></ul>
</td>
</tr>
</tbody>
</table>

**SetDefaultIVFSQConfig<a name="section18396165022414"></a>**

<a name="table14953182017255"></a>
<table><tbody><tr id="row1495372015250"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p495312072515"><a name="p495312072515"></a><a name="p495312072515"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58281361827"><a name="p58281361827"></a><a name="p58281361827"></a><code>inline void SetDefaultIVFSQConfig();</code></p>
</td>
</tr>
<tr id="row69531020142513"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p14953152072517"><a name="p14953152072517"></a><a name="p14953152072517"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p19535201256"><a name="p19535201256"></a><a name="p19535201256"></a>Performs the default initialization, sets the number of iterations to <code>16</code>, and sets a maximum of <code>512</code> points for each centroid.</p>
</td>
</tr>
<tr id="row495362020258"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5953182032513"><a name="p5953182032513"></a><a name="p5953182032513"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p3953420122514"><a name="p3953420122514"></a><a name="p3953420122514"></a>None</p>
</td>
</tr>
<tr id="row1895392013254"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p139538203259"><a name="p139538203259"></a><a name="p139538203259"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p895322019257"><a name="p895322019257"></a><a name="p895322019257"></a>None</p>
</td>
</tr>
<tr id="row109531520182518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p995312017256"><a name="p995312017256"></a><a name="p995312017256"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7953152032518"><a name="p7953152032518"></a><a name="p7953152032518"></a>None</p>
</td>
</tr>
<tr id="row1795392052510"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1095382052518"><a name="p1095382052518"></a><a name="p1095382052518"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p2095302062513"><a name="p2095302062513"></a><a name="p2095302062513"></a>None</p>
</td>
</tr>
</tbody>
</table>
