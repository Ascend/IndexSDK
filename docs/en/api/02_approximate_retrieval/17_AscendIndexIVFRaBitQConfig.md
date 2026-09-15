# `AscendIndexIVFRaBitQConfig`<a name="en-us_TOPIC_0000002544944511"></a>

`AscendIndexIVFRaBitQ` must use the corresponding `AscendIndexIVFRaBitQConfig` to initialize the relevant resources.

## Member Overview<a name="section4211138173219"></a>

<a name="table388535175015"></a>
<table><thead align="left"><tr id="row11881435135015"><th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.1.4.1.1"><p id="p688635145015"><a name="p688635145015"></a><a name="p688635145015"></a>Member</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.1.4.1.2"><p id="p208815352501"><a name="p208815352501"></a><a name="p208815352501"></a>Type</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.1.4.1.3"><p id="p5891535145012"><a name="p5891535145012"></a><a name="p5891535145012"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row2890354502"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p1561239193314"><a name="p1561239193314"></a><a name="p1561239193314"></a>useRandomOrthogonalMatrix</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p1589135125017"><a name="p1589135125017"></a><a name="p1589135125017"></a><code>bool</code></p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p1789103575010"><a name="p1789103575010"></a><a name="p1789103575010"></a>Whether to use a random orthogonal matrix. Default: <code>true</code>.</p>
</td>
</tr>
<tr id="row78912359503"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p10523201623317"><a name="p10523201623317"></a><a name="p10523201623317"></a>needRefine</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p18993510505"><a name="p18993510505"></a><a name="p18993510505"></a><code>bool</code></p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p689113555010"><a name="p689113555010"></a><a name="p689113555010"></a>Whether refinement is required. Default: <code>false</code>.</p>
</td>
</tr>
<tr id="row188933513506"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p9321123113316"><a name="p9321123113316"></a><a name="p9321123113316"></a>matrixSeed</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p98919354505"><a name="p98919354505"></a><a name="p98919354505"></a><code>int</code></p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p2089235135016"><a name="p2089235135016"></a><a name="p2089235135016"></a>Random seed used to generate the random orthogonal matrix. Default: <code>12345</code>.</p>
</td>
</tr>
<tr id="row192882773310"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.1 "><p id="p8877163119331"><a name="p8877163119331"></a><a name="p8877163119331"></a>refineAlpha</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.2 "><p id="p12928122719330"><a name="p12928122719330"></a><a name="p12928122719330"></a><code>float</code></p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.1.4.1.3 "><p id="p3928142753311"><a name="p3928142753311"></a><a name="p3928142753311"></a>Refinement-related parameter. During retrieval, if the original plan is to retrieve the top <code>k</code>, refinement retrieves the top <code>k * refineAlpha</code> results first, and then takes the top <code>k</code> from them.</p>
<p id="p169972614290"><a name="p169972614290"></a><a name="p169972614290"></a>The default value is <code>2</code>. A larger value gives higher recall but lower retrieval efficiency.</p>
</td>
</tr>
</tbody>
</table>

## `AscendIndexIVFRaBitQConfig`<a name="section6579185362314"></a>

>**Note:**
>`AscendIndexIVFRaBitQConfig` inherits from <a href="../02_approximate_retrieval/04_AscendIndexIVFConfig.md#ascendindexivfconfig"><code>AscendIndexIVFConfig</code></a>.

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172966052218"><a name="p172966052218"></a><a name="p172966052218"></a><code>inline AscendIndexIVFRaBitQConfig();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13114316114718"><a name="p13114316114718"></a><a name="p13114316114718"></a>Default constructor. The default <code>devices</code> is <code>{0}</code>. Computation uses the 0th Ascend AI Processor, and the default <code>resource</code> is <code>128 MB</code>.</p>
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

<a name="table3725347611"></a>
<table><tbody><tr id="row137251141265"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1372544561"><a name="p1372544561"></a><a name="p1372544561"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4706533988"><a name="p4706533988"></a><a name="p4706533988"></a><code>inline AscendIndexIVFRaBitQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row0725941369"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p87251143611"><a name="p87251143611"></a><a name="p87251143611"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p099917815338"><a name="p099917815338"></a><a name="p099917815338"></a>Constructor for <code>AscendIndexIVFRaBitQConfig</code>, which creates an <code>AscendIndexIVFRaBitQConfig</code>. It configures Device-side Ascend AI Processor resources according to the values in <code>devices</code>, sets the resource pool size, and performs default initialization.</p>
</td>
</tr>
<tr id="row872516411614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p157251441762"><a name="p157251441762"></a><a name="p157251441762"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1172515411612"><a name="p1172515411612"></a><a name="p1172515411612"></a><strong id="b74801235171213"><a name="b74801235171213"></a><a name="b74801235171213"></a><code>std::initializer_list&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p999908173313"><a name="p999908173313"></a><a name="p999908173313"></a><strong id="b851894117126"><a name="b851894117126"></a><a name="b851894117126"></a><code>int64_t resourceSize</code></strong>: Size of the preallocated memory pool on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended.</p>
</td>
</tr>
<tr id="row13725184068"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p0725844620"><a name="p0725844620"></a><a name="p0725844620"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p37251747615"><a name="p37251747615"></a><a name="p37251747615"></a>None</p>
</td>
</tr>
<tr id="row19725104260"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p9725446613"><a name="p9725446613"></a><a name="p9725446613"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p77251043619"><a name="p77251043619"></a><a name="p77251043619"></a>None</p>
</td>
</tr>
<tr id="row7725641869"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p272634161"><a name="p272634161"></a><a name="p272634161"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5591115071213"></a><a name="ul5591115071213"></a><ul id="ul5591115071213"><li><code>devices</code> must be valid, unique device IDs. The maximum number is <code>64</code>.</li><li>The configured value of <code>resourceSize</code> must not exceed <code>4 * 1024 MB</code> (<code>4 * 1024 * 1024 * 1024</code> bytes). When set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value of <code>128 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table745471811619"></a>
<table><tbody><tr id="row445418187618"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p145417181561"><a name="p145417181561"></a><a name="p145417181561"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172151146898"><a name="p172151146898"></a><a name="p172151146898"></a><code>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row845519181169"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p164551418362"><a name="p164551418362"></a><a name="p164551418362"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1445513182614"><a name="p1445513182614"></a><a name="p1445513182614"></a>Constructor for <code>AscendIndexIVFRaBitQConfig</code>, which creates an <code>AscendIndexIVFRaBitQConfig</code>. It configures Device-side Ascend AI Processor resources according to the values in <code>devices</code>, sets the resource pool size, and performs default initialization.</p>
</td>
</tr>
<tr id="row845512181667"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14551718264"><a name="p14551718264"></a><a name="p14551718264"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1945571813613"><a name="p1945571813613"></a><a name="p1945571813613"></a><strong id="b9403131414155"><a name="b9403131414155"></a><a name="b9403131414155"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p11455318966"><a name="p11455318966"></a><a name="p11455318966"></a><strong id="b132471122150"><a name="b132471122150"></a><a name="b132471122150"></a><code>int resourceSize</code></strong>: Size of the preallocated memory pool on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended.</p>
</td>
</tr>
<tr id="row12455718267"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p645513184613"><a name="p645513184613"></a><a name="p645513184613"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1545511816618"><a name="p1545511816618"></a><a name="p1545511816618"></a>None</p>
</td>
</tr>
<tr id="row11455318162"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p845511186617"><a name="p845511186617"></a><a name="p845511186617"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p104556181962"><a name="p104556181962"></a><a name="p104556181962"></a>None</p>
</td>
</tr>
<tr id="row17455118361"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p64551618167"><a name="p64551618167"></a><a name="p64551618167"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9168241111516"></a><a name="ul9168241111516"></a><ul id="ul9168241111516"><li><code>devices</code> must be valid, unique device IDs. The maximum number is <code>64</code>.</li><li>The configured value of <code>resourceSize</code> must not exceed <code>4 * 1024 MB</code> (<code>4 * 1024 * 1024 * 1024</code> bytes). When set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value of <code>128 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1037111614358"></a>
<table><tbody><tr id="row837916103513"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p237101615359"><a name="p237101615359"></a><a name="p237101615359"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1237151683519"><a name="p1237151683519"></a><a name="p1237151683519"></a><code>inline AscendIndexIVFRaBitQConfig(std::vector&lt;int&gt; devices, bool useRandomOrthogonalMatrix_, bool needRefine_, int matrixSeed_, float alpha_, int64_t resourceSize = IVF_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row173761693511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p19379164358"><a name="p19379164358"></a><a name="p19379164358"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p14374169350"><a name="p14374169350"></a><a name="p14374169350"></a>Constructor for <code>AscendIndexIVFRaBitQConfig</code>, which creates an <code>AscendIndexIVFRaBitQConfig</code>. It performs initialization according to the input parameters.</p>
</td>
</tr>
<tr id="row163791683511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14371816183518"><a name="p14371816183518"></a><a name="p14371816183518"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p837151615356"><a name="p837151615356"></a><a name="p837151615356"></a><strong id="b163741683510"><a name="b163741683510"></a><a name="b163741683510"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p173258481363"><a name="p173258481363"></a><a name="p173258481363"></a><strong id="b87628480367"><a name="b87628480367"></a><a name="b87628480367"></a><code>bool useRandomOrthogonalMatrix_</code></strong>: Whether to use a random orthogonal matrix.</p>
<p id="p131691249163620"><a name="p131691249163620"></a><a name="p131691249163620"></a><strong id="b9570114914362"><a name="b9570114914362"></a><a name="b9570114914362"></a><code>bool needRefine_</code></strong>: Whether refinement is required.</p>
<p id="p1692910498362"><a name="p1692910498362"></a><a name="p1692910498362"></a><strong id="b227917501366"><a name="b227917501366"></a><a name="b227917501366"></a><code>int matrixSeed_</code></strong>: Random seed used to generate the random orthogonal matrix.</p>
<p id="p655111502366"><a name="p655111502366"></a><a name="p655111502366"></a><strong id="b208741450183610"><a name="b208741450183610"></a><a name="b208741450183610"></a><code>float alpha_</code></strong>: Refinement-related parameter.</p>
<p id="p1237101653516"><a name="p1237101653516"></a><a name="p1237101653516"></a><strong id="b153781620352"><a name="b153781620352"></a><a name="b153781620352"></a><code>int resourceSize</code></strong>: Size of the preallocated memory pool on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the <code>search</code> batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended.</p>
</td>
</tr>
<tr id="row3379162354"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p03711623518"><a name="p03711623518"></a><a name="p03711623518"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2037121633519"><a name="p2037121633519"></a><a name="p2037121633519"></a>None</p>
</td>
</tr>
<tr id="row1237716203511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20371816183513"><a name="p20371816183513"></a><a name="p20371816183513"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8371316133513"><a name="p8371316133513"></a><a name="p8371316133513"></a>None</p>
</td>
</tr>
<tr id="row193701643510"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p10371816193518"><a name="p10371816193518"></a><a name="p10371816193518"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul193714166355"></a><a name="ul193714166355"></a><ul id="ul193714166355"><li><code>devices</code> must be valid, unique device IDs. The maximum number is <code>64</code>.</li><li>The configured value of <code>resourceSize</code> must not exceed <code>4 * 1024 MB</code> (<code>4 * 1024 * 1024 * 1024</code> bytes). When set to <code>-1</code>, the Device-side Ascend AI Processor resource is configured to the default value of <code>128 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>
