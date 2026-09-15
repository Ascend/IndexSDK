# `AscendIndexIVFConfig`<a name="en-us_TOPIC_0000001456535024"></a>

## Function Description<a name="en-us_TOPIC_0000001456695128"></a>

`AscendIndexIVF` uses the corresponding `AscendIndexIVFConfig` to initialize the corresponding resources.

**Members<a name="section1372191465013"></a>**

|Member|Type|Description|
|--|--|--|
|flatConfig|`AscendIndexConfig`|Parameter configuration object.|
|useKmeansPP|`bool`|Whether to use NPU acceleration for the IVF clustering process.|
|cp|`ClusteringParameters`|Clustering-related parameters. For details, see the relevant Faiss API documentation. You are not advised to modify this parameter. The default number of training iterations is `16`. Setting the number of iterations too large significantly increases the training time.|

> [!NOTE]
>
> `AscendIndexIVFConfig` inherits from <a href="../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig">`AscendIndexConfig`</a>.

## `AscendIndexIVFConfig`<a name="en-us_TOPIC_0000001506334629"></a>

<a name="table1319620316150"></a>
<table><tbody><tr id="row19196173161512"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p8196736151"><a name="p8196736151"></a><a name="p8196736151"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p91961331157"><a name="p91961331157"></a><a name="p91961331157"></a><code>inline AscendIndexIVFConfig();</code></p>
</td>
</tr>
<tr id="row519612310152"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p131967381517"><a name="p131967381517"></a><a name="p131967381517"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p319616311155"><a name="p319616311155"></a><a name="p319616311155"></a>Default constructor. The default value of <code>devices</code> is <code>{0}</code>, which uses the 0th Ascend AI Processor for computation. The default value of <code>resources</code> is <code>128 MB</code>. The default value of <code>useKmeansPP</code> is <code>false</code>.</p>
</td>
</tr>
<tr id="row191967381510"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1719683121520"><a name="p1719683121520"></a><a name="p1719683121520"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p31961435151"><a name="p31961435151"></a><a name="p31961435151"></a>None</p>
</td>
</tr>
<tr id="row191966331518"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1719613141519"><a name="p1719613141519"></a><a name="p1719613141519"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1319615331515"><a name="p1319615331515"></a><a name="p1319615331515"></a>None</p>
</td>
</tr>
<tr id="row1019673161519"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p14196173191518"><a name="p14196173191518"></a><a name="p14196173191518"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p719603201510"><a name="p719603201510"></a><a name="p719603201510"></a>None</p>
</td>
</tr>
<tr id="row519633171513"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2197163171510"><a name="p2197163171510"></a><a name="p2197163171510"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4197143151510"><a name="p4197143151510"></a><a name="p4197143151510"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table3725347611"></a>
<table><tbody><tr id="row137251141265"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1372544561"><a name="p1372544561"></a><a name="p1372544561"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4706533988"><a name="p4706533988"></a><a name="p4706533988"></a><code>inline AscendIndexIVFConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row0725941369"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p87251143611"><a name="p87251143611"></a><a name="p87251143611"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendIndexIVFConfig</code>. It creates <code>AscendIndexIVFConfig</code>, sets the Device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, configures the memory pool size, and sets the default number of iterations.</p>
</td>
</tr>
<tr id="row872516411614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p157251441762"><a name="p157251441762"></a><a name="p157251441762"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1172515411612"><a name="p1172515411612"></a><a name="p1172515411612"></a><strong id="b74801235171213"><a name="b74801235171213"></a><a name="b74801235171213"></a><code>std::initializer_list&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b851894117126"><a name="b851894117126"></a><a name="b851894117126"></a><code>int64_t resourceSize</code></strong>: Preset memory pool size on the Device side, in bytes. This memory space stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the search batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to 1024 MB.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5591115071213"></a><a name="ul5591115071213"></a><ul id="ul5591115071213"><li><code>devices</code> must contain valid, non-duplicated device IDs, and the maximum number is <code>64</code>.</li><li>The configured <code>resourceSize</code> cannot exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When it is set to <code>-1</code>, the Device-side Ascend AI Processor resource configuration uses the default value of <code>128 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table745471811619"></a>
<table><tbody><tr id="row445418187618"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p145417181561"><a name="p145417181561"></a><a name="p145417181561"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172151146898"><a name="p172151146898"></a><a name="p172151146898"></a><code>inline AscendIndexIVFConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = IVF_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row845519181169"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p164551418362"><a name="p164551418362"></a><a name="p164551418362"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1445513182614"><a name="p1445513182614"></a><a name="p1445513182614"></a>Constructor of <code>AscendIndexIVFConfig</code>. It creates <code>AscendIndexIVFConfig</code>, sets the Device-side Ascend AI Processor resources according to the values configured in <code>devices</code>, configures the memory pool size, and sets the default number of iterations.</p>
</td>
</tr>
<tr id="row845512181667"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14551718264"><a name="p14551718264"></a><a name="p14551718264"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1945571813613"><a name="p1945571813613"></a><a name="p1945571813613"></a><strong id="b9403131414155"><a name="b9403131414155"></a><a name="b9403131414155"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs.</p>
<p id="p11455318966"><a name="p11455318966"></a><a name="p11455318966"></a><strong id="b132471122150"><a name="b132471122150"></a><a name="b132471122150"></a><code>int resourceSize</code></strong>: Preset memory pool size on the Device side, in bytes. This memory space stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default parameter is <code>IVF_DEFAULT_MEM</code> in the header file. This parameter is determined jointly by the base library size and the search batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to 1024 MB.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul9168241111516"></a><a name="ul9168241111516"></a><ul id="ul9168241111516"><li><code>devices</code> must contain valid, non-duplicated device IDs, and the maximum number is <code>64</code>.</li><li>The configured <code>resourceSize</code> cannot exceed <code>10 * 1024 MB</code> (<code>10 * 1024 * 1024 * 1024</code> bytes). When it is set to <code>-1</code>, the Device-side Ascend AI Processor resource configuration uses the default value of <code>128 MB</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SetDefaultClusteringConfig`<a name="en-us_TOPIC_0000001506495669"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p172151146898"><a name="p172151146898"></a><a name="p172151146898"></a><code>inline void SetDefaultClusteringConfig();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7535131221216"><a name="p7535131221216"></a><a name="p7535131221216"></a>Sets the number of iterations for <code>AscendIndexIVF</code> to the default value <code>10</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a>None</p>
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
