# `AscendIndexBinaryFlatConfig`<a name="en-us_TOPIC_0000001506495777"></a>

`AscendIndexBinaryFlat` uses the corresponding `AscendIndexBinaryFlatConfig` to initialize the corresponding resources and configure the Device-side hardware resources `devices` and the preset memory pool size `resources` during retrieval.

- `AscendIndexBinaryFlat` supports only <term>Atlas Inference Series products</term> with a single Ascend AI Processor. It depends on the AICPU operator and the BinaryFlat operator. See <a href="../../05_user_guide.md#custom-operator-introduction">Introduction to Custom Operators</a> to generate the corresponding operators.
- `AscendIndexBinaryFlat` supports only standard deployment mode.

**Members<a name="section1372191465013"></a>**

|Member|Type|Description|
|--|--|--|
|deviceList|`std::vector\<int>`|Device-side device IDs. The `AscendIndexBinaryFlat` class supports only a single accelerator card of the <term>Atlas Inference Series products</term>.|
|resourceSize|`int64_t`|Size of the Device-side memory pool, in bytes. The default value is `1024 MB`. The valid range is `[1024*1024*1024, 32*1024*1024*1024]`. For a base library with 10 million vectors, `5 GB` is recommended.|

**API Description<a name="section108610580175"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1551916503464"><a name="p1551916503464"></a><a name="p1551916503464"></a><code>AscendIndexBinaryFlatConfig() = default;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Default constructor. The default value of <code>devices</code> is <code>{ 0 }</code>, which uses the 0th Ascend AI Processor for computation. The default value of <code>resources</code> is <code>1024 MB</code>.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><code>AscendIndexBinaryFlat</code> supports only <term>Atlas Inference Series products</term> with a single Ascend AI Processor. If the 0th Ascend AI Processor is unavailable, you cannot use the default constructor.</p>
</td>
</tr>
</tbody>
</table>

<a name="table092314378186"></a>
<table><tbody><tr id="row6923173719182"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18923337131816"><a name="p18923337131816"></a><a name="p18923337131816"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3686859135116"><a name="p3686859135116"></a><a name="p3686859135116"></a><code>AscendIndexBinaryFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row1692315371180"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p19923153751814"><a name="p19923153751814"></a><a name="p19923153751814"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p14661924131015"><a name="p14661924131015"></a><a name="p14661924131015"></a>Constructor that uses <code>initializer_list</code> for <code>devices</code>.</p>
</td>
</tr>
<tr id="row092353751820"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1292333771814"><a name="p1292333771814"></a><a name="p1292333771814"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p192393741812"><a name="p192393741812"></a><a name="p192393741812"></a><strong id="b7745577018"><a name="b7745577018"></a><a name="b7745577018"></a><code>std::initializer_list&lt;int&gt; devices</code></strong>: Device-side device IDs. For this class, only a single device is supported, that is, the length of <code>devices</code> must be <code>1</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b120518131001"><a name="b120518131001"></a><a name="b120518131001"></a><code>int64_t resources</code></strong>: Preset memory pool size. The default value is <code>1024 MB</code>.</p>
</td>
</tr>
<tr id="row4923163715183"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1692313375186"><a name="p1692313375186"></a><a name="p1692313375186"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15923173719188"><a name="p15923173719188"></a><a name="p15923173719188"></a>None</p>
</td>
</tr>
<tr id="row392317375180"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p119239370187"><a name="p119239370187"></a><a name="p119239370187"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p109231372187"><a name="p109231372187"></a><a name="p109231372187"></a>None</p>
</td>
</tr>
<tr id="row119241937191814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p11924193715185"><a name="p11924193715185"></a><a name="p11924193715185"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5609290011"></a><a name="ul5609290011"></a><ul id="ul5609290011"><li><code>devices</code> must contain valid, non-duplicated device IDs, and the length must be <code>1</code>.</li><li>The valid range of <code>resources</code> is <code>[1024*1024*1024, 32*1024*1024*1024]</code>. For a 10 million base library, <code>5 GB</code> is recommended.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1743710521181"></a>
<table><tbody><tr id="row18437752161818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2437175281813"><a name="p2437175281813"></a><a name="p2437175281813"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19298152493214"><a name="p19298152493214"></a><a name="p19298152493214"></a><code>AscendIndexBinaryFlatConfig(std::vector&lt;int&gt; devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);</code></p>
</td>
</tr>
<tr id="row1243755211815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p15437452131811"><a name="p15437452131811"></a><a name="p15437452131811"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10437125271814"><a name="p10437125271814"></a><a name="p10437125271814"></a>Constructor that uses <code>vector</code> for <code>devices</code>.</p>
</td>
</tr>
<tr id="row843735251817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p8437152171817"><a name="p8437152171817"></a><a name="p8437152171817"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p74371152181812"><a name="p74371152181812"></a><a name="p74371152181812"></a><strong id="b76053103337"><a name="b76053103337"></a><a name="b76053103337"></a><code>std::vector&lt;int&gt; devices</code></strong>: Device-side device IDs. For this class, only a single device is supported, that is, the length of <code>devices</code> must be <code>1</code>.</p>
<p id="p843710522187"><a name="p843710522187"></a><a name="p843710522187"></a><strong id="b175341016133314"><a name="b175341016133314"></a><a name="b175341016133314"></a><code>int64_t resources</code></strong>: Preset memory pool size. The default value is <code>1024 MB</code>.</p>
</td>
</tr>
<tr id="row243775261813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9437452171817"><a name="p9437452171817"></a><a name="p9437452171817"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p4437352101816"><a name="p4437352101816"></a><a name="p4437352101816"></a>None</p>
</td>
</tr>
<tr id="row17437195218187"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p743785215182"><a name="p743785215182"></a><a name="p743785215182"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p15437852171815"><a name="p15437852171815"></a><a name="p15437852171815"></a>None</p>
</td>
</tr>
<tr id="row143717524181"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1743735241812"><a name="p1743735241812"></a><a name="p1743735241812"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul144378526181"></a><a name="ul144378526181"></a><ul id="ul144378526181"><li><code>devices</code> must contain valid, non-duplicated device IDs, and the length must be <code>1</code>.</li><li>The valid range of <code>resources</code> is <code>[1024*1024*1024, 32*1024*1024*1024]</code>. For a 10 million base library, <code>5 GB</code> is recommended.</li></ul>
</td>
</tr>
</tbody>
</table>
