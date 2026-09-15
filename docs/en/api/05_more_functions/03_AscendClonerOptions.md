# `AscendClonerOptions`<a name="en-us_TOPIC_0000001456854804"></a>

## Function Description<a name="en-us_TOPIC_0000001456535196"></a>

Configuration parameters for the `AscendCloner` interface.

**Members<a name="section1372191465013"></a>**

| Member | Type | Description |
|--|--|--|
| reserveVecs | long | Currently unused. Number of features reserved in memory. |
| verbose | bool | Whether to print copy logs. |
| resourceSize | int64_t | Resource pool size. |
| slim | bool | Member variable of `AscendClonerOptions`. Whether to dynamically increase memory. The default value is `false`. |
| filterable | bool | Member variable of `AscendClonerOptions`. Whether to filter by ID. The default value is `false`. |
| indexMode | uint32_t | Index INT8 retrieval mode. The default value is `0` (`DEFAULT_MODE`). |
| blockSize | uint32_t | `blockSize` configured on the device side. The default value of `BLOCK_SIZE` is `16384 * 16 = 262144`. |

## `AscendClonerOptions`<a name="en-us_TOPIC_0000001506414885"></a>

<a name="en-us_topic_0000001340833369_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001340833369_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001340833369_p12559123810"><a name="en-us_topic_0000001340833369_p12559123810"></a><a name="en-us_topic_0000001340833369_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001340833369_p1212917467412"><a name="en-us_topic_0000001340833369_p1212917467412"></a><a name="en-us_topic_0000001340833369_p1212917467412"></a><code>AscendClonerOptions()</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001340833369_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001340833369_p1212599383"><a name="en-us_topic_0000001340833369_p1212599383"></a><a name="en-us_topic_0000001340833369_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001340833369_p131714208358"><a name="en-us_topic_0000001340833369_p131714208358"></a><a name="en-us_topic_0000001340833369_p131714208358"></a>Constructor of <code>AscendClonerOptions</code>.</p>
</td>
</tr>
<tr id="en-us_topic_0000001340833369_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001340833369_p112195910383"><a name="en-us_topic_0000001340833369_p112195910383"></a><a name="en-us_topic_0000001340833369_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001340833369_p1745111429517"><a name="en-us_topic_0000001340833369_p1745111429517"></a><a name="en-us_topic_0000001340833369_p1745111429517"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001340833369_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001340833369_p17235973820"><a name="en-us_topic_0000001340833369_p17235973820"></a><a name="en-us_topic_0000001340833369_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001340833369_p6295973819"><a name="en-us_topic_0000001340833369_p6295973819"></a><a name="en-us_topic_0000001340833369_p6295973819"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001340833369_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001340833369_p182459113812"><a name="en-us_topic_0000001340833369_p182459113812"></a><a name="en-us_topic_0000001340833369_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001340833369_p912917864516"><a name="en-us_topic_0000001340833369_p912917864516"></a><a name="en-us_topic_0000001340833369_p912917864516"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001340833369_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001340833369_p423590386"><a name="en-us_topic_0000001340833369_p423590386"></a><a name="en-us_topic_0000001340833369_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001340833369_p3947640162619"><a name="en-us_topic_0000001340833369_p3947640162619"></a><a name="en-us_topic_0000001340833369_p3947640162619"></a>None</p>
</td>
</tr>
</tbody>
</table>
