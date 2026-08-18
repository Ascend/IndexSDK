# AscendClonerOptions<a name="ZH-CN_TOPIC_0000001456854804"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456535196"></a>

AscendCloner接口的配置参数。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|reserveVecs|long|当前无效，预留内存的特征数。|
|verbose|bool|是否打印拷贝日志。|
|resourceSize|int64_t|资源池大小。|
|slim|bool|AscendClonerOptions成员变量，是否动态增加内存。默认为false。|
|filterable|bool|AscendClonerOptions成员变量，是否按照id进行过滤。默认为false。|
|indexMode|uint32_t|Index int8检索模式，默认值为0 （DEFAULT_MODE）。|
|blockSize|uint32_t|配置Device侧的blockSize，默认值“BLOCK_SIZE”为16384 * 16 = 262144。|

## AscendClonerOptions接口<a name="ZH-CN_TOPIC_0000001506414885"></a>

<a name="zh-cn_topic_0000001340833369_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001340833369_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001340833369_p12559123810"><a name="zh-cn_topic_0000001340833369_p12559123810"></a><a name="zh-cn_topic_0000001340833369_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001340833369_p1212917467412"><a name="zh-cn_topic_0000001340833369_p1212917467412"></a><a name="zh-cn_topic_0000001340833369_p1212917467412"></a>AscendClonerOptions()</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001340833369_p1212599383"><a name="zh-cn_topic_0000001340833369_p1212599383"></a><a name="zh-cn_topic_0000001340833369_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001340833369_p131714208358"><a name="zh-cn_topic_0000001340833369_p131714208358"></a><a name="zh-cn_topic_0000001340833369_p131714208358"></a>AscendClonerOptions的构造函数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001340833369_p112195910383"><a name="zh-cn_topic_0000001340833369_p112195910383"></a><a name="zh-cn_topic_0000001340833369_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001340833369_p1745111429517"><a name="zh-cn_topic_0000001340833369_p1745111429517"></a><a name="zh-cn_topic_0000001340833369_p1745111429517"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001340833369_p17235973820"><a name="zh-cn_topic_0000001340833369_p17235973820"></a><a name="zh-cn_topic_0000001340833369_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001340833369_p6295973819"><a name="zh-cn_topic_0000001340833369_p6295973819"></a><a name="zh-cn_topic_0000001340833369_p6295973819"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001340833369_p182459113812"><a name="zh-cn_topic_0000001340833369_p182459113812"></a><a name="zh-cn_topic_0000001340833369_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001340833369_p912917864516"><a name="zh-cn_topic_0000001340833369_p912917864516"></a><a name="zh-cn_topic_0000001340833369_p912917864516"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001340833369_p423590386"><a name="zh-cn_topic_0000001340833369_p423590386"></a><a name="zh-cn_topic_0000001340833369_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001340833369_p3947640162619"><a name="zh-cn_topic_0000001340833369_p3947640162619"></a><a name="zh-cn_topic_0000001340833369_p3947640162619"></a>无</p>
</td>
</tr>
</tbody>
</table>
