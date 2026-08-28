# AscendClonerOptions<a name="ZH-CN_TOPIC_0000001456854804"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456535196"></a>

AscendCloner接口的配置参数。

**成员介绍<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="140" align="center" valign="middle">成员</td><td valign="middle">类型</td><td valign="middle">说明</td></tr>
<tr><td width="140" align="center" valign="middle">reserveVecs</td><td valign="middle">long</td><td valign="middle">当前无效，预留内存的特征数。</td></tr>
<tr><td width="140" align="center" valign="middle">verbose</td><td valign="middle">bool</td><td valign="middle">是否打印拷贝日志。</td></tr>
<tr><td width="140" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">slim</td><td valign="middle">bool</td><td valign="middle">AscendClonerOptions成员变量，是否动态增加内存。默认为false。</td></tr>
<tr><td width="140" align="center" valign="middle">filterable</td><td valign="middle">bool</td><td valign="middle">AscendClonerOptions成员变量，是否按照id进行过滤。默认为false。</td></tr>
<tr><td width="140" align="center" valign="middle">indexMode</td><td valign="middle">uint32_t</td><td valign="middle">Index int8检索模式，默认值为0 （DEFAULT_MODE）。</td></tr>
<tr><td width="140" align="center" valign="middle">blockSize</td><td valign="middle">uint32_t</td><td valign="middle">配置Device侧的blockSize，默认值“BLOCK_SIZE”为16384 * 16 = 262144。</td></tr>
</tbody></table>

## AscendClonerOptions接口<a name="ZH-CN_TOPIC_0000001506414885"></a>

<a name="zh-cn_topic_0000001340833369_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendClonerOptions()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendClonerOptions的构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
