# AscendIndexInt8<a id="ZH-CN_TOPIC_0000001506495841"></a>

## 功能介绍<a id="ZH-CN_TOPIC_0000001506495913"></a>

AscendIndexInt8作为特征检索组件中的采用INT8特征向量的Index的基类，为特征检索中的其他INT8的Index定义接口。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## add接口<a name="ZH-CN_TOPIC_0000001506334825"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>void add(idx_t n, const int8_t *x);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>向AscendIndexInt8底库中添加新的特征向量。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b1235119155343"><a name="b1235119155343"></a><a name="b1235119155343"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b6839817183412"><a name="b6839817183412"></a><a name="b6839817183412"></a>const int8_t *x</strong>：添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul9411102014349"></a><a name="ul9411102014349"></a><ul id="ul9411102014349"><li>此处指针<span class="parmname" id="parmname119561922183411"><a name="parmname119561922183411"></a><a name="parmname119561922183411"></a>“x”</span>的长度应该为dims * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>底库向量总数的取值范围：0 &lt; n &lt; 1e9。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table6211414109"></a>
<table><tbody><tr id="row19219141603"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p132101414015"><a name="p132101414015"></a><a name="p132101414015"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1821101419018"><a name="p1821101419018"></a><a name="p1821101419018"></a>void add(idx_t n, const char *x);</p>
</td>
</tr>
<tr id="row02111141013"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p152212147010"><a name="p152212147010"></a><a name="p152212147010"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p222191411013"><a name="p222191411013"></a><a name="p222191411013"></a>向AscendIndexInt8底库中添加新的特征向量。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
</td>
</tr>
<tr id="row11224141604"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p7221714203"><a name="p7221714203"></a><a name="p7221714203"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p11221148019"><a name="p11221148019"></a><a name="p11221148019"></a><strong id="b1416416257368"><a name="b1416416257368"></a><a name="b1416416257368"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p022151410016"><a name="p022151410016"></a><a name="p022151410016"></a><strong id="b5663162723611"><a name="b5663162723611"></a><a name="b5663162723611"></a>const char *x</strong>：添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row122251416018"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p162281415018"><a name="p162281415018"></a><a name="p162281415018"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p1922171411011"><a name="p1922171411011"></a><a name="p1922171411011"></a>无</p>
</td>
</tr>
<tr id="row3225141020"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p52214145020"><a name="p52214145020"></a><a name="p52214145020"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1222614206"><a name="p1222614206"></a><a name="p1222614206"></a>无</p>
</td>
</tr>
<tr id="row1922131418013"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p16221814502"><a name="p16221814502"></a><a name="p16221814502"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul13290113043617"></a><a name="ul13290113043617"></a><ul id="ul13290113043617"><li>此处指针<span class="parmname" id="parmname10462153319369"><a name="parmname10462153319369"></a><a name="parmname10462153319369"></a>“x”</span>的长度应该为dims * <strong id="b145561227144114"><a name="b145561227144114"></a><a name="b145561227144114"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>底库向量总数的取值范围：0 &lt; n &lt; 1e9。</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。

## add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001506614905"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p95747912314"><a name="p95747912314"></a><a name="p95747912314"></a>void add_with_ids(idx_t n, const int8_t *x, const idx_t *ids);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>向AscendIndexInt8底库中添加新的特征向量，且指定特征ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b1352464163513"><a name="b1352464163513"></a><a name="b1352464163513"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b137614417358"><a name="b137614417358"></a><a name="b137614417358"></a>const int8_t *x</strong>：添加进底库的特征向量。</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b21165467359"><a name="b21165467359"></a><a name="b21165467359"></a>const idx_t *ids</strong>：添加进底库的特征向量ID。ID在Index实例中需唯一。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul6110252163513"></a><a name="ul6110252163513"></a><ul id="ul6110252163513"><li>此处指针<span class="parmname" id="parmname35061556354"><a name="parmname35061556354"></a><a name="parmname35061556354"></a>“x”</span>的长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，指针ids的长度应该为<span class="parmname" id="parmname16225814153615"><a name="parmname16225814153615"></a><a name="parmname16225814153615"></a>“n”</span>，否则可能出现越界读写错误并引起程序崩溃。</li><li>底库向量总数的取值范围：0 &lt; n &lt; 1e9。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table38814511704"></a>
<table><tbody><tr id="row138812511016"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p178817511000"><a name="p178817511000"></a><a name="p178817511000"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p19881351509"><a name="p19881351509"></a><a name="p19881351509"></a>void add_with_ids(idx_t n, const char *x, const idx_t *ids);</p>
</td>
</tr>
<tr id="row88855119016"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p10885519011"><a name="p10885519011"></a><a name="p10885519011"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p72832460516"><a name="p72832460516"></a><a name="p72832460516"></a>向AscendIndexInt8底库中添加新的特征向量，且指定特征ID。</p>
</td>
</tr>
<tr id="row88885115010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p17881515011"><a name="p17881515011"></a><a name="p17881515011"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p6882511707"><a name="p6882511707"></a><a name="p6882511707"></a><strong id="b19626747203618"><a name="b19626747203618"></a><a name="b19626747203618"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p168916512008"><a name="p168916512008"></a><a name="p168916512008"></a><strong id="b20455154923610"><a name="b20455154923610"></a><a name="b20455154923610"></a>const char *x</strong>：添加进底库的特征向量。</p>
<p id="p16897517012"><a name="p16897517012"></a><a name="p16897517012"></a><strong id="b587415312369"><a name="b587415312369"></a><a name="b587415312369"></a>const idx_t *ids</strong>：添加进底库的特征向量对应的ID。ID在Index实例中需唯一。</p>
</td>
</tr>
<tr id="row6895513016"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p689195119019"><a name="p689195119019"></a><a name="p689195119019"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p118915120011"><a name="p118915120011"></a><a name="p118915120011"></a>无</p>
</td>
</tr>
<tr id="row1689551609"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p7893511403"><a name="p7893511403"></a><a name="p7893511403"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p3896511015"><a name="p3896511015"></a><a name="p3896511015"></a>无</p>
</td>
</tr>
<tr id="row18915120017"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p7898511307"><a name="p7898511307"></a><a name="p7898511307"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul61401059163611"></a><a name="ul61401059163611"></a><ul id="ul61401059163611"><li>此处指针<span class="parmname" id="parmname926212153720"><a name="parmname926212153720"></a><a name="parmname926212153720"></a>“x”</span>的长度应该为dims * <strong id="b7891551107"><a name="b7891551107"></a><a name="b7891551107"></a>n</strong>，指针<span class="parmname" id="parmname1265763411412"><a name="parmname1265763411412"></a><a name="parmname1265763411412"></a>“ids”</span>的长度应该为<span class="parmname" id="parmname12601627193614"><a name="parmname12601627193614"></a><a name="parmname12601627193614"></a>“n”</span>，否则可能出现越界读写错误并引起程序崩溃。</li><li>底库向量总数的取值范围：0 &lt; n &lt; 1e9。</li></ul>
</td>
</tr>
</tbody>
</table>

## assign接口<a name="ZH-CN_TOPIC_0000001506495721"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p63341832173418"><a name="p63341832173418"></a><a name="p63341832173418"></a>void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndexInt8特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname2876185784115"><a name="parmname2876185784115"></a><a name="parmname2876185784115"></a>“k”</span>条特征的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><strong id="b115116618384"><a name="b115116618384"></a><a name="b115116618384"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><strong id="b5752118123813"><a name="b5752118123813"></a><a name="b5752118123813"></a>const int8_t *x</strong>：特征向量数据。</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><strong id="b2801121053818"><a name="b2801121053818"></a><a name="b2801121053818"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b84731813143813"><a name="b84731813143813"></a><a name="b84731813143813"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname52444018429"><a name="parmname52444018429"></a><a name="parmname52444018429"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul3740916183818"></a><a name="ul3740916183818"></a><ul id="ul3740916183818"><li>查询的特征向量数据<span class="parmname" id="parmname186421632182120"><a name="parmname186421632182120"></a><a name="parmname186421632182120"></a>“x”</span>的长度应符合dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，<span class="parmname" id="parmname1553510216382"><a name="parmname1553510216382"></a><a name="parmname1553510216382"></a>“labels”</span>的长度应符合<strong id="b63841945103710"><a name="b63841945103710"></a><a name="b63841945103710"></a>k</strong> * <strong id="b645545116371"><a name="b645545116371"></a><a name="b645545116371"></a>n</strong>，否则可能会出现越界读写的情况，引起程序的崩溃。</li><li>此处<span class="parmname" id="parmname5729115673612"><a name="parmname5729115673612"></a><a name="parmname5729115673612"></a>“n”</span>大于0且小于1e9。</li><li>此处<span class="parmname" id="parmname14960158143612"><a name="parmname14960158143612"></a><a name="parmname14960158143612"></a>“k”</span>大于0且小于等于4096。</li><li>此处<strong id="b1996519313814"><a name="b1996519313814"></a><a name="b1996519313814"></a>n</strong> * <strong id="b9489131323819"><a name="b9489131323819"></a><a name="b9489131323819"></a>k</strong>小于1e10。</li></ul>
</td>
</tr>
</tbody>
</table>

## AscendIndexInt8接口<a name="ZH-CN_TOPIC_0000001506614993"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p156319619286"><a name="p156319619286"></a><a name="p156319619286"></a>AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8的构造函数，生成维度为dims的AscendIndexInt8（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="parmname21731230154119"><a name="parmname21731230154119"></a><a name="parmname21731230154119"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b161951846142616"><a name="b161951846142616"></a><a name="b161951846142616"></a>int dims</strong>：AscendIndexInt8管理的一组特征向量的维度。</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b125311249162612"><a name="b125311249162612"></a><a name="b125311249162612"></a>faiss::MetricType metric</strong>：AscendIndexInt8在执行特征向量相似度检索的时候使用的距离度量类型，当前支持<span class="parmvalue" id="parmvalue1285217010278"><a name="parmvalue1285217010278"></a><a name="parmvalue1285217010278"></a>“faiss::MetricType::METRIC_L2”</span>和<span class="parmvalue" id="parmvalue14081843271"><a name="parmvalue14081843271"></a><a name="parmvalue14081843271"></a>“faiss::MetricType::METRIC_INNER_PRODUCT”</span>。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b64712535265"><a name="b64712535265"></a><a name="b64712535265"></a>AscendIndexInt8Config config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname915112379369"><a name="parmname915112379369"></a><a name="parmname915112379369"></a>“dims”</span>为不小于64，不大于1024的整数，且需要能被64整除。</p>
</td>
</tr>
</tbody>
</table>

<a name="table103312407520"></a>
<table><tbody><tr id="row9331540657"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p933940155"><a name="p933940155"></a><a name="p933940155"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8(const AscendIndexInt8&amp;) = delete;</p>
</td>
</tr>
<tr id="row163311401851"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p13311408510"><a name="p13311408510"></a><a name="p13311408510"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1233154015517"><a name="p1233154015517"></a><a name="p1233154015517"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row203364017512"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p53304012513"><a name="p53304012513"></a><a name="p53304012513"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b1690325711272"><a name="b1690325711272"></a><a name="b1690325711272"></a>const AscendIndexInt8&amp;</strong>：AscendIndexInt8对象。</p>
</td>
</tr>
<tr id="row33318406512"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1833840458"><a name="p1833840458"></a><a name="p1833840458"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p9331140556"><a name="p9331140556"></a><a name="p9331140556"></a>无</p>
</td>
</tr>
<tr id="row1533740858"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p733184020518"><a name="p733184020518"></a><a name="p733184020518"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p93315408518"><a name="p93315408518"></a><a name="p93315408518"></a>无</p>
</td>
</tr>
<tr id="row7339405511"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p143319401358"><a name="p143319401358"></a><a name="p143319401358"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p163315401851"><a name="p163315401851"></a><a name="p163315401851"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table1882220715614"></a>
<table><tbody><tr id="row282214719612"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p10822167367"><a name="p10822167367"></a><a name="p10822167367"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexInt8();</p>
</td>
</tr>
<tr id="row128221171266"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p98221971361"><a name="p98221971361"></a><a name="p98221971361"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p19822271613"><a name="p19822271613"></a><a name="p19822271613"></a>AscendIndexInt8的析构函数，销毁AscendIndexInt8对象，释放资源。</p>
</td>
</tr>
<tr id="row2082217362"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p138221771067"><a name="p138221771067"></a><a name="p138221771067"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row15822977619"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1682212712617"><a name="p1682212712617"></a><a name="p1682212712617"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18221278617"><a name="p18221278617"></a><a name="p18221278617"></a>无</p>
</td>
</tr>
<tr id="row382247166"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p88221378615"><a name="p88221378615"></a><a name="p88221378615"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p682214717616"><a name="p682214717616"></a><a name="p682214717616"></a>无</p>
</td>
</tr>
<tr id="row198221076614"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p38221974612"><a name="p38221974612"></a><a name="p38221974612"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p148221171060"><a name="p148221171060"></a><a name="p148221171060"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getDeviceList接口<a name="ZH-CN_TOPIC_0000001672982421"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>std::vector&lt;int&gt; getDeviceList() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>返回Index中管理的Device<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>设置，交由子类继承并实现，在本类中不提供相应的实现，仅会返回一个空<strong id="b7815174613297"><a name="b7815174613297"></a><a name="b7815174613297"></a>vector&lt;int&gt;</strong>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>Index中管理的Device<span id="ph5275135316438"><a name="ph5275135316438"></a><a name="ph5275135316438"></a>昇腾AI处理器</span>设置。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getDim接口<a name="ZH-CN_TOPIC_0000001690599922"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p192220207614"><a name="p192220207614"></a><a name="p192220207614"></a>int getDim() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3221620764"><a name="p3221620764"></a><a name="p3221620764"></a>获取AscendIndexInt8管理的一组特征向量的维度。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8220620861"><a name="p8220620861"></a><a name="p8220620861"></a>AscendIndexInt8管理的一组特征向量的维度。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p172160208615"><a name="p172160208615"></a><a name="p172160208615"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getNTotal接口<a name="ZH-CN_TOPIC_0000001738718517"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p192220207614"><a name="p192220207614"></a><a name="p192220207614"></a>faiss::idx_t getNTotal() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3221620764"><a name="p3221620764"></a><a name="p3221620764"></a>获取AscendIndexInt8已添加进底库的特征向量数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p45014471818"><a name="p45014471818"></a><a name="p45014471818"></a>AscendIndexInt8已添加进底库的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p172160208615"><a name="p172160208615"></a><a name="p172160208615"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getMetricType接口<a name="ZH-CN_TOPIC_0000001738678653"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58591491393"><a name="p58591491393"></a><a name="p58591491393"></a>faiss::MetricType getMetricType() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18599491290"><a name="p18599491290"></a><a name="p18599491290"></a>获取AscendIndexInt8执行特征向量相似度检索的时候使用的距离度量类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1685710491096"><a name="p1685710491096"></a><a name="p1685710491096"></a>返回AscendIndexInt8执行特征向量相似度检索的时候使用的距离度量类型。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p785724919915"><a name="p785724919915"></a><a name="p785724919915"></a>无</p>
</td>
</tr>
</tbody>
</table>

## isTrained接口<a name="ZH-CN_TOPIC_0000001690759666"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p79107141490"><a name="p79107141490"></a><a name="p79107141490"></a>bool isTrained() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5909201417911"><a name="p5909201417911"></a><a name="p5909201417911"></a>判断AscendIndexInt8是否已训练。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p42211620264"><a name="p42211620264"></a><a name="p42211620264"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p192200201266"><a name="p192200201266"></a><a name="p192200201266"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1790810141395"><a name="p1790810141395"></a><a name="p1790810141395"></a>AscendIndexInt8已训练状态，<span class="parmvalue" id="parmvalue1069142914134"><a name="parmvalue1069142914134"></a><a name="parmvalue1069142914134"></a>“true”</span>表示已训练，<span class="parmvalue" id="parmvalue11991931181313"><a name="parmvalue11991931181313"></a><a name="parmvalue11991931181313"></a>“false”</span>表示未训练。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p129074145914"><a name="p129074145914"></a><a name="p129074145914"></a>无</p>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506414841"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8&amp; operator=(const AscendIndexInt8&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b11871311182818"><a name="b11871311182818"></a><a name="b11871311182818"></a>const AscendIndexInt8&amp;</strong>：常量AscendIndexInt8。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## reclaimMemory接口<a name="ZH-CN_TOPIC_0000001506615133"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>virtual size_t reclaimMemory();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>基类中定义的虚函数，具体描述参考子类。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16621234163"><a name="p16621234163"></a><a name="p16621234163"></a>无</p>
</td>
</tr>
</tbody>
</table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001456695088"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>size_t remove_ids(const faiss::IDSelector &amp;sel);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndexInt8删除底库中指定的特征向量的接口。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.3.1 "><p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b832314302375"><a name="b832314302375"></a><a name="b832314302375"></a>const faiss::IDSelector &amp;sel</strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>返回被删除的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.75%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.25%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## reserveMemory接口<a name="ZH-CN_TOPIC_0000001506615065"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>virtual void reserveMemory(size_t numVecs);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>基类中定义的虚函数，具体描述参考子类。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a><strong id="b15426142311418"><a name="b15426142311418"></a><a name="b15426142311418"></a>size_t numVecs</strong>：申请预留内存的底库数量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6587140131617"><a name="p6587140131617"></a><a name="p6587140131617"></a>无</p>
</td>
</tr>
</tbody>
</table>

## search接口<a name="ZH-CN_TOPIC_0000001506414889"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p17821720121118"><a name="p17821720121118"></a><a name="p17821720121118"></a>void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndexInt8特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname26821757174214"><a name="parmname26821757174214"></a><a name="parmname26821757174214"></a>“k”</span>条特征的距离及ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><strong id="b164231528153919"><a name="b164231528153919"></a><a name="b164231528153919"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><strong id="b819943111393"><a name="b819943111393"></a><a name="b819943111393"></a>const int8_t *x</strong>：特征向量数据。</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><strong id="b1388123318393"><a name="b1388123318393"></a><a name="b1388123318393"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b7329163673918"><a name="b7329163673918"></a><a name="b7329163673918"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname459815174213"><a name="parmname459815174213"></a><a name="parmname459815174213"></a>“k”</span>个向量间的距离值。当有效的检索结果不足<span class="parmname" id="parmname177121349488"><a name="parmname177121349488"></a><a name="parmname177121349488"></a>“k”</span>个时，剩余无效距离用65504或-65504填充（因metric而异）。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b736913816394"><a name="b736913816394"></a><a name="b736913816394"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname12191354134220"><a name="parmname12191354134220"></a><a name="parmname12191354134220"></a>“k”</span>个向量的ID。当有效的检索结果不足<span class="parmname" id="parmname108767350487"><a name="parmname108767350487"></a><a name="parmname108767350487"></a>“k”</span>个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul4165643183912"></a><a name="ul4165643183912"></a><ul id="ul4165643183912"><li>查询的特征向量数据<span class="parmname" id="parmname141916404213"><a name="parmname141916404213"></a><a name="parmname141916404213"></a>“x”</span>的长度应该为dims * <strong id="b1511244182819"><a name="b1511244182819"></a><a name="b1511244182819"></a>n</strong>，<span class="parmname" id="parmname674185217398"><a name="parmname674185217398"></a><a name="parmname674185217398"></a>“distances”</span>以及<span class="parmname" id="parmname3790185653920"><a name="parmname3790185653920"></a><a name="parmname3790185653920"></a>“labels”</span>的长度应该为k * <strong id="b179013662819"><a name="b179013662819"></a><a name="b179013662819"></a>n</strong>，否则可能会出现越界读写的情况，引起程序的崩溃。</li><li>此处<span class="parmname" id="parmname7264111613717"><a name="parmname7264111613717"></a><a name="parmname7264111613717"></a>“n”</span>大于0且小于1e9。</li><li>此处<span class="parmname" id="parmname464319193712"><a name="parmname464319193712"></a><a name="parmname464319193712"></a>“k”</span>大于0且小于等于4096。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table88671631181418"></a>
<table><tbody><tr id="row6867133191414"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p14867153118141"><a name="p14867153118141"></a><a name="p14867153118141"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p28671831101414"><a name="p28671831101414"></a><a name="p28671831101414"></a>void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;</p>
</td>
</tr>
<tr id="row8867631151417"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1086733161419"><a name="p1086733161419"></a><a name="p1086733161419"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p10867231161419"><a name="p10867231161419"></a><a name="p10867231161419"></a>实现AscendIndexInt8特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname146430814314"><a name="parmname146430814314"></a><a name="parmname146430814314"></a>“k”</span>条特征的距离及ID。</p>
</td>
</tr>
<tr id="row1686713131418"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p286783116148"><a name="p286783116148"></a><a name="p286783116148"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p13867203110149"><a name="p13867203110149"></a><a name="p13867203110149"></a><strong id="b12172172224011"><a name="b12172172224011"></a><a name="b12172172224011"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p11867173116148"><a name="p11867173116148"></a><a name="p11867173116148"></a><strong id="b18868152434011"><a name="b18868152434011"></a><a name="b18868152434011"></a>const char *x</strong>：特征向量数据。</p>
<p id="p20867031131410"><a name="p20867031131410"></a><a name="p20867031131410"></a><strong id="b146561626194012"><a name="b146561626194012"></a><a name="b146561626194012"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
</td>
</tr>
<tr id="row188673319140"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p14867193115144"><a name="p14867193115144"></a><a name="p14867193115144"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p2086715317143"><a name="p2086715317143"></a><a name="p2086715317143"></a><strong id="b61081329164012"><a name="b61081329164012"></a><a name="b61081329164012"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname99271330132811"><a name="parmname99271330132811"></a><a name="parmname99271330132811"></a>“k”</span>个向量间的距离值。</p>
<p id="p88672310144"><a name="p88672310144"></a><a name="p88672310144"></a><strong id="b1355816302407"><a name="b1355816302407"></a><a name="b1355816302407"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname178171214104312"><a name="parmname178171214104312"></a><a name="parmname178171214104312"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row1786719315149"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p08674311147"><a name="p08674311147"></a><a name="p08674311147"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p886710319142"><a name="p886710319142"></a><a name="p886710319142"></a>无。</p>
</td>
</tr>
<tr id="row11867231121415"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p16867931161410"><a name="p16867931161410"></a><a name="p16867931161410"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul176971234124016"></a><a name="ul176971234124016"></a><ul id="ul176971234124016"><li>查询的特征向量数据<span class="parmname" id="parmname795513441401"><a name="parmname795513441401"></a><a name="parmname795513441401"></a>“x”</span>的长度应该为dims * <strong id="b19129917182820"><a name="b19129917182820"></a><a name="b19129917182820"></a>n</strong>，<span class="parmname" id="parmname18850636144019"><a name="parmname18850636144019"></a><a name="parmname18850636144019"></a>“distances”</span>以及<span class="parmname" id="parmname48111338114012"><a name="parmname48111338114012"></a><a name="parmname48111338114012"></a>“labels”</span>的长度应该为k * <strong id="b17501722102811"><a name="b17501722102811"></a><a name="b17501722102811"></a>n</strong>，否则可能会出现越界读写的情况，引起程序的崩溃。</li><li>此处<span class="parmname" id="parmname886733191414"><a name="parmname886733191414"></a><a name="parmname886733191414"></a>“n”</span>大于0且小于1e9。</li><li>此处<span class="parmname" id="parmname1986743111146"><a name="parmname1986743111146"></a><a name="parmname1986743111146"></a>“k”</span>大于0且小于等于4096。</li></ul>
</td>
</tr>
</tbody>
</table>

## train接口<a name="ZH-CN_TOPIC_0000001456534956"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>virtual void train(idx_t n, const int8_t *x);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>基类中定义的虚函数，具体描述参考子类。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b19388103111281"><a name="b19388103111281"></a><a name="b19388103111281"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b2560133392817"><a name="b2560133392817"></a><a name="b2560133392817"></a>const int8_t *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p255165217139"><a name="p255165217139"></a><a name="p255165217139"></a>无</p>
</td>
</tr>
</tbody>
</table>

## updateCentroids接口<a name="ZH-CN_TOPIC_0000001506414833"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p107821866408"><a name="p107821866408"></a><a name="p107821866408"></a>virtual void updateCentroids(idx_t n, const int8_t *x);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p283153493915"><a name="p283153493915"></a><a name="p283153493915"></a>基类中定义的虚函数，具体描述参考子类。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b10732151473012"><a name="b10732151473012"></a><a name="b10732151473012"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b1196218162309"><a name="b1196218162309"></a><a name="b1196218162309"></a>const int8_t *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p211213924718"><a name="p211213924718"></a><a name="p211213924718"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table2023134918146"></a>
<table><tbody><tr id="row5231649201420"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p4233499147"><a name="p4233499147"></a><a name="p4233499147"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p18234493149"><a name="p18234493149"></a><a name="p18234493149"></a>virtual void updateCentroids(idx_t n, const char *x);</p>
</td>
</tr>
<tr id="row7232497144"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p8238490147"><a name="p8238490147"></a><a name="p8238490147"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p105115579251"><a name="p105115579251"></a><a name="p105115579251"></a>基类中定义的虚函数，具体描述参考子类。</p>
</td>
</tr>
<tr id="row1023164911414"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p1723349161412"><a name="p1723349161412"></a><a name="p1723349161412"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p323149141417"><a name="p323149141417"></a><a name="p323149141417"></a><strong id="b1824862973311"><a name="b1824862973311"></a><a name="b1824862973311"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p16231949131414"><a name="p16231949131414"></a><a name="p16231949131414"></a><strong id="b1020914312339"><a name="b1020914312339"></a><a name="b1020914312339"></a>const char *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row1231749111414"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p1723114911419"><a name="p1723114911419"></a><a name="p1723114911419"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p323449121419"><a name="p323449121419"></a><a name="p323449121419"></a>无</p>
</td>
</tr>
<tr id="row72374910141"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p32311490142"><a name="p32311490142"></a><a name="p32311490142"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p82354971417"><a name="p82354971417"></a><a name="p82354971417"></a>无</p>
</td>
</tr>
<tr id="row11230494140"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p1923124951413"><a name="p1923124951413"></a><a name="p1923124951413"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p157561925193210"><a name="p157561925193210"></a><a name="p157561925193210"></a>无</p>
</td>
</tr>
</tbody>
</table>
