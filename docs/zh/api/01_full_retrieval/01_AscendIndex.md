# AscendIndex<a id="ZH-CN_TOPIC_0000001456375304"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506414937"></a>

AscendIndex作为特征检索组件中的大部分检索的Index的基类，向上承接Faiss，向下为特征检索中的其他Index定义接口。

## add接口<a id="ZH-CN_TOPIC_0000001506614985"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p5684112753414"><a name="p5684112753414"></a><a name="p5684112753414"></a>void add(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndex建库和向底库中添加新的特征向量的功能。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b812832571217"><a name="b812832571217"></a><a name="b812832571217"></a>idx_t n</strong>：待添加进底库的特征向量数量。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b261412289122"><a name="b261412289122"></a><a name="b261412289122"></a>const float *x</strong>：待添加进底库的特征向量。</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p14372206704"><a name="p14372206704"></a><a name="p14372206704"></a>指针<span class="parmname" id="parmname197639415139"><a name="parmname197639415139"></a><a name="parmname197639415139"></a>“x”</span>的长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</p>
<p id="p967614571013"><a name="p967614571013"></a><a name="p967614571013"></a><span class="parmname" id="parmname31332191105"><a name="parmname31332191105"></a><a name="parmname31332191105"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</p>
<div class="note" id="note153615513612"><a name="note153615513612"></a><a name="note153615513612"></a><span class="notetitle">
说明： </span><div class="notebody"><a name="ul103685518369"></a><a name="ul103685518369"></a><ul id="ul103685518369"><li>add接口不能与add_with_ids接口混用。</li><li>使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add_with_ids接口。</li></ul>
</div></div>
</td>
</tr>
</tbody>
</table>

<a name="table17254342193617"></a>
<table><tbody><tr id="row1254164217362"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p225474203614"><a name="p225474203614"></a><a name="p225474203614"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p23521745171019"><a name="p23521745171019"></a><a name="p23521745171019"></a>void add(idx_t n, const uint16_t *x);</p>
</td>
</tr>
<tr id="row18254442183618"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p22541442163617"><a name="p22541442163617"></a><a name="p22541442163617"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p4352184531013"><a name="p4352184531013"></a><a name="p4352184531013"></a>实现AscendIndex建库和向底库中添加新的特征向量的功能。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
</td>
</tr>
<tr id="row7254184215362"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p1025415425363"><a name="p1025415425363"></a><a name="p1025415425363"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p9352184551012"><a name="p9352184551012"></a><a name="p9352184551012"></a><strong id="b8773101351111"><a name="b8773101351111"></a><a name="b8773101351111"></a>idx_t n</strong>：待添加进底库的特征向量数量。</p>
<p id="p1935220453102"><a name="p1935220453102"></a><a name="p1935220453102"></a><strong id="b186341618114"><a name="b186341618114"></a><a name="b186341618114"></a>const uint16_t *x</strong>：待添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row5254194273613"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p19254542133611"><a name="p19254542133611"></a><a name="p19254542133611"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p14254164213614"><a name="p14254164213614"></a><a name="p14254164213614"></a>无</p>
</td>
</tr>
<tr id="row182547427362"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p102541942173615"><a name="p102541942173615"></a><a name="p102541942173615"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p152541742103611"><a name="p152541742103611"></a><a name="p152541742103611"></a>无</p>
</td>
</tr>
<tr id="row425404212368"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p19254204218367"><a name="p19254204218367"></a><a name="p19254204218367"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1457312121112"><a name="p1457312121112"></a><a name="p1457312121112"></a>指针“x”的长度应该为dims * n，否则可能出现越界读写错误并引起程序崩溃。</p>
<p id="p19688575103"><a name="p19688575103"></a><a name="p19688575103"></a>“n”的取值范围：0 &lt; n &lt; 1e9。</p>
</td>
</tr>
</tbody>
</table>

## add\_with\_ids接口<a id="ZH-CN_TOPIC_0000001456694864"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>void add_with_ids(idx_t n, const float *x, const idx_t *ids)  override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndex建库和向底库中添加新的特征向量的功能，添加时底库特征都有对应的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b21773411615"><a name="b21773411615"></a><a name="b21773411615"></a>idx_t n</strong>：待添加进底库的特征向量数量。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1733711363162"><a name="b1733711363162"></a><a name="b1733711363162"></a>const float *x</strong>：待添加进底库的特征向量。</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b990063701613"><a name="b990063701613"></a><a name="b990063701613"></a>const idx_t *ids</strong>：待添加进底库的特征向量对应的ID。</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul733045716013"></a><a name="ul733045716013"></a><ul id="ul733045716013"><li>指针<span class="parmname" id="parmname1328133181717"><a name="parmname1328133181717"></a><a name="parmname1328133181717"></a>“x”</span>的长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，指针<span class="parmname" id="parmname88221109175"><a name="parmname88221109175"></a><a name="parmname88221109175"></a>“ids”</span>的长度应该为<span class="parmname" id="parmname01471241135"><a name="parmname01471241135"></a><a name="parmname01471241135"></a>“n”</span>，否则可能出现越界读写错误并引起程序崩溃。<span class="parmname" id="parmname1341462691113"><a name="parmname1341462691113"></a><a name="parmname1341462691113"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>当filter开关<a href="./03_AscendIndexConfig.md#ZH-CN_TOPIC_0000001506414705">filterable</a>为<span class="parmvalue" id="parmvalue1815553511215"><a name="parmvalue1815553511215"></a><a name="parmvalue1815553511215"></a>“true”</span>时，需要保证<span class="parmname" id="parmname1568325717215"><a name="parmname1568325717215"></a><a name="parmname1568325717215"></a>“ids”</span>中的时间戳为正。<p id="p94232041214"><a name="p94232041214"></a><a name="p94232041214"></a><span class="parmname" id="parmname16853541142914"><a name="parmname16853541142914"></a><a name="parmname16853541142914"></a>“ids”</span>（类型为uint64_t）中包含了timestamp（时间戳，类型为int32_t）和cid（camera id，类型为uint8_t），如下所示：</p>
<pre class="screen" id="screen2086011148112"><a name="screen2086011148112"></a><a name="screen2086011148112"></a>-----| cid | timestamp | -----
 14  |  8  |    32     |  10</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table562574920111"></a>
<table><tbody><tr id="row176667494111"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.1.1"><p id="p466617492115"><a name="p466617492115"></a><a name="p466617492115"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.1.1 "><p id="p866615493118"><a name="p866615493118"></a><a name="p866615493118"></a>void add_with_ids(idx_t n, const uint16_t *x, const idx_t *ids);</p>
</td>
</tr>
<tr id="row7666184961113"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.2.1"><p id="p66661749191116"><a name="p66661749191116"></a><a name="p66661749191116"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p46661497116"><a name="p46661497116"></a><a name="p46661497116"></a>实现AscendIndex建库和向底库中添加新的特征向量的功能，添加时底库特征都有对应的ID。</p>
</td>
</tr>
<tr id="row17666649161114"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.3.1"><p id="p11666124917117"><a name="p11666124917117"></a><a name="p11666124917117"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.3.1 "><p id="p15666164911115"><a name="p15666164911115"></a><a name="p15666164911115"></a><strong id="b1077450181210"><a name="b1077450181210"></a><a name="b1077450181210"></a>idx_t n</strong>：待添加进底库的特征向量数量。</p>
<p id="p566644951113"><a name="p566644951113"></a><a name="p566644951113"></a><strong id="b3312361212"><a name="b3312361212"></a><a name="b3312361212"></a>const uint16_t *x</strong>：待添加进底库的特征向量。</p>
<p id="p18666249141110"><a name="p18666249141110"></a><a name="p18666249141110"></a><strong id="b111511511218"><a name="b111511511218"></a><a name="b111511511218"></a>const idx_t *ids</strong>：待添加进底库的特征向量对应的ID。</p>
</td>
</tr>
<tr id="row14666144911116"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.4.1"><p id="p1166674921110"><a name="p1166674921110"></a><a name="p1166674921110"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.4.1 "><p id="p196661949181117"><a name="p196661949181117"></a><a name="p196661949181117"></a>无</p>
</td>
</tr>
<tr id="row4666449191111"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.5.1"><p id="p7666749101110"><a name="p7666749101110"></a><a name="p7666749101110"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.5.1 "><p id="p1766784912116"><a name="p1766784912116"></a><a name="p1766784912116"></a>无</p>
</td>
</tr>
<tr id="row86671349131119"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.6.1"><p id="p1266714991113"><a name="p1266714991113"></a><a name="p1266714991113"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><a name="ul1215264517123"></a><a name="ul1215264517123"></a><ul id="ul1215264517123"><li>指针“x”的长度应该为dims * n，指针“ids”的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。“n”的取值范围：0 &lt; n &lt; 1e9。</li><li>当filter开关<a href="./03_AscendIndexConfig.md#ZH-CN_TOPIC_0000001506414705">filterable</a>为“true”时，需要保证“ids”中的时间戳为正。“ids”（类型为uint64_t）中包含了timestamp（时间戳，类型为int32_t）和cid（camera id，类型为uint8_t），如下所示：<a name="screen11981113915128"></a><a name="screen11981113915128"></a><pre class="screen" codetype="ColdFusion" id="screen11981113915128">-----| cid | timestamp | -----
 14  |  8  |    32     |  10</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

## AscendIndex接口<a name="ZH-CN_TOPIC_0000001456695048"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndex(int dims, faiss::MetricType metric, AscendIndexConfig config)</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndex的构造函数，生成维度为dims的AscendIndex（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="parmname18664330662"><a name="parmname18664330662"></a><a name="parmname18664330662"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b48571551268"><a name="b48571551268"></a><a name="b48571551268"></a>int dims</strong>：AscendIndex管理的一组特征向量的维度。</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b69551602076"><a name="b69551602076"></a><a name="b69551602076"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型，当前支持<span class="parmvalue" id="parmvalue131913419312"><a name="parmvalue131913419312"></a><a name="parmvalue131913419312"></a>“faiss::MetricType::METRIC_L2”</span>以及<span class="parmvalue" id="parmvalue645016449313"><a name="parmvalue645016449313"></a><a name="parmvalue645016449313"></a>“faiss::MetricType::METRIC_INNER_PRODUCT”</span>。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b167641631570"><a name="b167641631570"></a><a name="b167641631570"></a>AscendIndexConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname95619186911"><a name="parmname95619186911"></a><a name="parmname95619186911"></a>“dims”</span>为(0, 4096]的整数且需要能被16整除。</p>
</td>
</tr>
</tbody>
</table>

<a name="table161511529133912"></a>
<table><tbody><tr id="row1615110293394"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2151429113910"><a name="p2151429113910"></a><a name="p2151429113910"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15151152943916"><a name="p15151152943916"></a><a name="p15151152943916"></a>AscendIndex(const AscendIndex&amp;) = delete;</p>
</td>
</tr>
<tr id="row51517295398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p21514294391"><a name="p21514294391"></a><a name="p21514294391"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015122918399"><a name="p2015122918399"></a><a name="p2015122918399"></a>声明AscendIndex拷贝构造函数为空，即AscendIndex为不可拷贝类型。</p>
</td>
</tr>
<tr id="row815120292398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7151122933917"><a name="p7151122933917"></a><a name="p7151122933917"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b71747511399"><a name="b71747511399"></a><a name="b71747511399"></a>const AscendIndex&amp;</strong>：常量AscendIndex。</p>
</td>
</tr>
<tr id="row18151172918399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p615182993916"><a name="p615182993916"></a><a name="p615182993916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8151329143914"><a name="p8151329143914"></a><a name="p8151329143914"></a>无</p>
</td>
</tr>
<tr id="row171511295399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p17151192917392"><a name="p17151192917392"></a><a name="p17151192917392"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16151122914394"><a name="p16151122914394"></a><a name="p16151122914394"></a>无</p>
</td>
</tr>
<tr id="row12151829153910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p615192973914"><a name="p615192973914"></a><a name="p615192973914"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15151929163918"><a name="p15151929163918"></a><a name="p15151929163918"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table62621513124018"></a>
<table><tbody><tr id="row726218134408"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1726212134400"><a name="p1726212134400"></a><a name="p1726212134400"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndex();</p>
</td>
</tr>
<tr id="row1926221314401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1926218134408"><a name="p1926218134408"></a><a name="p1926218134408"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82621213184020"><a name="p82621213184020"></a><a name="p82621213184020"></a>AscendIndex的析构函数，销毁AscendIndex对象，释放资源。</p>
</td>
</tr>
<tr id="row15262213104015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p826221314402"><a name="p826221314402"></a><a name="p826221314402"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row1726271324017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p5262213154014"><a name="p5262213154014"></a><a name="p5262213154014"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16262131311400"><a name="p16262131311400"></a><a name="p16262131311400"></a>无</p>
</td>
</tr>
<tr id="row0262121324020"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p8262191319409"><a name="p8262191319409"></a><a name="p8262191319409"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p726201319407"><a name="p726201319407"></a><a name="p726201319407"></a>无</p>
</td>
</tr>
<tr id="row526241324016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p526201310404"><a name="p526201310404"></a><a name="p526201310404"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15262111319403"><a name="p15262111319403"></a><a name="p15262111319403"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getDeviceList接口<a name="ZH-CN_TOPIC_0000001506495857"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>std::vector&lt;int&gt; getDeviceList();</p>
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

## operator= 接口<a name="ZH-CN_TOPIC_0000001506334661"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11253135664517"><a name="p11253135664517"></a><a name="p11253135664517"></a>AscendIndex&amp; operator=(const AscendIndex&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明AscendIndex赋值构造函数为空，即AscendIndex为不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b105275248101"><a name="b105275248101"></a><a name="b105275248101"></a>const AscendIndex&amp;</strong>：常量AscendIndex。</p>
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

## reclaimMemory接口<a name="ZH-CN_TOPIC_0000001456695092"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>virtual size_t reclaimMemory();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p257751955420"><a name="p257751955420"></a><a name="p257751955420"></a>在保证底库数量不变的情况下，缩减底库占用的内存，交由子类继承并实现，在本类中不提供相应的实现。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>缩减的内存大小，单位为Byte。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001456535000"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndex删除底库中指定的特征向量的接口。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b94414427185"><a name="b94414427185"></a><a name="b94414427185"></a>const faiss::IDSelector &amp;sel</strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>返回被删除的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## reserveMemory接口<a name="ZH-CN_TOPIC_0000001456375348"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13227195413508"><a name="p13227195413508"></a><a name="p13227195413508"></a>virtual void reserveMemory(size_t numVecs);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p169016017519"><a name="p169016017519"></a><a name="p169016017519"></a>在建立底库前为底库申请预留内存的抽象接口，交由子类继承并实现，在本类中不提供相应的实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p58431121125220"><a name="p58431121125220"></a><a name="p58431121125220"></a><strong id="b10230125512282"><a name="b10230125512282"></a><a name="b10230125512282"></a>size_t numVecs</strong>：申请预留内存的底库数量。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## reset接口<a name="ZH-CN_TOPIC_0000001506414901"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void reset() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>清空该AscendIndex的底库向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a>无</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1911619471633"><a name="p1911619471633"></a><a name="p1911619471633"></a>无</p>
</td>
</tr>
</tbody>
</table>

## search接口<a name="ZH-CN_TOPIC_0000001506334641"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.1.1 "><p id="p8820054142218"><a name="p8820054142218"></a><a name="p8820054142218"></a>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndex特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname9885102471911"><a name="parmname9885102471911"></a><a name="parmname9885102471911"></a>“k”</span>条特征的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><strong id="b6402181191915"><a name="b6402181191915"></a><a name="b6402181191915"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><strong id="b8615513161912"><a name="b8615513161912"></a><a name="b8615513161912"></a>const float *x</strong>：特征向量数据。</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><strong id="b82719159198"><a name="b82719159198"></a><a name="b82719159198"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p179121029182319"><a name="p179121029182319"></a><a name="p179121029182319"></a><strong id="b48208372230"><a name="b48208372230"></a><a name="b48208372230"></a>const SearchParameters *params：</strong>Faiss的可选参数，默认为<span class="parmvalue" id="parmvalue89412315242"><a name="parmvalue89412315242"></a><a name="parmvalue89412315242"></a>“nullptr”</span>，暂不支持该参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b74651934121914"><a name="b74651934121914"></a><a name="b74651934121914"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname8247205815587"><a name="parmname8247205815587"></a><a name="parmname8247205815587"></a>“k”</span>个向量间的距离值。当有效的检索结果不足<span class="parmname" id="parmname78458394464"><a name="parmname78458394464"></a><a name="parmname78458394464"></a>“k”</span>个时，剩余无效距离用65504或-65504填充（因metric而异）。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b11761113620191"><a name="b11761113620191"></a><a name="b11761113620191"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname1016310116599"><a name="parmname1016310116599"></a><a name="parmname1016310116599"></a>“k”</span>个向量的ID。当有效的检索结果不足<span class="parmname" id="parmname1267616380468"><a name="parmname1267616380468"></a><a name="parmname1267616380468"></a>“k”</span>个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.6.1 "><p id="p13601965223"><a name="p13601965223"></a><a name="p13601965223"></a>查询的特征向量数据<span class="parmname" id="parmname20164145332011"><a name="parmname20164145332011"></a><a name="parmname20164145332011"></a>“x”</span>的长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，<span class="parmname" id="parmname1180921315209"><a name="parmname1180921315209"></a><a name="parmname1180921315209"></a>“distances”</span>以及<span class="parmname" id="parmname6637916162017"><a name="parmname6637916162017"></a><a name="parmname6637916162017"></a>“labels”</span>的长度应该为<strong id="b7409174322613"><a name="b7409174322613"></a><a name="b7409174322613"></a>k</strong> * <strong id="b17392135042613"><a name="b17392135042613"></a><a name="b17392135042613"></a>n</strong>，否则可能会出现越界读写的情况，引起程序的崩溃。其中，<span class="parmname" id="parmname6386539553"><a name="parmname6386539553"></a><a name="parmname6386539553"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9；<span class="parmname" id="parmname57591741410"><a name="parmname57591741410"></a><a name="parmname57591741410"></a>“k”</span>通常不允许超过4096。</p>
</td>
</tr>
<tr id="row68701915173311"><th class="firstcol" valign="top" width="20.09%" id="mcps1.1.3.7.1"><p id="p11871615173310"><a name="p11871615173310"></a><a name="p11871615173310"></a>注意事项</p>
</th>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.1.3.7.1 "><p id="p68716152338"><a name="p68716152338"></a><a name="p68716152338"></a>使用小库暴搜算法的场景中，如果在底库和batch数较大时出现性能下降现象，需要增大AscendIndexConfig中的<span class="parmname" id="parmname298114512273"><a name="parmname298114512273"></a><a name="parmname298114512273"></a>“resources”</span>参数值（暴搜算法默认值为128MB）。</p>
</td>
</tr>
</tbody>
</table>

<a name="table03178548130"></a>
<table><tbody><tr id="row133713545133"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.1.1"><p id="p0371145411316"><a name="p0371145411316"></a><a name="p0371145411316"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.1.1 "><p id="p1737125421319"><a name="p1737125421319"></a><a name="p1737125421319"></a>void search(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels) const;</p>
</td>
</tr>
<tr id="row93719547138"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.2.1"><p id="p3371165419130"><a name="p3371165419130"></a><a name="p3371165419130"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p537135414133"><a name="p537135414133"></a><a name="p537135414133"></a>实现AscendIndex特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。</p>
</td>
</tr>
<tr id="row537295491313"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.3.1"><p id="p1137213548130"><a name="p1137213548130"></a><a name="p1137213548130"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.3.1 "><p id="p1437219546136"><a name="p1437219546136"></a><a name="p1437219546136"></a><strong id="b588011420147"><a name="b588011420147"></a><a name="b588011420147"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p0372135401314"><a name="p0372135401314"></a><a name="p0372135401314"></a><strong id="b15881487142"><a name="b15881487142"></a><a name="b15881487142"></a>const uint16_t *x</strong>：特征向量数据。</p>
<p id="p13372205419135"><a name="p13372205419135"></a><a name="p13372205419135"></a><strong id="b128361411131413"><a name="b128361411131413"></a><a name="b128361411131413"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
</td>
</tr>
<tr id="row13721254131312"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.4.1"><p id="p11372254101315"><a name="p11372254101315"></a><a name="p11372254101315"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.4.1 "><p id="p10372165412137"><a name="p10372165412137"></a><a name="p10372165412137"></a><strong id="b173463192141"><a name="b173463192141"></a><a name="b173463192141"></a>float *distances</strong>：查询向量与距离最近的前“k”个向量间的距离值。当有效的检索结果不足“k”个时，剩余无效距离用65504或-65504填充（因metric而异）。</p>
<p id="p53727546138"><a name="p53727546138"></a><a name="p53727546138"></a><strong id="b8454102181419"><a name="b8454102181419"></a><a name="b8454102181419"></a>idx_t *labels</strong>：查询的距离最近的前“k”个向量的ID。当有效的检索结果不足“k”个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row43722544139"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.5.1"><p id="p937225491317"><a name="p937225491317"></a><a name="p937225491317"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.5.1 "><p id="p1337275461310"><a name="p1337275461310"></a><a name="p1337275461310"></a>无</p>
</td>
</tr>
<tr id="row15372954111319"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.6.1"><p id="p437220547131"><a name="p437220547131"></a><a name="p437220547131"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><p id="p337255411313"><a name="p337255411313"></a><a name="p337255411313"></a>查询的特征向量数据“x”的长度应该为dims * n，“distances”以及“labels”的长度应该为k * n，否则可能会出现越界读写的情况，引起程序的崩溃。其中，“n”的取值范围：0 &lt; n &lt; 1e9；“k”通常不允许超过4096。</p>
</td>
</tr>
<tr id="row19372135418134"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.7.1"><p id="p7372155411136"><a name="p7372155411136"></a><a name="p7372155411136"></a>注意事项</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.7.1 "><p id="p15372105431319"><a name="p15372105431319"></a><a name="p15372105431319"></a>使用小库暴搜算法的场景中，如果在底库和batch数较大时出现性能下降现象，需要增大AscendIndexConfig中的“resources”参数值（暴搜算法默认值为128MB）。</p>
</td>
</tr>
</tbody>
</table>
