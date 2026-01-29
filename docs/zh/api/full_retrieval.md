
## 全量检索<a name="ZH-CN_TOPIC_0000001533164645"></a>

### AscendIndex<a id="ZH-CN_TOPIC_0000001456375304"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001506414937"></a>

AscendIndex作为特征检索组件中的大部分检索的Index的基类，向上承接Faiss，向下为特征检索中的其他Index定义接口。


#### add接口<a id="ZH-CN_TOPIC_0000001506614985"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p5684112753414"><a name="p5684112753414"></a><a name="p5684112753414"></a>void add(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndex建库和往底库中添加新的特征向量的功能。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
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
说明： </span><div class="notebody"><a name="ul103685518369"></a><a name="ul103685518369"></a><ul id="ul103685518369"><li>add接口不能与add_with_ids接口混用。</li><li>使用add接口后，search结果的labels可能会重复，如果业 务上对label有要求，建议使用add_with_ids接口。</li></ul>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p4352184531013"><a name="p4352184531013"></a><a name="p4352184531013"></a>实现AscendIndex建库和往底库中添加新的特征向量的功能。使用add接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
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


#### add\_with\_ids接口<a id="ZH-CN_TOPIC_0000001456694864"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>void add_with_ids(idx_t n, const float *x, const idx_t *ids)  override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndex建库和往底库中添加新的特征向量的功能，添加时底库特征都有对应的ID。</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul733045716013"></a><a name="ul733045716013"></a><ul id="ul733045716013"><li>指针<span class="parmname" id="parmname1328133181717"><a name="parmname1328133181717"></a><a name="parmname1328133181717"></a>“x”</span>的长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，指针<span class="parmname" id="parmname88221109175"><a name="parmname88221109175"></a><a name="parmname88221109175"></a>“ids”</span>的长度应该为<span class="parmname" id="parmname01471241135"><a name="parmname01471241135"></a><a name="parmname01471241135"></a>“n”</span>，否则可能出现越界读写错误并引起程序崩溃。<span class="parmname" id="parmname1341462691113"><a name="parmname1341462691113"></a><a name="parmname1341462691113"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>当filter开关<a href="#ZH-CN_TOPIC_0000001506414705">filterable</a></a>为<span class="parmvalue" id="parmvalue1815553511215"><a name="parmvalue1815553511215"></a><a name="parmvalue1815553511215"></a>“true”</span>时，需要保证<span class="parmname" id="parmname1568325717215"><a name="parmname1568325717215"></a><a name="parmname1568325717215"></a>“ids”</span>中的时间戳为正。<p id="p94232041214"><a name="p94232041214"></a><a name="p94232041214"></a><span class="parmname" id="parmname16853541142914"><a name="parmname16853541142914"></a><a name="parmname16853541142914"></a>“ids”</span>（类型为uint64_t）中包含了timestamp（时间戳，类型为int32_t）和cid（camera id，类型为uint8_t），如下所示：</p>
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
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p46661497116"><a name="p46661497116"></a><a name="p46661497116"></a>实现AscendIndex建库和往底库中添加新的特征向量的功能，添加时底库特征都有对应的ID。</p>
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
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><a name="ul1215264517123"></a><a name="ul1215264517123"></a><ul id="ul1215264517123"><li>指针“x”的长度应该为dims * n，指针“ids”的长度应该为“n”，否则可能出现越界读写错误并引起程序崩溃。“n”的取值范围：0 &lt; n &lt; 1e9。</li><li>当filter开关<a href="#ZH-CN_TOPIC_0000001506414705">filterable</a>为“true”时，需要保证“ids”中的时间戳为正。“ids”（类型为uint64_t）中包含了timestamp（时间戳，类型为int32_t）和cid（camera id，类型为uint8_t），如下所示：<a name="screen11981113915128"></a><a name="screen11981113915128"></a><pre class="screen" codetype="ColdFusion" id="screen11981113915128">-----| cid | timestamp | -----
 14  |  8  |    32     |  10</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>


#### AscendIndex接口<a name="ZH-CN_TOPIC_0000001456695048"></a>

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


#### getDeviceList接口<a name="ZH-CN_TOPIC_0000001506495857"></a>

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


#### operator= 接口<a name="ZH-CN_TOPIC_0000001506334661"></a>

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


#### reclaimMemory接口<a name="ZH-CN_TOPIC_0000001456695092"></a>

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


#### remove\_ids接口<a name="ZH-CN_TOPIC_0000001456535000"></a>

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


#### reserveMemory接口<a name="ZH-CN_TOPIC_0000001456375348"></a>

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


#### reset接口<a name="ZH-CN_TOPIC_0000001506414901"></a>

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


#### search接口<a name="ZH-CN_TOPIC_0000001506334641"></a>

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



### AscendIndexCluster<a id="ZH-CN_TOPIC_0000001614744825"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001564586790"></a>

AscendIndexCluster需要使用[Init](#init接口)指定对应资源的初始化，初始化完之后会申请一段完整空间用于存储底库。在使用完之后，需要调用[Finalize](#finalize接口)接口对资源进行释放。

AscendIndexCluster仅支持使用Atlas 推理系列产品，在标准态部署方式下的向量内积距离类型。AscendIndexCluster在使用时依赖Flat和AICPU算子，具体请参见[Flat](../user_guide.md#flat)和[AICPU](../user_guide.md#aicpu)。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。


#### AddFeatures接口<a name="ZH-CN_TOPIC_0000001614746533"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122877269522"><a name="p122877269522"></a><a name="p122877269522"></a>APP_ERROR AddFeatures(int n, const float *features, const uint32_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16287112617527"><a name="p16287112617527"></a><a name="p16287112617527"></a>向特征库插入<span class="parmname" id="parmname51041610175311"><a name="parmname51041610175311"></a><a name="parmname51041610175311"></a>“n”</span>个指定下标索引的特征向量，如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1728872675217"><a name="p1728872675217"></a><a name="p1728872675217"></a><strong id="b428813268527"><a name="b428813268527"></a><a name="b428813268527"></a>int n</strong>：插入特征向量数目。</p>
<p id="p172889261527"><a name="p172889261527"></a><a name="p172889261527"></a><strong id="b92884267526"><a name="b92884267526"></a><a name="b92884267526"></a>const float *feature</strong>：待插入的特征向量，长度为<strong id="b317413192537"><a name="b317413192537"></a><a name="b317413192537"></a>n</strong> * 向量维度dim。</p>
<p id="p6288192619521"><a name="p6288192619521"></a><a name="p6288192619521"></a><strong id="b1228832615215"><a name="b1228832615215"></a><a name="b1228832615215"></a>const uint32_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为<strong id="b429616208533"><a name="b429616208533"></a><a name="b429616208533"></a>n</strong>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1628882619521"><a name="p1628882619521"></a><a name="p1628882619521"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p528872613525"><a name="p528872613525"></a><a name="p528872613525"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li><strong id="b6288132610521"><a name="b6288132610521"></a><a name="b6288132610521"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname3195938155216"><a name="varname3195938155216"></a><a name="varname3195938155216"></a>capacity</span></i>)之间，indices要求是连续的。</li><li><strong id="b628892695220"><a name="b628892695220"></a><a name="b628892695220"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname12559104095211"><a name="varname12559104095211"></a><a name="varname12559104095211"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname3816191310019"><a name="parmname3816191310019"></a><a name="parmname3816191310019"></a>“features”</span>和<span class="parmname" id="parmname6551185632610"><a name="parmname6551185632610"></a><a name="parmname6551185632610"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table772538154310"></a>
<table><tbody><tr id="row97256854317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p372568194310"><a name="p372568194310"></a><a name="p372568194310"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19725148124318"><a name="p19725148124318"></a><a name="p19725148124318"></a>APP_ERROR AddFeatures(int n, const uint16_t *features, const int64_t *indices);</p>
</td>
</tr>
<tr id="row9725983433"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1372519814431"><a name="p1372519814431"></a><a name="p1372519814431"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p47259810431"><a name="p47259810431"></a><a name="p47259810431"></a>向特征库插入<span class="parmname" id="parmname872512814432"><a name="parmname872512814432"></a><a name="parmname872512814432"></a>“n”</span>个指定下标索引的特征向量，如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row1272528104315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p167251682439"><a name="p167251682439"></a><a name="p167251682439"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1872513819432"><a name="p1872513819432"></a><a name="p1872513819432"></a><strong id="b1772528184310"><a name="b1772528184310"></a><a name="b1772528184310"></a>int n</strong>：插入特征向量数目。</p>
<p id="p87251087438"><a name="p87251087438"></a><a name="p87251087438"></a><strong id="b16758135110442"><a name="b16758135110442"></a><a name="b16758135110442"></a>const uint16_t *features</strong>：待插入的特征向量，长度为<strong id="b1072519824314"><a name="b1072519824314"></a><a name="b1072519824314"></a>n</strong> * 向量维度dim。</p>
<p id="p672518884310"><a name="p672518884310"></a><a name="p672518884310"></a><strong id="b13589161410453"><a name="b13589161410453"></a><a name="b13589161410453"></a>const int64_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为<strong id="b1672510864314"><a name="b1672510864314"></a><a name="b1672510864314"></a>n</strong>。</p>
</td>
</tr>
<tr id="row187251389432"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p20725788435"><a name="p20725788435"></a><a name="p20725788435"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p3725198194318"><a name="p3725198194318"></a><a name="p3725198194318"></a>无</p>
</td>
</tr>
<tr id="row672517820435"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p57254812434"><a name="p57254812434"></a><a name="p57254812434"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p107256824313"><a name="p107256824313"></a><a name="p107256824313"></a><strong id="b97252810435"><a name="b97252810435"></a><a name="b97252810435"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1972548114318"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9725582431"><a name="p9725582431"></a><a name="p9725582431"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul87251988433"></a><a name="ul87251988433"></a><ul id="ul87251988433"><li><strong id="b1872558124320"><a name="b1872558124320"></a><a name="b1872558124320"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname1272528114317"><a name="varname1272528114317"></a><a name="varname1272528114317"></a>capacity</span></i>)之间。</li><li><strong id="b67251181438"><a name="b67251181438"></a><a name="b67251181438"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname117251088432"><a name="varname117251088432"></a><a name="varname117251088432"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname872512894317"><a name="parmname872512894317"></a><a name="parmname872512894317"></a>“features”</span>和<span class="parmname" id="parmname1172514824315"><a name="parmname1172514824315"></a><a name="parmname1172514824315"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### AscendIndexCluster接口<a name="ZH-CN_TOPIC_0000001564746410"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p9608143314716"><a name="p9608143314716"></a><a name="p9608143314716"></a>AscendIndexCluster();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1760814333474"><a name="p1760814333474"></a><a name="p1760814333474"></a>AscendIndexCluster的构造函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p204291821488"><a name="p204291821488"></a><a name="p204291821488"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p194301211486"><a name="p194301211486"></a><a name="p194301211486"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1743014244819"><a name="p1743014244819"></a><a name="p1743014244819"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20430172184812"><a name="p20430172184812"></a><a name="p20430172184812"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table15621560282"></a>
<table><tbody><tr id="row1256265642816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p20562165614286"><a name="p20562165614286"></a><a name="p20562165614286"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p29285682519"><a name="p29285682519"></a><a name="p29285682519"></a>AscendIndexCluster(const AscendIndexCluster&amp;) = delete;</p>
</td>
</tr>
<tr id="row756235619282"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p95621656152818"><a name="p95621656152818"></a><a name="p95621656152818"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p538993116244"><a name="p538993116244"></a><a name="p538993116244"></a>声明此Index拷贝函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row356225619283"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1956245682817"><a name="p1956245682817"></a><a name="p1956245682817"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p998472882614"><a name="p998472882614"></a><a name="p998472882614"></a><strong id="b188011316264"><a name="b188011316264"></a><a name="b188011316264"></a>const AscendIndexCluster&amp;</strong>：AscendIndexCluster对象。</p>
</td>
</tr>
<tr id="row55621556102815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p135627560287"><a name="p135627560287"></a><a name="p135627560287"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row0562256142813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1656225622819"><a name="p1656225622819"></a><a name="p1656225622819"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row145621856162814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19562105616284"><a name="p19562105616284"></a><a name="p19562105616284"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### \~AscendIndexCluster接口<a name="ZH-CN_TOPIC_0000002399598393"></a>

<a name="table179216322487"></a>
<table><tbody><tr id="row2092173214484"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3929320480"><a name="p3929320480"></a><a name="p3929320480"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p118119814816"><a name="p118119814816"></a><a name="p118119814816"></a>virtual ~AscendIndexCluster() = default;</p>
</td>
</tr>
<tr id="row092163217481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p493532184819"><a name="p493532184819"></a><a name="p493532184819"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2811384485"><a name="p2811384485"></a><a name="p2811384485"></a>AscendIndexCluster的析构函数。</p>
</td>
</tr>
<tr id="row1193163244813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p19316328485"><a name="p19316328485"></a><a name="p19316328485"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1581178114813"><a name="p1581178114813"></a><a name="p1581178114813"></a>无</p>
</td>
</tr>
<tr id="row9931932104818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p793173212484"><a name="p793173212484"></a><a name="p793173212484"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p148111782488"><a name="p148111782488"></a><a name="p148111782488"></a>无</p>
</td>
</tr>
<tr id="row89333214481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p89316323489"><a name="p89316323489"></a><a name="p89316323489"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p58111481480"><a name="p58111481480"></a><a name="p58111481480"></a>无</p>
</td>
</tr>
<tr id="row49311328486"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p993113234818"><a name="p993113234818"></a><a name="p993113234818"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4811148104812"><a name="p4811148104812"></a><a name="p4811148104812"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000002446061685"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p68995910575"><a name="p68995910575"></a><a name="p68995910575"></a>APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num, const uint32_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>ComputeDistance计算待查询向量与所有底库向量的距离，而ComputeDistanceByIdx接口只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1178514265435"><a name="b1178514265435"></a><a name="b1178514265435"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b276887014"><a name="b276887014"></a><a name="b276887014"></a>const uint16_t *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname1441759144217"><a name="parmname1441759144217"></a><a name="parmname1441759144217"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a>const int *num</strong>：给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b921610409216"><a name="b921610409216"></a><a name="b921610409216"></a>const uint32_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname2711154912437"><a name="parmname2711154912437"></a><a name="parmname2711154912437"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname742124364316"><a name="parmname742124364316"></a><a name="parmname742124364316"></a>“indices”</span>长度为n * max(num)。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b156251035184313"><a name="b156251035184313"></a><a name="b156251035184313"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b25863377438"><a name="b25863377438"></a><a name="b25863377438"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname136035014443"><a name="parmname136035014443"></a><a name="parmname136035014443"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue2069717180212"><a name="parmvalue2069717180212"></a><a name="parmvalue2069717180212"></a>“48”</span>，即<span class="parmname" id="parmname1997224217"><a name="parmname1997224217"></a><a name="parmname1997224217"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p057182814222"><a name="p057182814222"></a><a name="p057182814222"></a><strong id="b0194557446"><a name="b0194557446"></a><a name="b0194557446"></a>float *distances</strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大<span class="parmname" id="parmname658971354417"><a name="parmname658971354417"></a><a name="parmname658971354417"></a>“num”</span>补齐空间占用，空间长度为n * max(num)。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b140412864414"><a name="b140412864414"></a><a name="b140412864414"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname82723561324"><a name="varname82723561324"></a><a name="varname82723561324"></a>capacity</span></i>]之间。</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0， ntotal]之间。</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
</td>
</tr>
</tbody>
</table>


#### ComputeDistanceByThreshold接口<a name="ZH-CN_TOPIC_0000001615066169"></a>

> [!NOTE] 说明 
>当前接口需配合[AddFeatures\(int n, const float \*features, const uint32\_t \*indices\);](#addfeatures接口)使用。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.1.1 "><p id="p15352133820537"><a name="p15352133820537"></a><a name="p15352133820537"></a>APP_ERROR ComputeDistanceByThreshold(const std::vector&lt;uint32_t&gt; &amp;queryIdxArr, uint32_t codeStartIdx,  uint32_t codeNum, float threshold, bool aboveFilter, std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr, std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.2.1 "><p id="p1935283855318"><a name="p1935283855318"></a><a name="p1935283855318"></a>查询指定条数在底库中的特征向量与指定的底库特征向量的距离，并根据阈值筛选，返回满足条件的距离和其label。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.3.1 "><p id="p1535283885320"><a name="p1535283885320"></a><a name="p1535283885320"></a><strong id="b976392112545"><a name="b976392112545"></a><a name="b976392112545"></a>const std::vector&lt;uint32_t&gt; &amp;queryIdxArr</strong>：要查询的向量在底库中的序号。</p>
<p id="p1835213385530"><a name="p1835213385530"></a><a name="p1835213385530"></a><strong id="b57881123165418"><a name="b57881123165418"></a><a name="b57881123165418"></a>uint32_t codeStartIdx</strong>：要计算距离的底库的起始序号。</p>
<p id="p23526383537"><a name="p23526383537"></a><a name="p23526383537"></a><strong id="b16334152510547"><a name="b16334152510547"></a><a name="b16334152510547"></a>uint32_t codeNum</strong>：要计算距离的底库向量的数量。</p>
<p id="p203521138195313"><a name="p203521138195313"></a><a name="p203521138195313"></a><strong id="b112831830115415"><a name="b112831830115415"></a><a name="b112831830115415"></a>float threshold</strong>：用于过滤的阈值，过滤掉比阈值小的距离。</p>
<p id="p1435223815538"><a name="p1435223815538"></a><a name="p1435223815538"></a><strong id="b2932133114541"><a name="b2932133114541"></a><a name="b2932133114541"></a>bool aboveFilter</strong>：预留参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.4.1 "><p id="p13522382534"><a name="p13522382534"></a><a name="p13522382534"></a><strong id="b0375144918549"><a name="b0375144918549"></a><a name="b0375144918549"></a>std::vector&lt;std::vector&lt;float&gt;&gt; &amp;resDistArr</strong>：返回的二维数组，每个要查询的向量与其满足阈值条件的底库向量的距离。</p>
<p id="p83524381530"><a name="p83524381530"></a><a name="p83524381530"></a><strong id="b166640512549"><a name="b166640512549"></a><a name="b166640512549"></a>std::vector&lt;std::vector&lt;uint32_t&gt;&gt; &amp;resIdxArr</strong>：返回的二维数组，每个要查询的向量与其满足阈值条件的底库向量的label。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.5.1 "><p id="p735215385536"><a name="p735215385536"></a><a name="p735215385536"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.04%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.96%" headers="mcps1.1.3.6.1 "><a name="ul1624216519526"></a><a name="ul1624216519526"></a><ul id="ul1624216519526"><li><span class="parmname" id="parmname138712196568"><a name="parmname138712196568"></a><a name="parmname138712196568"></a>“queryIdxArr”</span>与<span class="parmname" id="parmname181661023145619"><a name="parmname181661023145619"></a><a name="parmname181661023145619"></a>“resDistArr”</span>和<span class="parmname" id="parmname1785414248565"><a name="parmname1785414248565"></a><a name="parmname1785414248565"></a>“resIdxArr”</span>长度要一致，即<strong id="b591716125616"><a name="b591716125616"></a><a name="b591716125616"></a>queryIdxArr.size() == resDistArr.size()</strong>。</li><li><span class="parmname" id="parmname1560612576523"><a name="parmname1560612576523"></a><a name="parmname1560612576523"></a>“queryIdxArr.size()”</span>需大于<span class="parmvalue" id="parmvalue71274012531"><a name="parmvalue71274012531"></a><a name="parmvalue71274012531"></a>“0”</span>并且小于等于<span class="parmname" id="parmname1723074011015"><a name="parmname1723074011015"></a><a name="parmname1723074011015"></a>“ntotal”</span>。</li><li><span class="parmname" id="parmname7798135125316"><a name="parmname7798135125316"></a><a name="parmname7798135125316"></a>“codeNum”</span>需大于<span class="parmvalue" id="parmvalue11855877538"><a name="parmvalue11855877538"></a><a name="parmvalue11855877538"></a>“0”</span>并且小于等于<span class="parmname" id="parmname18896901916"><a name="parmname18896901916"></a><a name="parmname18896901916"></a>“ntotal”</span>。</li><li><span class="parmname" id="parmname19911327165616"><a name="parmname19911327165616"></a><a name="parmname19911327165616"></a>“codeStartIdx”</span> + <span class="parmname" id="parmname10710830185612"><a name="parmname10710830185612"></a><a name="parmname10710830185612"></a>“codeNum”</span>不大于<span class="parmname" id="parmname163571213569"><a name="parmname163571213569"></a><a name="parmname163571213569"></a>“ntotal”</span>（底库大小）。</li><li><span class="parmname" id="parmname6690927114312"><a name="parmname6690927114312"></a><a name="parmname6690927114312"></a>“codeStartIdx”</span>需大于等于<span class="parmvalue" id="parmvalue136896102437"><a name="parmvalue136896102437"></a><a name="parmvalue136896102437"></a>“0”</span>并且小于等于<span class="parmname" id="parmname3409151774318"><a name="parmname3409151774318"></a><a name="parmname3409151774318"></a>“ntotal”</span></li></ul>
</td>
</tr>
</tbody>
</table>


#### Finalize接口<a name="ZH-CN_TOPIC_0000001614906601"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p377414915513"><a name="p377414915513"></a><a name="p377414915513"></a>void Finalize();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>释放特征库管理资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p877484919511"><a name="p877484919511"></a><a name="p877484919511"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14774134917519"><a name="p14774134917519"></a><a name="p14774134917519"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1277416496515"><a name="p1277416496515"></a><a name="p1277416496515"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16774184918516"><a name="p16774184918516"></a><a name="p16774184918516"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### GetFeatures接口<a name="ZH-CN_TOPIC_0000002412742482"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeatures(int n, uint16_t *features, const int64_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname9635334135520"><a name="parmname9635334135520"></a><a name="parmname9635334135520"></a>“n”</span>条指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b13667438181211"><a name="b13667438181211"></a><a name="b13667438181211"></a>const int64_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b2033375501314"><a name="b2033375501314"></a><a name="b2033375501314"></a>uint16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### GetNTotal接口<a name="ZH-CN_TOPIC_0000002412582646"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p61958153167"><a name="p61958153167"></a><a name="p61958153167"></a>int GetNTotal() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询当前特征库特征向量数目的理论最大值。如果插入特征向量indices连续，则ntotal等于特征向量数目。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a><strong id="b445021732816"><a name="b445021732816"></a><a name="b445021732816"></a>int ntotal</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b171461107533"><a name="b171461107533"></a><a name="b171461107533"></a>int</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### Init接口<a name="ZH-CN_TOPIC_0000001614866169"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p463919713918"><a name="p463919713918"></a><a name="p463919713918"></a>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p4783164014487"><a name="p4783164014487"></a><a name="p4783164014487"></a>AscendIndexCluster的初始化函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14783940124815"><a name="p14783940124815"></a><a name="p14783940124815"></a><strong id="b10219125824811"><a name="b10219125824811"></a><a name="b10219125824811"></a>int dim</strong>：AscendIndexCluster管理的特征向量的维度。</p>
<p id="p178314407484"><a name="p178314407484"></a><a name="p178314407484"></a><strong id="b391943494"><a name="b391943494"></a><a name="b391943494"></a>int capacity</strong>：底库最大容量，接口会根据<span class="parmname" id="parmname15611164915567"><a name="parmname15611164915567"></a><a name="parmname15611164915567"></a>“capacity”</span>值申请capacity * dim * sizeof(fp16) 字节内存数据。</p>
<p id="p1478354004816"><a name="p1478354004816"></a><a name="p1478354004816"></a><strong id="b05501363492"><a name="b05501363492"></a><a name="b05501363492"></a>faiss::MetricType metricType</strong>：特征距离类别（向量内积、欧氏距离、余弦相似度）。</p>
<p id="p978314010482"><a name="p978314010482"></a><a name="p978314010482"></a><strong id="b12591941184914"><a name="b12591941184914"></a><a name="b12591941184914"></a>const std::vector&lt;int&gt; &amp;deviceList</strong>：Device侧资源配置。</p>
<p id="p278364014481"><a name="p278364014481"></a><a name="p278364014481"></a><strong id="b19288144414914"><a name="b19288144414914"></a><a name="b19288144414914"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为<span class="parmvalue" id="parmvalue116601371297"><a name="parmvalue116601371297"></a><a name="parmvalue116601371297"></a>“-1”</span>，表示设置为<span class="parmvalue" id="parmvalue16975511132911"><a name="parmvalue16975511132911"></a><a name="parmvalue16975511132911"></a>“128MB”</span>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1378314016485"><a name="p1378314016485"></a><a name="p1378314016485"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p4783174017482"><a name="p4783174017482"></a><a name="p4783174017482"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3783154018480"></a><a name="ul3783154018480"></a><ul id="ul3783154018480"><li>dim ∈ {32, 64, 128, 256, 384, 512}。</li><li>metricType：AscendIndexCluster目前只实现了向量内积距离，即只支持<span class="parmvalue" id="parmvalue153881199571"><a name="parmvalue153881199571"></a><a name="parmvalue153881199571"></a>“faiss::MetricType::METRIC_INNER_PRODUCT。”</span></li><li>接口允许为底库申请的内存上限设为12,288,000,000Byte，同时<span class="parmname" id="parmname1951055020570"><a name="parmname1951055020570"></a><a name="parmname1951055020570"></a>“capacity”</span>的值域约束为[0, 12000000]。</li><li>以512维、FP16类型的底库向量为例，最大支持的<span class="parmname" id="parmname10415122055714"><a name="parmname10415122055714"></a><a name="parmname10415122055714"></a>“capacity”</span>为1200万( 12288000000 / (512 * sizeof(fp_16)) )。</li><li>对于256维、FP16类型的底库向量，尽管内存约束支持更大的<span class="parmname" id="parmname15756192115712"><a name="parmname15756192115712"></a><a name="parmname15756192115712"></a>“capacity”</span>，<span class="parmname" id="parmname3788102218578"><a name="parmname3788102218578"></a><a name="parmname3788102218578"></a>“capacity”</span>最大也只能设为1200万。</li><li>仅支持配置单卡，暂不支持配置多卡，需满足<strong id="b1129012151811"><a name="b1129012151811"></a><a name="b1129012151811"></a>deviceList.size() == 1</strong>。</li><li><span class="parmname" id="parmname2017164055710"><a name="parmname2017164055710"></a><a name="parmname2017164055710"></a>“resourceSize”</span>：可以配置为-1或[134217728，4294967296]之间的值，相当于[128MB，4096MB]。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置为<span class="parmvalue" id="parmvalue15293111673918"><a name="parmvalue15293111673918"></a><a name="parmvalue15293111673918"></a>“1024MB”</span>。</li></ul>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000001897100377"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19940753122510"><a name="p19940753122510"></a><a name="p19940753122510"></a>AscendIndexCluster&amp; operator=(const AscendIndexCluster&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3198107264"><a name="p3198107264"></a><a name="p3198107264"></a>声明此Index赋值构造函数为空，即不可拷贝类型</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6538114402614"><a name="p6538114402614"></a><a name="p6538114402614"></a><strong id="b18689747142614"><a name="b18689747142614"></a><a name="b18689747142614"></a>const AscendIndexCluster&amp;</strong>：AscendIndexCluster对象。</p>
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


#### RemoveFeatures接口<a name="ZH-CN_TOPIC_0000002446181741"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p457211594914"><a name="p457211594914"></a><a name="p457211594914"></a>APP_ERROR RemoveFeatures(int n, const int64_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>删除向量库中<span class="parmname" id="parmname15331161155517"><a name="parmname15331161155517"></a><a name="parmname15331161155517"></a>“n”</span>个指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b10676733203011"><a name="b10676733203011"></a><a name="b10676733203011"></a>int n</strong>：删除特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b16204233201013"><a name="b16204233201013"></a><a name="b16204233201013"></a>const int64_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b727535618302"><a name="b727535618302"></a><a name="b727535618302"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85131151122111"><a name="varname85131151122111"></a><a name="varname85131151122111"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### SearchByThreshold接口<a name="ZH-CN_TOPIC_0000002446061689"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p126941862232"><a name="p126941862232"></a><a name="p126941862232"></a>APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num, int64_t * indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p138361625164412"><a name="p138361625164412"></a><a name="p138361625164412"></a>在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1378124514396"><a name="b1378124514396"></a><a name="b1378124514396"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1148123874520"><a name="b1148123874520"></a><a name="b1148123874520"></a>const uint16_t *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p11518191412248"><a name="p11518191412248"></a><a name="p11518191412248"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname166164371714"><a name="parmname166164371714"></a><a name="parmname166164371714"></a>“threshold”</span>进行过滤。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1578817211311"><a name="parmname1578817211311"></a><a name="parmname1578817211311"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname279351319407"><a name="parmname279351319407"></a><a name="parmname279351319407"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue376417339011"><a name="parmvalue376417339011"></a><a name="parmvalue376417339011"></a>“48”</span>，即<span class="parmname" id="parmname19896039903"><a name="parmname19896039903"></a><a name="parmname19896039903"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18962112242616"><a name="p18962112242616"></a><a name="p18962112242616"></a><strong id="b39757334268"><a name="b39757334268"></a><a name="b39757334268"></a>int *num：</strong>每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p1796272252611"><a name="p1796272252611"></a><a name="p1796272252611"></a><strong id="b647125110265"><a name="b647125110265"></a><a name="b647125110265"></a>int64_t *indices：</strong>满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname10888162862715"><a name="parmname10888162862715"></a><a name="parmname10888162862715"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname2399203922710"><a name="parmname2399203922710"></a><a name="parmname2399203922710"></a>“indices”</span>总长度为n * topk。</p>
<p id="p296222222618"><a name="p296222222618"></a><a name="p296222222618"></a><strong id="b167278112714"><a name="b167278112714"></a><a name="b167278112714"></a>float *distances：</strong>满足阈值条件的底库向量与待查询向量距离，记录方式和长度与<span class="parmname" id="parmname2406996287"><a name="parmname2406996287"></a><a name="parmname2406996287"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b8300175210408"><a name="b8300175210408"></a><a name="b8300175210408"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname616535816"><a name="varname616535816"></a><a name="varname616535816"></a>capacity</span></i>]之间。</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a>topk</strong>：k取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname1431611816133"><a name="parmname1431611816133"></a><a name="parmname1431611816133"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### SetNTotal<a name="ZH-CN_TOPIC_0000002412742486"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR SetNTotal(int n);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p7313759183119"><a name="p7313759183119"></a><a name="p7313759183119"></a>为外部提供调整<span class="parmname" id="parmname159164443116"><a name="parmname159164443116"></a><a name="parmname159164443116"></a>“ntotal”</span>计数。</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>每次增加底库向量后，Index内部尽管会根据最大插入下标更新<span class="parmname" id="parmname114061192322"><a name="parmname114061192322"></a><a name="parmname114061192322"></a>“ntotal”</span>值，但并没有记录[0, <i><span class="varname" id="varname1917151317325"><a name="varname1917151317325"></a><a name="varname1917151317325"></a>ntotal</span></i>]范围内哪些区域是无效的空间，因此<strong id="b144482818157"><a name="b144482818157"></a><a name="b144482818157"></a>RemoveFeatures</strong>操作没有改变<span class="parmname" id="parmname1274441121512"><a name="parmname1274441121512"></a><a name="parmname1274441121512"></a>“ntotal”</span>的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置<span class="parmname" id="parmname159521818143214"><a name="parmname159521818143214"></a><a name="parmname159521818143214"></a>“ntotal”</span>，这样可以在可控范围内减少算子的计算量，以提高接口性能。</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>例如：当前插入100条向量，底库索引为0~99 时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部<span class="parmname" id="parmname974517165332"><a name="parmname974517165332"></a><a name="parmname974517165332"></a>“ntotal”</span>保持不变，只能设为[<i><span class="varname" id="varname169891216331"><a name="varname169891216331"></a><a name="varname169891216331"></a>ntotal</span></i>, <i><span class="varname" id="varname91661324163313"><a name="varname91661324163313"></a><a name="varname91661324163313"></a>capacity</span></i>]之间的值，再次执行删除索引为90~99的底库，此时可以手动把<span class="parmname" id="parmname18801143812373"><a name="parmname18801143812373"></a><a name="parmname18801143812373"></a>“ntotal”</span>设置为[80, <i><span class="varname" id="varname737175673612"><a name="varname737175673612"></a><a name="varname737175673612"></a>capacity</span></i>]之间的值，设置为<span class="parmvalue" id="parmvalue8748356153711"><a name="parmvalue8748356153711"></a><a name="parmvalue8748356153711"></a>“80”</span>时，可以使参与比对的底库数据量有效减少20条。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1215142783714"><a name="b1215142783714"></a><a name="b1215142783714"></a>int n</strong>：由用户在业务面管理的最大底库的索引加1。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</p>
</td>
</tr>
</tbody>
</table>



### AscendIndexConfig<a id="ZH-CN_TOPIC_0000001506414705"></a>

AscendIndex需要使用对应的AscendIndexConfig执行对应资源的初始化，AscendIndexConfig中需要配置执行检索过程中的硬件资源和内存池大小等。

> [!NOTE] 说明 
>内存池大小单位为**Byte**，此参数用于指定Device侧预留的内存池大小。内存池用于存储昇腾硬件上进行距离计算的结果，底库规模较大时，建议预留更大的内存池大小。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|deviceList|std::vector<int>|Device侧设备ID。|
|resourceSize|int64_t|Device侧内存池大小，单位为字节，默认参数为头文件中的**INDEX_DEFAULT_MEM**。|
|slim|bool|AscendIndexConfig成员变量，是否动态增加内存。|
|filterable|bool|AscendIndexConfig成员变量，是否按照id进行过滤。|
|dBlockSize|uint32_t|配置Device侧的blockSize。|


**接口说明<a name="section1197816229504"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexConfig()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexConfig默认构造函数，默认指定的deviceList为0（即指定NPU的第0个<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>作为AscendFaiss执行检索的异构计算平台），默认的资源池大小为32MB（32*1024*1024字节）。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
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

<a name="table0786126165110"></a>
<table><tbody><tr id="row2787106115110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16787196145110"><a name="p16787196145110"></a><a name="p16787196145110"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1370313297588"><a name="p1370313297588"></a><a name="p1370313297588"></a>AscendIndexConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row378710616519"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p278776125114"><a name="p278776125114"></a><a name="p278776125114"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p978718675120"><a name="p978718675120"></a><a name="p978718675120"></a>AscendIndexConfig的构造函数，生成AscendIndexConfig，此时根据<span class="parmname" id="parmname16341331367"><a name="parmname16341331367"></a><a name="parmname16341331367"></a>“devices”</span>中配置的值设置Device侧<span id="ph8787176125114"><a name="ph8787176125114"></a><a name="ph8787176125114"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row167879675117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p6787136185119"><a name="p6787136185119"></a><a name="p6787136185119"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p578796165119"><a name="p578796165119"></a><a name="p578796165119"></a><strong id="b7745577018"><a name="b7745577018"></a><a name="b7745577018"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b120518131001"><a name="b120518131001"></a><a name="b120518131001"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname1331442416116"><a name="parmname1331442416116"></a><a name="parmname1331442416116"></a>“INDEX_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p45336291716"><a name="p45336291716"></a><a name="p45336291716"></a><strong id="b138053362110"><a name="b138053362110"></a><a name="b138053362110"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname12815818101613"><a name="parmname12815818101613"></a><a name="parmname12815818101613"></a>“DEFAULT_BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
</td>
</tr>
<tr id="row6787166165117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p77879695115"><a name="p77879695115"></a><a name="p77879695115"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1278716612512"><a name="p1278716612512"></a><a name="p1278716612512"></a>无</p>
</td>
</tr>
<tr id="row1787116135119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1278711610519"><a name="p1278711610519"></a><a name="p1278711610519"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p4787166125116"><a name="p4787166125116"></a><a name="p4787166125116"></a>无</p>
</td>
</tr>
<tr id="row87873611514"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2787126195113"><a name="p2787126195113"></a><a name="p2787126195113"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul5609290011"></a><a name="ul5609290011"></a><ul id="ul5609290011"><li><span class="parmname" id="parmname869315371603"><a name="parmname869315371603"></a><a name="parmname869315371603"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname460517421216"><a name="parmname460517421216"></a><a name="parmname460517421216"></a>“resources”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table23967285518"></a>
<table><tbody><tr id="row17396102845111"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.1.1"><p id="p83961128145117"><a name="p83961128145117"></a><a name="p83961128145117"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.1.1 "><p id="p894518112592"><a name="p894518112592"></a><a name="p894518112592"></a>AscendIndexConfig(std::vector&lt;int&gt; devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row03962028165110"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.2.1"><p id="p439632811517"><a name="p439632811517"></a><a name="p439632811517"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.2.1 "><p id="p163968283515"><a name="p163968283515"></a><a name="p163968283515"></a>AscendIndexConfig的构造函数，生成AscendIndexConfig，此时根据<span class="parmname" id="parmname1778119110614"><a name="parmname1778119110614"></a><a name="parmname1778119110614"></a>“devices”</span>中配置的值设置Device侧<span id="ph1039613285519"><a name="ph1039613285519"></a><a name="ph1039613285519"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row2396172875119"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.3.1"><p id="p12396228175120"><a name="p12396228175120"></a><a name="p12396228175120"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.3.1 "><p id="p43961828195110"><a name="p43961828195110"></a><a name="p43961828195110"></a><strong id="b183551822226"><a name="b183551822226"></a><a name="b183551822226"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p839622813512"><a name="p839622813512"></a><a name="p839622813512"></a><strong id="b1160625821"><a name="b1160625821"></a><a name="b1160625821"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname14396928125116"><a name="parmname14396928125116"></a><a name="parmname14396928125116"></a>“INDEX_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p192819288016"><a name="p192819288016"></a><a name="p192819288016"></a><strong id="b07647394111"><a name="b07647394111"></a><a name="b07647394111"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname9394163317166"><a name="parmname9394163317166"></a><a name="parmname9394163317166"></a>“DEFAULT_BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
</td>
</tr>
<tr id="row9396182816510"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.4.1"><p id="p18396152865113"><a name="p18396152865113"></a><a name="p18396152865113"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.4.1 "><p id="p1539616283512"><a name="p1539616283512"></a><a name="p1539616283512"></a>无</p>
</td>
</tr>
<tr id="row1439642825118"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.5.1"><p id="p2396928115112"><a name="p2396928115112"></a><a name="p2396928115112"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.5.1 "><p id="p1939617284511"><a name="p1939617284511"></a><a name="p1939617284511"></a>无</p>
</td>
</tr>
<tr id="row12396182811516"><th class="firstcol" valign="top" width="19.91%" id="mcps1.1.3.6.1"><p id="p1439682811514"><a name="p1439682811514"></a><a name="p1439682811514"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.08999999999999%" headers="mcps1.1.3.6.1 "><a name="ul133383413220"></a><a name="ul133383413220"></a><ul id="ul133383413220"><li><span class="parmname" id="parmname1593203817210"><a name="parmname1593203817210"></a><a name="parmname1593203817210"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname11396162818517"><a name="parmname11396162818517"></a><a name="parmname11396162818517"></a>“resources”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>


### AscendIndexInt8<a id="ZH-CN_TOPIC_0000001506495841"></a>

#### 功能介绍<a id="ZH-CN_TOPIC_0000001506495913"></a>

AscendIndexInt8作为特征检索组件中的采用INT8特征向量的Index的基类，为特征检索中的其他INT8的Index定义接口。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。


#### add接口<a name="ZH-CN_TOPIC_0000001506334825"></a>

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

> [!NOTE] 说明
>-   add接口不能与add\_with\_ids接口混用。
>-   使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。


#### add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001506614905"></a>

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


#### assign接口<a name="ZH-CN_TOPIC_0000001506495721"></a>

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


#### AscendIndexInt8接口<a name="ZH-CN_TOPIC_0000001506614993"></a>

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


#### getDeviceList接口<a name="ZH-CN_TOPIC_0000001672982421"></a>

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


#### getDim接口<a name="ZH-CN_TOPIC_0000001690599922"></a>

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


#### getNTotal接口<a name="ZH-CN_TOPIC_0000001738718517"></a>

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


#### getMetricType接口<a name="ZH-CN_TOPIC_0000001738678653"></a>

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


#### isTrained接口<a name="ZH-CN_TOPIC_0000001690759666"></a>

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


#### operator = 接口<a name="ZH-CN_TOPIC_0000001506414841"></a>

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


#### reclaimMemory接口<a name="ZH-CN_TOPIC_0000001506615133"></a>

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


#### remove\_ids接口<a name="ZH-CN_TOPIC_0000001456695088"></a>

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


#### reserveMemory接口<a name="ZH-CN_TOPIC_0000001506615065"></a>

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


#### search接口<a name="ZH-CN_TOPIC_0000001506414889"></a>

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


#### train接口<a name="ZH-CN_TOPIC_0000001456534956"></a>

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


#### updateCentroids接口<a name="ZH-CN_TOPIC_0000001506414833"></a>

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



### AscendIndexInt8Config<a id="ZH-CN_TOPIC_0000001456854968"></a>

AscendIndexInt8需要使用对应的AscendIndexInt8Config执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|deviceList|std::vector<int>|Device侧设备ID。|
|resourceSize|int64_t|设备侧预置的内存池大小，单位为字节。|


**接口说明<a name="section135441937164218"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexInt8Config()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8Config的默认构造函数，默认指定的deviceList为0（即指定NPU的第0个<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>作为AscendFaiss执行检索的异构计算平台），采用默认的资源池大小。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
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

<a name="table012165162914"></a>
<table><tbody><tr id="row71210582913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p31219512297"><a name="p31219512297"></a><a name="p31219512297"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132201932191815"><a name="p132201932191815"></a><a name="p132201932191815"></a>AscendIndexInt8Config(std::initializer_list&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row212554294"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1012185102912"><a name="p1012185102912"></a><a name="p1012185102912"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p71214510295"><a name="p71214510295"></a><a name="p71214510295"></a>AscendIndexInt8Config的构造函数，生成AscendIndexInt8Config，此时根据<span class="parmname" id="parmname131034538596"><a name="parmname131034538596"></a><a name="parmname131034538596"></a>“devices”</span>中配置的值设置Device侧<span id="ph151213517298"><a name="ph151213517298"></a><a name="ph151213517298"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row101210562912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1612856291"><a name="p1612856291"></a><a name="p1612856291"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p13185103613195"><a name="p13185103613195"></a><a name="p13185103613195"></a><strong id="b17802134542320"><a name="b17802134542320"></a><a name="b17802134542320"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b139931451162315"><a name="b139931451162315"></a><a name="b139931451162315"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname19157165610231"><a name="parmname19157165610231"></a><a name="parmname19157165610231"></a>“INDEX_INT8_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row201311582910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7139542914"><a name="p7139542914"></a><a name="p7139542914"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p913145182911"><a name="p913145182911"></a><a name="p913145182911"></a>无</p>
</td>
</tr>
<tr id="row61312510298"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6137572910"><a name="p6137572910"></a><a name="p6137572910"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7131256297"><a name="p7131256297"></a><a name="p7131256297"></a>无</p>
</td>
</tr>
<tr id="row81316519296"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p41316552918"><a name="p41316552918"></a><a name="p41316552918"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul12323102516242"></a><a name="ul12323102516242"></a><ul id="ul12323102516242"><li><span class="parmname" id="parmname1876622712243"><a name="parmname1876622712243"></a><a name="parmname1876622712243"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname460517421216"><a name="parmname460517421216"></a><a name="parmname460517421216"></a>“resources”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table9202719152913"></a>
<table><tbody><tr id="row620221922910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p152021219162918"><a name="p152021219162918"></a><a name="p152021219162918"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p620271952915"><a name="p620271952915"></a><a name="p620271952915"></a>AscendIndexInt8Config(std::vector&lt;int&gt; devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row720217191294"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7202151915293"><a name="p7202151915293"></a><a name="p7202151915293"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8202419152912"><a name="p8202419152912"></a><a name="p8202419152912"></a>AscendIndexInt8Config的构造函数，生成AscendIndexInt8Config，此时根据<span class="parmname" id="parmname55384581598"><a name="parmname55384581598"></a><a name="parmname55384581598"></a>“devices”</span>中配置的值设置Device侧<span id="ph182026195293"><a name="ph182026195293"></a><a name="ph182026195293"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row7202101919297"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p02021219172918"><a name="p02021219172918"></a><a name="p02021219172918"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p17202319192913"><a name="p17202319192913"></a><a name="p17202319192913"></a><strong id="b3189458112410"><a name="b3189458112410"></a><a name="b3189458112410"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p32021619132913"><a name="p32021619132913"></a><a name="p32021619132913"></a><strong id="b18933954250"><a name="b18933954250"></a><a name="b18933954250"></a>int64_t resources</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname514412106256"><a name="parmname514412106256"></a><a name="parmname514412106256"></a>“INDEX_INT8_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
</td>
</tr>
<tr id="row22021519142913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p202027199296"><a name="p202027199296"></a><a name="p202027199296"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p202021619112918"><a name="p202021619112918"></a><a name="p202021619112918"></a>无</p>
</td>
</tr>
<tr id="row120218193297"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2202171902914"><a name="p2202171902914"></a><a name="p2202171902914"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1720211192299"><a name="p1720211192299"></a><a name="p1720211192299"></a>无</p>
</td>
</tr>
<tr id="row520291962913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1720213191294"><a name="p1720213191294"></a><a name="p1720213191294"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul373416141258"></a><a name="ul373416141258"></a><ul id="ul373416141258"><li><span class="parmname" id="parmname67839163255"><a name="parmname67839163255"></a><a name="parmname67839163255"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname112023193297"><a name="parmname112023193297"></a><a name="parmname112023193297"></a>“resources”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节）。</li></ul>
</td>
</tr>
</tbody>
</table>


### AscendIndexInt8Flat<a name="ZH-CN_TOPIC_0000001506334741"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001506615033"></a>

AscendIndexInt8Flat存储INT8类型特征向量并执行暴力搜索。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。


#### AscendIndexInt8Flat接口<a name="ZH-CN_TOPIC_0000001456375168"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p43030218474"><a name="p43030218474"></a><a name="p43030218474"></a>AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8Flat的构造函数，生成维度为dims的AscendIndexInt8（单个Index管理的一组向量的维度是唯一的），此时根据config中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b13765143611523"><a name="b13765143611523"></a><a name="b13765143611523"></a>int dims</strong>：AscendIndexInt8管理的一组特征向量的维度。</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1884823913524"><a name="b1884823913524"></a><a name="b1884823913524"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b713664225211"><a name="b713664225211"></a><a name="b713664225211"></a>AscendIndexInt8FlatConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul246474615523"></a><a name="ul246474615523"></a><ul id="ul246474615523"><li>dims ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table08035919302"></a>
<table><tbody><tr id="row280317933013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p080310943012"><a name="p080310943012"></a><a name="p080310943012"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a>AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</p>
</td>
</tr>
<tr id="row1880379113018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p128039983019"><a name="p128039983019"></a><a name="p128039983019"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p28031693305"><a name="p28031693305"></a><a name="p28031693305"></a>AscendIndexInt8Flat的构造函数，基于一个已有的<span class="parmname" id="parmname366313478466"><a name="parmname366313478466"></a><a name="parmname366313478466"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row168031396307"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p13803993309"><a name="p13803993309"></a><a name="p13803993309"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p198038933015"><a name="p198038933015"></a><a name="p198038933015"></a><strong id="b5607121845310"><a name="b5607121845310"></a><a name="b5607121845310"></a>const faiss::IndexScalarQuantizer *index</strong>：CPU侧Index资源。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1545613219539"><a name="b1545613219539"></a><a name="b1545613219539"></a>AscendIndexInt8FlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row1580359153014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p88038983010"><a name="p88038983010"></a><a name="p88038983010"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p138034993013"><a name="p138034993013"></a><a name="p138034993013"></a>无</p>
</td>
</tr>
<tr id="row13803119153019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p16803159103018"><a name="p16803159103018"></a><a name="p16803159103018"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p108037918306"><a name="p108037918306"></a><a name="p108037918306"></a>无</p>
</td>
</tr>
<tr id="row78038912309"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p88031919307"><a name="p88031919307"></a><a name="p88031919307"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname105161728105317"><a name="parmname105161728105317"></a><a name="parmname105161728105317"></a>“index”</span>需要为合法有效的CPU Index指针，必须为<strong id="b17970185610179"><a name="b17970185610179"></a><a name="b17970185610179"></a>AscendIndexInt8Flat</strong>执行<strong id="b470691181811"><a name="b470691181811"></a><a name="b470691181811"></a>copyTo</strong>接口生成的faiss::IndexScalarQuantizer类型指针。</p>
</td>
</tr>
</tbody>
</table>

<a name="table11312020103012"></a>
<table><tbody><tr id="row18131520123011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p51314208305"><a name="p51314208305"></a><a name="p51314208305"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p134321816154815"><a name="p134321816154815"></a><a name="p134321816154815"></a>AscendIndexInt8Flat(const faiss::IndexIDMap*index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());</p>
</td>
</tr>
<tr id="row14131152033015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p171311220173012"><a name="p171311220173012"></a><a name="p171311220173012"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16131202010306"><a name="p16131202010306"></a><a name="p16131202010306"></a>AscendIndexInt8Flat的构造函数，基于一个已有的<span class="parmname" id="parmname9417357194619"><a name="parmname9417357194619"></a><a name="parmname9417357194619"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row213118206301"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p613111201309"><a name="p613111201309"></a><a name="p613111201309"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1131192014300"><a name="p1131192014300"></a><a name="p1131192014300"></a><strong id="b7365124165315"><a name="b7365124165315"></a><a name="b7365124165315"></a>const faiss::IndexIDMap*index</strong>：CPU侧Index资源。</p>
<p id="p01311220103014"><a name="p01311220103014"></a><a name="p01311220103014"></a><strong id="b7835143185317"><a name="b7835143185317"></a><a name="b7835143185317"></a>AscendIndexInt8FlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row1113242019308"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14132132015303"><a name="p14132132015303"></a><a name="p14132132015303"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8132220193011"><a name="p8132220193011"></a><a name="p8132220193011"></a>无</p>
</td>
</tr>
<tr id="row8132132093017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p91321620163010"><a name="p91321620163010"></a><a name="p91321620163010"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1132720163012"><a name="p1132720163012"></a><a name="p1132720163012"></a>无</p>
</td>
</tr>
<tr id="row12132820203018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2132142003020"><a name="p2132142003020"></a><a name="p2132142003020"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p313222019308"><a name="p313222019308"></a><a name="p313222019308"></a><span class="parmname" id="parmname153575476538"><a name="parmname153575476538"></a><a name="parmname153575476538"></a>“index”</span>需要为合法有效的CPU Index指针，必须为<strong id="b13735204215184"><a name="b13735204215184"></a><a name="b13735204215184"></a>AscendIndexInt8Flat</strong>执行<strong id="b1583184418185"><a name="b1583184418185"></a><a name="b1583184418185"></a>copyTo</strong>接口生成的faiss::IndexIDMap类型指针。</p>
</td>
</tr>
</tbody>
</table>

<a name="table186285584308"></a>
<table><tbody><tr id="row11628358133010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p14628358193015"><a name="p14628358193015"></a><a name="p14628358193015"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8Flat(const AscendIndexInt8Flat&amp;) = delete;</p>
</td>
</tr>
<tr id="row1362814589304"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p5628195813011"><a name="p5628195813011"></a><a name="p5628195813011"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p96281058103020"><a name="p96281058103020"></a><a name="p96281058103020"></a>声明此Index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row56281058123019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1962825833011"><a name="p1962825833011"></a><a name="p1962825833011"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b11456125611539"><a name="b11456125611539"></a><a name="b11456125611539"></a>const AscendIndexInt8Flat&amp;</strong>：常量AscendIndexInt8Flat。</p>
</td>
</tr>
<tr id="row16281558103012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p36281758163011"><a name="p36281758163011"></a><a name="p36281758163011"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p86282058183018"><a name="p86282058183018"></a><a name="p86282058183018"></a>无</p>
</td>
</tr>
<tr id="row6628175820307"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p662825863011"><a name="p662825863011"></a><a name="p662825863011"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p6628658183012"><a name="p6628658183012"></a><a name="p6628658183012"></a>无</p>
</td>
</tr>
<tr id="row12628958103010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p6628195853016"><a name="p6628195853016"></a><a name="p6628195853016"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p962855810301"><a name="p962855810301"></a><a name="p962855810301"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table206471151315"></a>
<table><tbody><tr id="row564841517316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p464801510315"><a name="p464801510315"></a><a name="p464801510315"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexInt8Flat();</p>
</td>
</tr>
<tr id="row156481015103116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1364813150319"><a name="p1364813150319"></a><a name="p1364813150319"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1064810155319"><a name="p1064810155319"></a><a name="p1064810155319"></a>AscendIndexInt8Flat的析构函数，销毁AscendIndexInt8Flat对象，释放资源。</p>
</td>
</tr>
<tr id="row1564851515314"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1648181512311"><a name="p1648181512311"></a><a name="p1648181512311"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row156481215103116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1264891513116"><a name="p1264891513116"></a><a name="p1264891513116"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p7648121583112"><a name="p7648121583112"></a><a name="p7648121583112"></a>无</p>
</td>
</tr>
<tr id="row564812154316"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p7648111593118"><a name="p7648111593118"></a><a name="p7648111593118"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p064841512313"><a name="p064841512313"></a><a name="p064841512313"></a>无</p>
</td>
</tr>
<tr id="row3648915103115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p86481115153115"><a name="p86481115153115"></a><a name="p86481115153115"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1364861518319"><a name="p1364861518319"></a><a name="p1364861518319"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### copyFrom接口<a name="ZH-CN_TOPIC_0000001456375340"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyFrom(const faiss::IndexIDMap* index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8Flat基于一个已有的<span class="parmname" id="parmname18533121195111"><a name="parmname18533121195111"></a><a name="parmname18533121195111"></a>“index”</span>拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b7731624145618"><a name="b7731624145618"></a><a name="b7731624145618"></a>const faiss::IndexIDMap*index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1742162695618"><a name="parmname1742162695618"></a><a name="parmname1742162695618"></a>“index”</span>需要为合法有效的IndexIDMap指针，该Index的成员索引维度d参数取值范围为{64, 128, 256, 384, 512, 768, 1024}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table862731073217"></a>
<table><tbody><tr id="row562716101326"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p6627201073216"><a name="p6627201073216"></a><a name="p6627201073216"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6908132519515"><a name="p6908132519515"></a><a name="p6908132519515"></a>void copyFrom(const faiss::IndexScalarQuantizer* index);</p>
</td>
</tr>
<tr id="row1562701093213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1962719100320"><a name="p1962719100320"></a><a name="p1962719100320"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16271610153210"><a name="p16271610153210"></a><a name="p16271610153210"></a>AscendIndexInt8Flat基于一个已有的<span class="parmname" id="parmname10631110135119"><a name="parmname10631110135119"></a><a name="parmname10631110135119"></a>“index”</span>拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row6627181014329"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p20627610203213"><a name="p20627610203213"></a><a name="p20627610203213"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p126277101327"><a name="p126277101327"></a><a name="p126277101327"></a><strong id="b6870101316567"><a name="b6870101316567"></a><a name="b6870101316567"></a>const faiss::IndexScalarQuantizer* index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="row362771017325"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p262713107326"><a name="p262713107326"></a><a name="p262713107326"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1762761014326"><a name="p1762761014326"></a><a name="p1762761014326"></a>无</p>
</td>
</tr>
<tr id="row18627710123214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p15627310183210"><a name="p15627310183210"></a><a name="p15627310183210"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1362715109324"><a name="p1362715109324"></a><a name="p1362715109324"></a>无</p>
</td>
</tr>
<tr id="row18627510133211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p26271100323"><a name="p26271100323"></a><a name="p26271100323"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1562781003214"><a name="p1562781003214"></a><a name="p1562781003214"></a><span class="parmname" id="parmname55631616145615"><a name="parmname55631616145615"></a><a name="parmname55631616145615"></a>“index”</span>需要为合法有效的CPU Index指针，Index的维度d参数取值范围为{64, 128, 256, 384, 512, 768, 1024}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>


#### copyTo接口<a name="ZH-CN_TOPIC_0000001506334805"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexScalarQuantizer* index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>将AscendIndexInt8Flat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b419043435615"><a name="b419043435615"></a><a name="b419043435615"></a>faiss::IndexScalarQuantizer* index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname947410378567"><a name="parmname947410378567"></a><a name="parmname947410378567"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1981952413329"></a>
<table><tbody><tr id="row6819122423218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9819112423211"><a name="p9819112423211"></a><a name="p9819112423211"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1981912463211"><a name="p1981912463211"></a><a name="p1981912463211"></a>void copyTo(faiss::IndexIDMap* index) const;</p>
</td>
</tr>
<tr id="row128191424163217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p16819224173215"><a name="p16819224173215"></a><a name="p16819224173215"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p28192024163218"><a name="p28192024163218"></a><a name="p28192024163218"></a>将AscendIndexInt8Flat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row1281910243329"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p188196241328"><a name="p188196241328"></a><a name="p188196241328"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p281916241321"><a name="p281916241321"></a><a name="p281916241321"></a><strong id="b2771344195619"><a name="b2771344195619"></a><a name="b2771344195619"></a>faiss::IndexIDMap*index</strong>：CPU侧index资源。</p>
</td>
</tr>
<tr id="row2819182413219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p281972453215"><a name="p281972453215"></a><a name="p281972453215"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6819192473220"><a name="p6819192473220"></a><a name="p6819192473220"></a>无</p>
</td>
</tr>
<tr id="row14819152483212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p4819024113219"><a name="p4819024113219"></a><a name="p4819024113219"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p178191624133216"><a name="p178191624133216"></a><a name="p178191624133216"></a>无</p>
</td>
</tr>
<tr id="row381919240326"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p78191224173213"><a name="p78191224173213"></a><a name="p78191224173213"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4819132473214"><a name="p4819132473214"></a><a name="p4819132473214"></a><span class="parmname" id="parmname71714475560"><a name="parmname71714475560"></a><a name="parmname71714475560"></a>“index”</span>需要为合法有效的IndexIDMap指针，index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>


#### getBase接口<a name="ZH-CN_TOPIC_0000001506334753"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getBase(int deviceId, std::vector&lt;int8_t&gt; &amp;xb) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexInt8Flat在特定<span class="parmname" id="parmname1195143920472"><a name="parmname1195143920472"></a><a name="parmname1195143920472"></a>“deviceId”</span>上管理的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b196131849175412"><a name="b196131849175412"></a><a name="b196131849175412"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b07321515549"><a name="b07321515549"></a><a name="b07321515549"></a>std::vector&lt;int8_t&gt; &amp;xb</strong>：AscendIndexInt8Flat在<span class="parmname" id="parmname20870161012482"><a name="parmname20870161012482"></a><a name="parmname20870161012482"></a>“deviceId”</span>上存储的底库特征向量。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname112791054165415"><a name="parmname112791054165415"></a><a name="parmname112791054165415"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>


#### getBaseSize接口<a name="ZH-CN_TOPIC_0000001506414709"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>size_t getBaseSize(int deviceId) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexInt8Flat在特定<span class="parmname" id="parmname141001228114714"><a name="parmname141001228114714"></a><a name="parmname141001228114714"></a>“deviceId”</span>上管理的特征向量数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b153555291543"><a name="b153555291543"></a><a name="b153555291543"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>在特定<span class="parmname" id="parmname1298021824811"><a name="parmname1298021824811"></a><a name="parmname1298021824811"></a>“deviceId”</span>上的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname20741633175413"><a name="parmname20741633175413"></a><a name="parmname20741633175413"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>


#### getIdxMap接口<a name="ZH-CN_TOPIC_0000001506495853"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexInt8Flat在特定<span class="parmname" id="parmname7542659164914"><a name="parmname7542659164914"></a><a name="parmname7542659164914"></a>“deviceId”</span>上管理的特征向量ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b1775544965512"><a name="b1775544965512"></a><a name="b1775544965512"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b632965711198"><a name="b632965711198"></a><a name="b632965711198"></a>std::vector&lt;idx_t&gt; &amp;idxMap</strong>：AscendIndexInt8Flat在<span class="parmname" id="parmname62818562490"><a name="parmname62818562490"></a><a name="parmname62818562490"></a>“deviceId”</span>上存储的底库特征向量ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.54%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.46%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname498819558551"><a name="parmname498819558551"></a><a name="parmname498819558551"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000001506414909"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161241236141910"><a name="p161241236141910"></a><a name="p161241236141910"></a>AscendIndexInt8Flat&amp; operator=(const AscendIndexInt8Flat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b2040411013542"><a name="b2040411013542"></a><a name="b2040411013542"></a>const AscendIndexInt8Flat&amp;</strong>：常量AscendIndexInt8Flat。</p>
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


#### reset接口<a name="ZH-CN_TOPIC_0000001506495889"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void reset();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>清空该AscendIndexInt8Flat的底库向量。</p>
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


#### search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001456694912"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void search_with_masks(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实现AscendIndexInt8特征向量查询接口，根据输入的特征向量以及<span class="parmname" id="parmname69861138105118"><a name="parmname69861138105118"></a><a name="parmname69861138105118"></a>“mask”</span>掩码返回最相似的<span class="parmname" id="parmname19493733155119"><a name="parmname19493733155119"></a><a name="parmname19493733155119"></a>“k”</span>条特征的距离及ID。mask为<strong id="b14721249104814"><a name="b14721249104814"></a><a name="b14721249104814"></a>0</strong>、<strong id="b596515503483"><a name="b596515503483"></a><a name="b596515503483"></a>1</strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong id="b295210616496"><a name="b295210616496"></a><a name="b295210616496"></a>1</strong>参与，<strong id="b93801987491"><a name="b93801987491"></a><a name="b93801987491"></a>0</strong>不参与。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a><strong id="b186310558563"><a name="b186310558563"></a><a name="b186310558563"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1587514917458"><a name="p1587514917458"></a><a name="p1587514917458"></a><strong id="b531925911561"><a name="b531925911561"></a><a name="b531925911561"></a>const int8_t* x</strong>：特征向量数据。</p>
<p id="p127711649459"><a name="p127711649459"></a><a name="p127711649459"></a><strong id="b7970109576"><a name="b7970109576"></a><a name="b7970109576"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p1838795616530"><a name="p1838795616530"></a><a name="p1838795616530"></a><strong id="b5798929577"><a name="b5798929577"></a><a name="b5798929577"></a>const void* mask</strong>：底库的过滤掩码。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b113310163576"><a name="b113310163576"></a><a name="b113310163576"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname64605085110"><a name="parmname64605085110"></a><a name="parmname64605085110"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b179807179572"><a name="b179807179572"></a><a name="b179807179572"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname772411523519"><a name="parmname772411523519"></a><a name="parmname772411523519"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul197738222579"></a><a name="ul197738222579"></a><ul id="ul197738222579"><li><span class="parmname" id="parmname95757582519"><a name="parmname95757582519"></a><a name="parmname95757582519"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li><span class="parmname" id="parmname382711402810"><a name="parmname382711402810"></a><a name="parmname382711402810"></a>“k”</span>通常不允许超过4096。</li><li>指针<span class="parmname" id="parmname17661432105710"><a name="parmname17661432105710"></a><a name="parmname17661432105710"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>指针<span class="parmname" id="parmname167723415714"><a name="parmname167723415714"></a><a name="parmname167723415714"></a>“distances”</span>/<span class="parmname" id="parmname193331836115710"><a name="parmname193331836115710"></a><a name="parmname193331836115710"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b5379163513393"><a name="b5379163513393"></a><a name="b5379163513393"></a>k</strong> * <strong id="b19874738123915"><a name="b19874738123915"></a><a name="b19874738123915"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>指针<span class="parmname" id="parmname187912297571"><a name="parmname187912297571"></a><a name="parmname187912297571"></a>“mask”</span>需要为非空指针，需保证传入的掩码长度为⌈ntotal / 8⌉ * n（<span class="parmname" id="parmname8177115424411"><a name="parmname8177115424411"></a><a name="parmname8177115424411"></a>“ntotal”</span>为底库向量的条数）。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>


#### setPageSize接口<a name="ZH-CN_TOPIC_0000002007453769"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p20912871594"><a name="p20912871594"></a><a name="p20912871594"></a>void setPageSize(uint16_t pageBlockNum);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p393922219912"><a name="p393922219912"></a><a name="p393922219912"></a>设置该AscendIndexInt8Flat在search时一次性连续计算底库的block数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1836319401096"><a name="p1836319401096"></a><a name="p1836319401096"></a><strong id="b16962165461012"><a name="b16962165461012"></a><a name="b16962165461012"></a>uint16_t pageBlockNum</strong>：一次性连续计算底库的block数量。不设置时，默认一次性连续计算16个block。一个block存储向量的大小由AscendIndexInt8FlatConfig中的blockSize决定。该值越大，search时占用的内存越大。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul250991973317"></a><a name="ul250991973317"></a><ul id="ul250991973317"><li><span class="parmname" id="parmname159984545424"><a name="parmname159984545424"></a><a name="parmname159984545424"></a>“pageBlockNum”</span>的取值范围：0 &lt; pageBlockNum ≤ 144</li><li>该接口主要用于大底库场景，search接口性能调优使用。该值越大，占用AscendIndexInt8FlatConfig中配置的resourceSize预置内存越大。建议申请足够大的预置内存，再利用该接口进行参数调优。</li></ul>
</td>
</tr>
</tbody>
</table>



### AscendIndexInt8FlatConfig<a name="ZH-CN_TOPIC_0000001456535040"></a>

AscendIndexInt8Flat需要使用对应的AscendIndexInt8FlatConfig执行对应资源的初始化。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|dIndexMode|Int8IndexMode|配置Index int8检索模式。|
|dBlockSize|uint32_t|配置Device侧的blockSize。|


**接口说明<a name="section136272015172914"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexInt8FlatConfig(uint32_t blockSize =BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，配置Device侧blockSize，配置int8的检索模式。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b12302102216503"><a name="b12302102216503"></a><a name="b12302102216503"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname241445820209"><a name="parmname241445820209"></a><a name="parmname241445820209"></a>“BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
<p id="p196421349716"><a name="p196421349716"></a><a name="p196421349716"></a><strong id="b49637218593"><a name="b49637218593"></a><a name="b49637218593"></a>Int8IndexMode indexMode</strong>：配置Index int8检索模式。默认值为<strong id="b13424175612711"><a name="b13424175612711"></a><a name="b13424175612711"></a>DEFAULT_MODE</strong>。</p>
<a name="ul112020291075"></a><a name="ul112020291075"></a><ul id="ul112020291075"><li><strong id="b8674833612"><a name="b8674833612"></a><a name="b8674833612"></a>DEFAULT_MODE</strong>模式，默认模式。</li><li><strong id="b197655512712"><a name="b197655512712"></a><a name="b197655512712"></a>PIPE_SEARCH_MODE</strong>模式，该模式针对batch大于或等于<strong id="b925325991216"><a name="b925325991216"></a><a name="b925325991216"></a>128</strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB<strong id="b142221114490"><a name="b142221114490"></a><a name="b142221114490"></a>。</strong></li><li><strong id="b193311527712"><a name="b193311527712"></a><a name="b193311527712"></a>WITHOUT_NORM_MODE</strong>模式，暂时不支持本模式。</li></ul>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul72731322134614"></a><a name="ul72731322134614"></a><ul id="ul72731322134614"><li><span class="parmname" id="parmname567764215019"><a name="parmname567764215019"></a><a name="parmname567764215019"></a>“blockSize”</span>可配置的值的集合为{16384， 32768， 65536， 131072， 262144}</li></ul>
<a name="ul17184174914612"></a><a name="ul17184174914612"></a><ul id="ul17184174914612"><li><strong id="b535836183515"><a name="b535836183515"></a><a name="b535836183515"></a><span class="parmname" id="parmname346219387426"><a name="parmname346219387426"></a><a name="parmname346219387426"></a>“indexMode”</span></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1258103643012"></a>
<table><tbody><tr id="row95803619306"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p15853617303"><a name="p15853617303"></a><a name="p15853617303"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p558236163018"><a name="p558236163018"></a><a name="p558236163018"></a>AscendIndexInt8FlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);</p>
</td>
</tr>
<tr id="row10580363301"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1058436123019"><a name="p1058436123019"></a><a name="p1058436123019"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p459183617305"><a name="p459183617305"></a><a name="p459183617305"></a>AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，此时根据Devices中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源，配置资源池大小。配置Device侧blockSize，配置int8的检索模式。</p>
</td>
</tr>
<tr id="row9592036113014"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p145953643012"><a name="p145953643012"></a><a name="p145953643012"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p165943663010"><a name="p165943663010"></a><a name="p165943663010"></a><strong id="b56134644720"><a name="b56134644720"></a><a name="b56134644720"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b10147012114711"><a name="b10147012114711"></a><a name="b10147012114711"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname143161514154715"><a name="parmname143161514154715"></a><a name="parmname143161514154715"></a>“INT8_FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置1024MB。</p>
<p id="p266920816252"><a name="p266920816252"></a><a name="p266920816252"></a><strong id="b14261165535013"><a name="b14261165535013"></a><a name="b14261165535013"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname13591036163017"><a name="parmname13591036163017"></a><a name="parmname13591036163017"></a>“BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
<p id="p920025214811"><a name="p920025214811"></a><a name="p920025214811"></a><strong id="b1359036103010"><a name="b1359036103010"></a><a name="b1359036103010"></a>Int8IndexMode indexMode</strong>：配置Index int8检索模式。默认值为<strong id="b109732583816"><a name="b109732583816"></a><a name="b109732583816"></a>DEFAULT_MODE</strong>。</p>
<a name="ul19317031497"></a><a name="ul19317031497"></a><ul id="ul19317031497"><li><strong id="b1220019526818"><a name="b1220019526818"></a><a name="b1220019526818"></a>DEFAULT_MODE</strong>模式，默认模式。</li><li><strong id="b162925151916"><a name="b162925151916"></a><a name="b162925151916"></a>PIPE_SEARCH_MODE</strong>模式，该模式针对batch大于或等于<strong id="b459736143011"><a name="b459736143011"></a><a name="b459736143011"></a>128</strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB。</li><li><strong id="b1772318287911"><a name="b1772318287911"></a><a name="b1772318287911"></a>WITHOUT_NORM_MODE</strong>模式，暂时不支持本模式。</li></ul>
</td>
</tr>
<tr id="row859836143020"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p359536193015"><a name="p359536193015"></a><a name="p359536193015"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p659336153018"><a name="p659336153018"></a><a name="p659336153018"></a>无</p>
</td>
</tr>
<tr id="row105953623010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p759536193011"><a name="p759536193011"></a><a name="p759536193011"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p195910363302"><a name="p195910363302"></a><a name="p195910363302"></a>无</p>
</td>
</tr>
<tr id="row1259173611301"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p1059173683011"><a name="p1059173683011"></a><a name="p1059173683011"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul07292414471"></a><a name="ul07292414471"></a><ul id="ul07292414471"><li><span class="parmname" id="parmname63243288473"><a name="parmname63243288473"></a><a name="parmname63243288473"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname9781626112312"><a name="parmname9781626112312"></a><a name="parmname9781626112312"></a>“resourceSize”</span>配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。当batch大于等于96时，为提升算法性能，建议<span class="parmname" id="parmname8997192411519"><a name="parmname8997192411519"></a><a name="parmname8997192411519"></a>“resourceSize”</span>设置为大于等于2 * 1024MB。</li><li><span class="parmname" id="parmname2376185113505"><a name="parmname2376185113505"></a><a name="parmname2376185113505"></a>“blockSize”</span>可配置的值的集合为{16384， 32768， 65536， 131072， 262144}</li><li><strong id="b259143623014"><a name="b259143623014"></a><a name="b259143623014"></a><span class="parmname" id="parmname759936143015"><a name="parmname759936143015"></a><a name="parmname759936143015"></a>“indexMode”</span></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table8629135217302"></a>
<table><tbody><tr id="row6630115223010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p126301152183015"><a name="p126301152183015"></a><a name="p126301152183015"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1363005243010"><a name="p1363005243010"></a><a name="p1363005243010"></a>AscendIndexInt8FlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)</p>
</td>
</tr>
<tr id="row1630175243019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p16630752193012"><a name="p16630752193012"></a><a name="p16630752193012"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p263012528307"><a name="p263012528307"></a><a name="p263012528307"></a>AscendIndexInt8FlatConfig的构造函数，生成AscendIndexInt8FlatConfig，此时根据<span class="parmname" id="parmname1810443014014"><a name="parmname1810443014014"></a><a name="parmname1810443014014"></a>“devices”</span>中配置的值设置Device侧<span id="ph763075283015"><a name="ph763075283015"></a><a name="ph763075283015"></a>昇腾AI处理器</span>资源，配置资源池大小。配置Device侧blockSize，配置int8的检索模式。</p>
</td>
</tr>
<tr id="row363011522300"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p196301452183019"><a name="p196301452183019"></a><a name="p196301452183019"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p4630205293012"><a name="p4630205293012"></a><a name="p4630205293012"></a><strong id="b658511585471"><a name="b658511585471"></a><a name="b658511585471"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p2063075223014"><a name="p2063075223014"></a><a name="p2063075223014"></a><strong id="b1075315534819"><a name="b1075315534819"></a><a name="b1075315534819"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname13771892480"><a name="parmname13771892480"></a><a name="parmname13771892480"></a>“INT8_FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置1024MB。</p>
<p id="p5467172216269"><a name="p5467172216269"></a><a name="p5467172216269"></a><strong id="b134751453101011"><a name="b134751453101011"></a><a name="b134751453101011"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size。默认值<span class="parmname" id="parmname4630052103015"><a name="parmname4630052103015"></a><a name="parmname4630052103015"></a>“BLOCK_SIZE”</span>为16384 * 16 = 262144。</p>
<p id="p38771541295"><a name="p38771541295"></a><a name="p38771541295"></a><strong id="b6630352153010"><a name="b6630352153010"></a><a name="b6630352153010"></a>Int8IndexMode indexMode</strong>：配置Index int8检索模式。默认值为<span class="parmvalue" id="parmvalue651015459212"><a name="parmvalue651015459212"></a><a name="parmvalue651015459212"></a>“DEFAULT_MODE”</span>。</p>
<a name="ul57151159696"></a><a name="ul57151159696"></a><ul id="ul57151159696"><li><strong id="b963075218304"><a name="b963075218304"></a><a name="b963075218304"></a>DEFAULT_MODE</strong>模式，默认模式。</li><li><strong id="b1936318711107"><a name="b1936318711107"></a><a name="b1936318711107"></a>PIPE_SEARCH_MODE</strong>模式，该模式针对batch大于或等于<strong id="b663015210304"><a name="b663015210304"></a><a name="b663015210304"></a>128</strong>的场景做了性能优化。使用该模式时，建议resourceSize至少配置为1324MB<strong id="b1763015217308"><a name="b1763015217308"></a><a name="b1763015217308"></a>。</strong></li><li><strong id="b11411131411019"><a name="b11411131411019"></a><a name="b11411131411019"></a>WITHOUT_NORM_MODE</strong>模式，暂时不支持本模式。</li></ul>
</td>
</tr>
<tr id="row20630135293014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p116301652203017"><a name="p116301652203017"></a><a name="p116301652203017"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15630952103017"><a name="p15630952103017"></a><a name="p15630952103017"></a>无</p>
</td>
</tr>
<tr id="row2630852173011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20630185213016"><a name="p20630185213016"></a><a name="p20630185213016"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p176301752133010"><a name="p176301752133010"></a><a name="p176301752133010"></a>无</p>
</td>
</tr>
<tr id="row156301552103017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9630105253015"><a name="p9630105253015"></a><a name="p9630105253015"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul115901014144816"></a><a name="ul115901014144816"></a><ul id="ul115901014144816"><li><span class="parmname" id="parmname34910164489"><a name="parmname34910164489"></a><a name="parmname34910164489"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname13630135263011"><a name="parmname13630135263011"></a><a name="parmname13630135263011"></a>“resourceSize”</span>配置的值不超过16 * 1024MB（16 * 1024 * 1024 * 1024字节）。当batch大于等于96时，为提升算法性能，建议<span class="parmname" id="parmname96301052143011"><a name="parmname96301052143011"></a><a name="parmname96301052143011"></a>“resourceSize”</span>设置为大于等于2 * 1024MB。</li><li><span class="parmname" id="parmname3456949185114"><a name="parmname3456949185114"></a><a name="parmname3456949185114"></a>“blockSize”</span>可配置的值的集合为{16384， 32768， 65536， 131072， 262144}。</li><li><strong id="b151606422049"><a name="b151606422049"></a><a name="b151606422049"></a><span class="parmname" id="parmname206301752163017"><a name="parmname206301752163017"></a><a name="parmname206301752163017"></a>“indexMode”</span></strong>中PIPE_SEARCH_MODE模式下的AscendIndexInt8Flat仅支持METRIC_L2。</li></ul>
</td>
</tr>
</tbody>
</table>


### AscendIndexFlat<a id="ZH-CN_TOPIC_0000001506334757"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001506334829"></a>

AscendIndexFlat是最基础的特征检索，存储FP16浮点数类型特征向量并执行暴力搜索。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

> [!NOTE] 说明 
>AscendIndexFlat算法L2和IP距离支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，使用在线算子需要用户在应用程序的最后显式调用 \(void\)aclFinalize\(\) （需要包含头文件：\#include "acl/acl.h"）。


#### AscendIndexFlat接口<a name="ZH-CN_TOPIC_0000001456375308"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a>AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexFlat的构造函数，基于一个已有的<span class="parmname" id="parmname186437475368"><a name="parmname186437475368"></a><a name="parmname186437475368"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b1735185317361"><a name="b1735185317361"></a><a name="b1735185317361"></a>const faiss::IndexFlat *index</strong>：CPU侧index资源。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1693555513368"><a name="b1693555513368"></a><a name="b1693555513368"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6239525111613"><a name="p6239525111613"></a><a name="p6239525111613"></a><span class="parmname" id="parmname721815874613"><a name="parmname721815874613"></a><a name="parmname721815874613"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1735274911381"></a>
<table><tbody><tr id="row163522495386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p113526492389"><a name="p113526492389"></a><a name="p113526492389"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p935216491386"><a name="p935216491386"></a><a name="p935216491386"></a>AscendIndexFlat(const faiss::IndexIDMap*index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="row14352124915385"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p2352749113819"><a name="p2352749113819"></a><a name="p2352749113819"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13528493381"><a name="p13528493381"></a><a name="p13528493381"></a>AscendIndexFlat的构造函数，基于一个已有的<span class="parmname" id="parmname1212316485395"><a name="parmname1212316485395"></a><a name="parmname1212316485395"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row13352184923815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p535219495381"><a name="p535219495381"></a><a name="p535219495381"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p11352164923819"><a name="p11352164923819"></a><a name="p11352164923819"></a><strong id="b37691251203911"><a name="b37691251203911"></a><a name="b37691251203911"></a>const faiss::IndexIDMap*index</strong>：CPU侧Index资源。</p>
<p id="p14352104915380"><a name="p14352104915380"></a><a name="p14352104915380"></a><strong id="b20797653143915"><a name="b20797653143915"></a><a name="b20797653143915"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row5352154943813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p12352114933817"><a name="p12352114933817"></a><a name="p12352114933817"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16352194953814"><a name="p16352194953814"></a><a name="p16352194953814"></a>无</p>
</td>
</tr>
<tr id="row1735214943812"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1535284912383"><a name="p1535284912383"></a><a name="p1535284912383"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p535264943818"><a name="p535264943818"></a><a name="p535264943818"></a>无</p>
</td>
</tr>
<tr id="row235216491381"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1035274933810"><a name="p1035274933810"></a><a name="p1035274933810"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname96759552468"><a name="parmname96759552468"></a><a name="parmname96759552468"></a>“index”</span>需要为合法有效的IndexIDMap指针，该Index的成员索引维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table142416323911"></a>
<table><tbody><tr id="row1257343916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p19251332393"><a name="p19251332393"></a><a name="p19251332393"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11207257504"><a name="p11207257504"></a><a name="p11207257504"></a>AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="row2258310398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p9252383918"><a name="p9252383918"></a><a name="p9252383918"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p102553123917"><a name="p102553123917"></a><a name="p102553123917"></a>AscendIndexFlat的构造函数，生成维度为dims的AscendIndexFlat（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="parmname1614731717426"><a name="parmname1614731717426"></a><a name="parmname1614731717426"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row1525633399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p16259393917"><a name="p16259393917"></a><a name="p16259393917"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10256318392"><a name="p10256318392"></a><a name="p10256318392"></a><strong id="b13406132154210"><a name="b13406132154210"></a><a name="b13406132154210"></a>int dims</strong>：AscendIndex管理的一组特征向量的维度。</p>
<p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1658993044220"><a name="b1658993044220"></a><a name="b1658993044220"></a>faiss::MetricType metric</strong>：AscendIndexFlat在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b159198328421"><a name="b159198328421"></a><a name="b159198328421"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row102514316397"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7254343920"><a name="p7254343920"></a><a name="p7254343920"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p22553153917"><a name="p22553153917"></a><a name="p22553153917"></a>无</p>
</td>
</tr>
<tr id="row22516313918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p62511393920"><a name="p62511393920"></a><a name="p62511393920"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p92519310395"><a name="p92519310395"></a><a name="p92519310395"></a>无</p>
</td>
</tr>
<tr id="row6251237398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p18251437396"><a name="p18251437396"></a><a name="p18251437396"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1988754314420"></a><a name="ul1988754314420"></a><ul id="ul1988754314420"><li>dims ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table5169814143913"></a>
<table><tbody><tr id="row1116961423914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p4169121473916"><a name="p4169121473916"></a><a name="p4169121473916"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7112274471"><a name="p7112274471"></a><a name="p7112274471"></a>AscendIndexFlat(const AscendIndexFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row1416991413916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p121699143394"><a name="p121699143394"></a><a name="p121699143394"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p191699142396"><a name="p191699142396"></a><a name="p191699142396"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row61691614163913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5169814123916"><a name="p5169814123916"></a><a name="p5169814123916"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b8275365435"><a name="b8275365435"></a><a name="b8275365435"></a>const AscendIndexFlat&amp;</strong>：常量AscendIndexFlat。</p>
</td>
</tr>
<tr id="row151691414153917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1016941403913"><a name="p1016941403913"></a><a name="p1016941403913"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p21691714113915"><a name="p21691714113915"></a><a name="p21691714113915"></a>无</p>
</td>
</tr>
<tr id="row181697141391"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3169161413395"><a name="p3169161413395"></a><a name="p3169161413395"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p01691714163917"><a name="p01691714163917"></a><a name="p01691714163917"></a>无</p>
</td>
</tr>
<tr id="row1416991443915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p01691014143911"><a name="p01691014143911"></a><a name="p01691014143911"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p161695143397"><a name="p161695143397"></a><a name="p161695143397"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table04891725153918"></a>
<table><tbody><tr id="row194894256391"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p13489525203913"><a name="p13489525203913"></a><a name="p13489525203913"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexFlat();</p>
</td>
</tr>
<tr id="row1248962513399"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p94891225163914"><a name="p94891225163914"></a><a name="p94891225163914"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1148902511397"><a name="p1148902511397"></a><a name="p1148902511397"></a>AscendIndexFlat的析构函数，销毁AscendIndexFlat对象，释放资源。</p>
</td>
</tr>
<tr id="row15489182583911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p10489142503915"><a name="p10489142503915"></a><a name="p10489142503915"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row6489525163919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p10489152514391"><a name="p10489152514391"></a><a name="p10489152514391"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p13489425133914"><a name="p13489425133914"></a><a name="p13489425133914"></a>无</p>
</td>
</tr>
<tr id="row1248992503912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p548912515395"><a name="p548912515395"></a><a name="p548912515395"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16489182512392"><a name="p16489182512392"></a><a name="p16489182512392"></a>无</p>
</td>
</tr>
<tr id="row16489725193918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p134891125183917"><a name="p134891125183917"></a><a name="p134891125183917"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15489102517393"><a name="p15489102517393"></a><a name="p15489102517393"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### copyFrom接口<a name="ZH-CN_TOPIC_0000001456535180"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyFrom(const faiss::IndexFlat *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexFlat基于一个已有的Index拷贝到Ascend，清空当前的AscendIndexFlat底库，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b166777494441"><a name="b166777494441"></a><a name="b166777494441"></a>const faiss::IndexFlat *index</strong>：CPU侧index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname16174215239"><a name="parmname16174215239"></a><a name="parmname16174215239"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>

<a name="table525914213409"></a>
<table><tbody><tr id="row16259174214406"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1025994215407"><a name="p1025994215407"></a><a name="p1025994215407"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p6259442124016"><a name="p6259442124016"></a><a name="p6259442124016"></a>void copyFrom(const faiss::IndexIDMap*index);</p>
</td>
</tr>
<tr id="row1925914423401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1025984212403"><a name="p1025984212403"></a><a name="p1025984212403"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p92592422406"><a name="p92592422406"></a><a name="p92592422406"></a>AscendIndexFlat基于一个已有的<span class="parmname" id="parmname063811221710"><a name="parmname063811221710"></a><a name="parmname063811221710"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexFlat底库，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row5259842124019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7259174224012"><a name="p7259174224012"></a><a name="p7259174224012"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p925984284010"><a name="p925984284010"></a><a name="p925984284010"></a><strong id="b95691227174516"><a name="b95691227174516"></a><a name="b95691227174516"></a>const faiss::IndexIDMap*index</strong>：CPU侧index资源。</p>
</td>
</tr>
<tr id="row14259042124016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1125954204015"><a name="p1125954204015"></a><a name="p1125954204015"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14260174234011"><a name="p14260174234011"></a><a name="p14260174234011"></a>无</p>
</td>
</tr>
<tr id="row20260174294018"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12260842114015"><a name="p12260842114015"></a><a name="p12260842114015"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p15260242114012"><a name="p15260242114012"></a><a name="p15260242114012"></a>无</p>
</td>
</tr>
<tr id="row1626015428401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p152607429407"><a name="p152607429407"></a><a name="p152607429407"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1726012424402"><a name="p1726012424402"></a><a name="p1726012424402"></a>index需要为合法有效的IndexIDMap指针，否则可能造成程序崩溃或功能不可用，该Index的成员索引维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
</td>
</tr>
</tbody>
</table>


#### copyTo接口<a name="ZH-CN_TOPIC_0000001456535148"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexFlat *index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>将AscendIndexFlat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b194718510463"><a name="b194718510463"></a><a name="b194718510463"></a>faiss::IndexFlat *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname773416518233"><a name="parmname773416518233"></a><a name="parmname773416518233"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

<a name="table154531752144016"></a>
<table><tbody><tr id="row12453652124015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1245375234010"><a name="p1245375234010"></a><a name="p1245375234010"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p194531452144019"><a name="p194531452144019"></a><a name="p194531452144019"></a>void copyTo(faiss::IndexIDMap*index) const;</p>
</td>
</tr>
<tr id="row74535524403"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7453752174010"><a name="p7453752174010"></a><a name="p7453752174010"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p045325210409"><a name="p045325210409"></a><a name="p045325210409"></a>将AscendIndexFlat的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row11453135211406"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p94531252184014"><a name="p94531252184014"></a><a name="p94531252184014"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16453175224020"><a name="p16453175224020"></a><a name="p16453175224020"></a><strong id="b172171419114612"><a name="b172171419114612"></a><a name="b172171419114612"></a>faiss::IndexIDMap*index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="row345495215407"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p8454135218406"><a name="p8454135218406"></a><a name="p8454135218406"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p74541752174010"><a name="p74541752174010"></a><a name="p74541752174010"></a>无</p>
</td>
</tr>
<tr id="row19454852184017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3454752144017"><a name="p3454752144017"></a><a name="p3454752144017"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p145465264013"><a name="p145465264013"></a><a name="p145465264013"></a>无</p>
</td>
</tr>
<tr id="row845415211403"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p14541452144011"><a name="p14541452144011"></a><a name="p14541452144011"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p24541452194012"><a name="p24541452194012"></a><a name="p24541452194012"></a><span class="parmname" id="parmname13154121312411"><a name="parmname13154121312411"></a><a name="parmname13154121312411"></a>“index”</span>需要为合法有效的IndexIDMap指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>


#### getBase接口<a name="ZH-CN_TOPIC_0000001456375236"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getBase(int deviceId, char* xb) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexFlat在特定<span class="parmname" id="parmname156459015472"><a name="parmname156459015472"></a><a name="parmname156459015472"></a>“deviceId”</span>上管理的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b10126123144712"><a name="b10126123144712"></a><a name="b10126123144712"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b1957812517475"><a name="b1957812517475"></a><a name="b1957812517475"></a>char* xb</strong>：AscendIndexFlat在<span class="parmname" id="parmname12584191319472"><a name="parmname12584191319472"></a><a name="parmname12584191319472"></a>“deviceId”</span>上存储的底库特征向量。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname645613115475"><a name="parmname645613115475"></a><a name="parmname645613115475"></a>“deviceId”</span>需要为合法值的设备ID。</p>
<p id="p835381215917"><a name="p835381215917"></a><a name="p835381215917"></a><span class="parmname" id="parmname14391592819"><a name="parmname14391592819"></a><a name="parmname14391592819"></a>“xb”</span>需要为非空指针，且长度应该为dims * BaseSize * sizeof(float32)字节，否则可能出现越界读写错误并引起程序崩溃，其中BaseSize为getBaseSize函数的返回值。</p>
</td>
</tr>
</tbody>
</table>


#### getBaseSize接口<a name="ZH-CN_TOPIC_0000001456854956"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>size_t getBaseSize(int deviceId) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexFlat在特定<span class="parmname" id="parmname5505194594611"><a name="parmname5505194594611"></a><a name="parmname5505194594611"></a>“deviceId”</span>上管理的特征向量数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b152596352465"><a name="b152596352465"></a><a name="b152596352465"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>在特定<span class="parmname" id="parmname362764114465"><a name="parmname362764114465"></a><a name="parmname362764114465"></a>“deviceId”</span>上的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname13287173918465"><a name="parmname13287173918465"></a><a name="parmname13287173918465"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>


#### getIdxMap接口<a name="ZH-CN_TOPIC_0000001506334785"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexFlat在特定<span class="parmname" id="parmname141894416475"><a name="parmname141894416475"></a><a name="parmname141894416475"></a>“deviceId”</span>上管理的特征向量ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b18839174713474"><a name="b18839174713474"></a><a name="b18839174713474"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b2587150104719"><a name="b2587150104719"></a><a name="b2587150104719"></a>std::vector&lt;idx_t&gt; &amp;idxMap</strong>：AscendIndexFlat在<span class="parmname" id="parmname27675634719"><a name="parmname27675634719"></a><a name="parmname27675634719"></a>“deviceId”</span>上存储的底库特征向量ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname886865894720"><a name="parmname886865894720"></a><a name="parmname886865894720"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000001506495701"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexFlat&amp; operator=(const AscendIndexFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b1210811552436"><a name="b1210811552436"></a><a name="b1210811552436"></a>const AscendIndexFlat&amp;</strong>：常量AscendIndexFlat。</p>
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


#### search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001810529650"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11977033135012"><a name="p11977033135012"></a><a name="p11977033135012"></a>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7584153011582"><a name="p7584153011582"></a><a name="p7584153011582"></a>AscendIndexFlat的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为<strong id="b177781744812"><a name="b177781744812"></a><a name="b177781744812"></a>0</strong>、<strong id="b61111618184814"><a name="b61111618184814"></a><a name="b61111618184814"></a>1</strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong id="b18384163024810"><a name="b18384163024810"></a><a name="b18384163024810"></a>1</strong>参与，<strong id="b2070119316483"><a name="b2070119316483"></a><a name="b2070119316483"></a>0</strong>不参与。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b8509184711813"><a name="b8509184711813"></a><a name="b8509184711813"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b2086114494814"><a name="b2086114494814"></a><a name="b2086114494814"></a>const float *x</strong>：特征向量数据。</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b2484155112813"><a name="b2484155112813"></a><a name="b2484155112813"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p841235065815"><a name="p841235065815"></a><a name="p841235065815"></a><strong id="b77116531187"><a name="b77116531187"></a><a name="b77116531187"></a>const void *mask</strong>：特征底库掩码。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b676467798"><a name="b676467798"></a><a name="b676467798"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname1152618119407"><a name="parmname1152618119407"></a><a name="parmname1152618119407"></a>“k”</span>个向量间的距离值。</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b5824159193"><a name="b5824159193"></a><a name="b5824159193"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname145221464014"><a name="parmname145221464014"></a><a name="parmname145221464014"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul184581435495"></a><a name="ul184581435495"></a><ul id="ul184581435495"><li>此处<span class="parmname" id="parmname11945191043317"><a name="parmname11945191043317"></a><a name="parmname11945191043317"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处<span class="parmname" id="parmname118167143338"><a name="parmname118167143338"></a><a name="parmname118167143338"></a>“k”</span>通常不允许超过4096。</li><li>此处指针<span class="parmname" id="parmname2847104218106"><a name="parmname2847104218106"></a><a name="parmname2847104218106"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname142691946111020"><a name="parmname142691946111020"></a><a name="parmname142691946111020"></a>“distances”</span>/<span class="parmname" id="parmname2586184851015"><a name="parmname2586184851015"></a><a name="parmname2586184851015"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b101511878307"><a name="b101511878307"></a><a name="b101511878307"></a>k</strong> * <strong id="b717661318306"><a name="b717661318306"></a><a name="b717661318306"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname199479521105"><a name="parmname199479521105"></a><a name="parmname199479521105"></a>“mask”</span>需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table0628133121511"></a>
<table><tbody><tr id="row5682739155"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.1.1"><p id="p156823331511"><a name="p156823331511"></a><a name="p156823331511"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.1.1 "><p id="p1868293131514"><a name="p1868293131514"></a><a name="p1868293131514"></a>void search_with_masks(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row1368233181518"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.2.1"><p id="p36821633157"><a name="p36821633157"></a><a name="p36821633157"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.2.1 "><p id="p8682432157"><a name="p8682432157"></a><a name="p8682432157"></a>AscendIndexFlat的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为0、1比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，1参与，0不参与。</p>
</td>
</tr>
<tr id="row196837312153"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.3.1"><p id="p10683233157"><a name="p10683233157"></a><a name="p10683233157"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.3.1 "><p id="p1768318381520"><a name="p1768318381520"></a><a name="p1768318381520"></a><strong id="b11627624191518"><a name="b11627624191518"></a><a name="b11627624191518"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p2683123121519"><a name="p2683123121519"></a><a name="p2683123121519"></a><strong id="b151811427191518"><a name="b151811427191518"></a><a name="b151811427191518"></a>const uint16_t *x</strong>：特征向量数据。</p>
<p id="p146831137157"><a name="p146831137157"></a><a name="p146831137157"></a><strong id="b16567292155"><a name="b16567292155"></a><a name="b16567292155"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p768320371520"><a name="p768320371520"></a><a name="p768320371520"></a><strong id="b136051632191510"><a name="b136051632191510"></a><a name="b136051632191510"></a>const void *mask</strong>：特征底库掩码。</p>
</td>
</tr>
<tr id="row1868310301516"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.4.1"><p id="p16683193111511"><a name="p16683193111511"></a><a name="p16683193111511"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.4.1 "><p id="p1168315312152"><a name="p1168315312152"></a><a name="p1168315312152"></a><strong id="b1930365216156"><a name="b1930365216156"></a><a name="b1930365216156"></a>float *distances</strong>：查询向量与距离最近的前“k”个向量间的距离值。</p>
<p id="p10683133158"><a name="p10683133158"></a><a name="p10683133158"></a><strong id="b952215610159"><a name="b952215610159"></a><a name="b952215610159"></a>idx_t *labels</strong>：查询的距离最近的前“k”个向量的ID。</p>
</td>
</tr>
<tr id="row1668310317152"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.5.1"><p id="p19683203141517"><a name="p19683203141517"></a><a name="p19683203141517"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.5.1 "><p id="p468393131518"><a name="p468393131518"></a><a name="p468393131518"></a>无</p>
</td>
</tr>
<tr id="row1768312318157"><th class="firstcol" valign="top" width="13.56%" id="mcps1.1.3.6.1"><p id="p1368343181514"><a name="p1368343181514"></a><a name="p1368343181514"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.1.3.6.1 "><a name="ul16154204141611"></a><a name="ul16154204141611"></a><ul id="ul16154204141611"><li>此处“n”的取值范围：0 &lt; n &lt; 1e9。</li><li>此处“k”通常不允许超过4096。</li><li>此处指针“x”需要为非空指针，且长度应该为dim * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针“distances”/“labels”需要为非空指针，且长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针“mask”需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>



### AscendIndexFlatConfig<a name="ZH-CN_TOPIC_0000001456375216"></a>

AscendIndexFlat需要使用对应的AscendIndexFlatConfig执行对应资源的初始化。

**接口说明<a name="section140920164419"></a>**

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>inline AscendIndexFlatConfig()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexFlatConfig的默认构造函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="18.15%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="81.85%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table46951722104415"></a>
<table><tbody><tr id="row186961822204410"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1069662218442"><a name="p1069662218442"></a><a name="p1069662218442"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1792514755010"><a name="p1792514755010"></a><a name="p1792514755010"></a>inline AscendIndexFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row169692210443"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p969622274410"><a name="p969622274410"></a><a name="p969622274410"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7696172274411"><a name="p7696172274411"></a><a name="p7696172274411"></a>AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据<span class="parmname" id="parmname468135223212"><a name="parmname468135223212"></a><a name="parmname468135223212"></a>“devices”</span>中配置的值设置Device侧<span id="ph10238132515618"><a name="ph10238132515618"></a><a name="ph10238132515618"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row136963220449"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p146961622204418"><a name="p146961622204418"></a><a name="p146961622204418"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p5696122214448"><a name="p5696122214448"></a><a name="p5696122214448"></a><strong id="b1274764723010"><a name="b1274764723010"></a><a name="b1274764723010"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p17641112731910"><a name="p17641112731910"></a><a name="p17641112731910"></a><strong id="b13831175053011"><a name="b13831175053011"></a><a name="b13831175053011"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname1331442416116"><a name="parmname1331442416116"></a><a name="parmname1331442416116"></a>“FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。</p>
<a name="ul1067904281918"></a><a name="ul1067904281918"></a><ul id="ul1067904281918"><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue17434174411812"><a name="parmvalue17434174411812"></a><a name="parmvalue17434174411812"></a>“faiss::METRIC_L2”</span>时建议设置1024MB。</li><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue634861211200"><a name="parmvalue634861211200"></a><a name="parmvalue634861211200"></a>“faiss::METRIC_INNER_PRODUCT”</span>时建议设置1280MB。</li></ul>
</td>
</tr>
<tr id="row16696172214415"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1069610226440"><a name="p1069610226440"></a><a name="p1069610226440"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p669620222449"><a name="p669620222449"></a><a name="p669620222449"></a>无</p>
</td>
</tr>
<tr id="row16696192264412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3696922114411"><a name="p3696922114411"></a><a name="p3696922114411"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p56962225445"><a name="p56962225445"></a><a name="p56962225445"></a>无</p>
</td>
</tr>
<tr id="row169602211448"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1469617229447"><a name="p1469617229447"></a><a name="p1469617229447"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul44541139318"></a><a name="ul44541139318"></a><ul id="ul44541139318"><li><span class="parmname" id="parmname1793182919311"><a name="parmname1793182919311"></a><a name="parmname1793182919311"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname15293195861219"><a name="parmname15293195861219"></a><a name="parmname15293195861219"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue13120151692414"><a name="parmvalue13120151692414"></a><a name="parmvalue13120151692414"></a>“-1”</span>时，Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table842319354444"></a>
<table><tbody><tr id="row1142318355442"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1842393554413"><a name="p1842393554413"></a><a name="p1842393554413"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p593315225213"><a name="p593315225213"></a><a name="p593315225213"></a>inline AscendIndexFlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM)</p>
</td>
</tr>
<tr id="row1242323524413"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p84231035164418"><a name="p84231035164418"></a><a name="p84231035164418"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1142333517449"><a name="p1142333517449"></a><a name="p1142333517449"></a>AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据<span class="parmname" id="parmname2062863616353"><a name="parmname2062863616353"></a><a name="parmname2062863616353"></a>“devices”</span>中配置的值设置Device侧<span id="ph37713595611"><a name="ph37713595611"></a><a name="ph37713595611"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row94235350446"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p742383516443"><a name="p742383516443"></a><a name="p742383516443"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p14423173514449"><a name="p14423173514449"></a><a name="p14423173514449"></a><strong id="b10613041173517"><a name="b10613041173517"></a><a name="b10613041173517"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b58045433352"><a name="b58045433352"></a><a name="b58045433352"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的<span class="parmname" id="parmname13423133524413"><a name="parmname13423133524413"></a><a name="parmname13423133524413"></a>“FLAT_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。</p>
<a name="ul3423163517446"></a><a name="ul3423163517446"></a><ul id="ul3423163517446"><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue34231835124411"><a name="parmvalue34231835124411"></a><a name="parmvalue34231835124411"></a>“faiss::METRIC_L2”</span>时建议设置1024MB。</li><li>当AscendIndexFlat的距离类型为<span class="parmvalue" id="parmvalue124230354440"><a name="parmvalue124230354440"></a><a name="parmvalue124230354440"></a>“faiss::METRIC_INNER_PRODUCT”</span>时建议设置1280MB。</li></ul>
</td>
</tr>
<tr id="row1842343514447"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p642343513448"><a name="p642343513448"></a><a name="p642343513448"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p4424183564416"><a name="p4424183564416"></a><a name="p4424183564416"></a>无</p>
</td>
</tr>
<tr id="row11424135174412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p24244358441"><a name="p24244358441"></a><a name="p24244358441"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16424135124411"><a name="p16424135124411"></a><a name="p16424135124411"></a>无</p>
</td>
</tr>
<tr id="row14424183594412"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p17424133524410"><a name="p17424133524410"></a><a name="p17424133524410"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1811714915354"></a><a name="ul1811714915354"></a><ul id="ul1811714915354"><li><span class="parmname" id="parmname06427236368"><a name="parmname06427236368"></a><a name="parmname06427236368"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname442411352446"><a name="parmname442411352446"></a><a name="parmname442411352446"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue31828359242"><a name="parmvalue31828359242"></a><a name="parmvalue31828359242"></a>“-1”</span>时，Device侧<span id="ph1942410353449"><a name="ph1942410353449"></a><a name="ph1942410353449"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li></ul>
</td>
</tr>
</tbody>
</table>


### AscendIndexFlatL2<a name="ZH-CN_TOPIC_0000001456375424"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001877955534"></a>

AscendIndexFlatL2是存储FP16浮点数类型并使用L2距离的特征暴力检索算法。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

>![](public_sys-resources/icon-note.gif) **说明：** 
>AscendIndexFlatL2算法支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，使用在线算子需要用户在应用程序的最后显式调用 \(void\)aclFinalize\(\) （需要包含头文件：\#include "acl/acl.h"）。


#### AscendIndexFlatL2接口<a name="ZH-CN_TOPIC_0000001506495761"></a>

<a name="zh-cn_topic_0000001294312541_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294312541_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294312541_p12559123810"><a name="zh-cn_topic_0000001294312541_p12559123810"></a><a name="zh-cn_topic_0000001294312541_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294312541_p2425655144613"><a name="zh-cn_topic_0000001294312541_p2425655144613"></a><a name="zh-cn_topic_0000001294312541_p2425655144613"></a>AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294312541_p1212599383"><a name="zh-cn_topic_0000001294312541_p1212599383"></a><a name="zh-cn_topic_0000001294312541_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294312541_p131714208358"><a name="zh-cn_topic_0000001294312541_p131714208358"></a><a name="zh-cn_topic_0000001294312541_p131714208358"></a>AscendIndexFlatL2的构造函数，基于一个已有的<span class="parmname" id="zh-cn_topic_0000001294312541_parmname69451751507"><a name="zh-cn_topic_0000001294312541_parmname69451751507"></a><a name="zh-cn_topic_0000001294312541_parmname69451751507"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294312541_p112195910383"><a name="zh-cn_topic_0000001294312541_p112195910383"></a><a name="zh-cn_topic_0000001294312541_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294312541_p874812810555"><a name="zh-cn_topic_0000001294312541_p874812810555"></a><a name="zh-cn_topic_0000001294312541_p874812810555"></a><strong id="zh-cn_topic_0000001294312541_b2688145217499"><a name="zh-cn_topic_0000001294312541_b2688145217499"></a><a name="zh-cn_topic_0000001294312541_b2688145217499"></a>faiss::IndexFlatL2 *index</strong>：CPU侧Index资源。</p>
<p id="zh-cn_topic_0000001294312541_p661314244382"><a name="zh-cn_topic_0000001294312541_p661314244382"></a><a name="zh-cn_topic_0000001294312541_p661314244382"></a><strong id="zh-cn_topic_0000001294312541_b278625404911"><a name="zh-cn_topic_0000001294312541_b278625404911"></a><a name="zh-cn_topic_0000001294312541_b278625404911"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294312541_p17235973820"><a name="zh-cn_topic_0000001294312541_p17235973820"></a><a name="zh-cn_topic_0000001294312541_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294312541_p973225082318"><a name="zh-cn_topic_0000001294312541_p973225082318"></a><a name="zh-cn_topic_0000001294312541_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294312541_p182459113812"><a name="zh-cn_topic_0000001294312541_p182459113812"></a><a name="zh-cn_topic_0000001294312541_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294312541_p132314362521"><a name="zh-cn_topic_0000001294312541_p132314362521"></a><a name="zh-cn_topic_0000001294312541_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312541_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294312541_p423590386"><a name="zh-cn_topic_0000001294312541_p423590386"></a><a name="zh-cn_topic_0000001294312541_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294312541_p182559163813"><a name="zh-cn_topic_0000001294312541_p182559163813"></a><a name="zh-cn_topic_0000001294312541_p182559163813"></a><span class="parmname" id="zh-cn_topic_0000001294312541_parmname6385211185011"><a name="zh-cn_topic_0000001294312541_parmname6385211185011"></a><a name="zh-cn_topic_0000001294312541_parmname6385211185011"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为faiss::MetricType::METRIC_L2。</p>
</td>
</tr>
</tbody>
</table>

<a name="zh-cn_topic_0000001294591937_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294591937_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294591937_p12559123810"><a name="zh-cn_topic_0000001294591937_p12559123810"></a><a name="zh-cn_topic_0000001294591937_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294591937_p144102184422"><a name="zh-cn_topic_0000001294591937_p144102184422"></a><a name="zh-cn_topic_0000001294591937_p144102184422"></a>AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294591937_p1212599383"><a name="zh-cn_topic_0000001294591937_p1212599383"></a><a name="zh-cn_topic_0000001294591937_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294591937_p94401440428"><a name="zh-cn_topic_0000001294591937_p94401440428"></a><a name="zh-cn_topic_0000001294591937_p94401440428"></a>AscendIndexFlatL2的构造函数，生成维度为dims的AscendIndexFlatL2（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="zh-cn_topic_0000001294591937_parmname18694103215115"><a name="zh-cn_topic_0000001294591937_parmname18694103215115"></a><a name="zh-cn_topic_0000001294591937_parmname18694103215115"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294591937_p112195910383"><a name="zh-cn_topic_0000001294591937_p112195910383"></a><a name="zh-cn_topic_0000001294591937_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294591937_p874812810555"><a name="zh-cn_topic_0000001294591937_p874812810555"></a><a name="zh-cn_topic_0000001294591937_p874812810555"></a><strong id="zh-cn_topic_0000001294591937_b8667143775117"><a name="zh-cn_topic_0000001294591937_b8667143775117"></a><a name="zh-cn_topic_0000001294591937_b8667143775117"></a>int dims</strong>：AscendIndexFlatL2管理的一组特征向量的维度。</p>
<p id="zh-cn_topic_0000001294591937_p1220621175115"><a name="zh-cn_topic_0000001294591937_p1220621175115"></a><a name="zh-cn_topic_0000001294591937_p1220621175115"></a><strong id="zh-cn_topic_0000001294591937_b6244340115115"><a name="zh-cn_topic_0000001294591937_b6244340115115"></a><a name="zh-cn_topic_0000001294591937_b6244340115115"></a>AscendIndexFlatConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294591937_p17235973820"><a name="zh-cn_topic_0000001294591937_p17235973820"></a><a name="zh-cn_topic_0000001294591937_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294591937_p973225082318"><a name="zh-cn_topic_0000001294591937_p973225082318"></a><a name="zh-cn_topic_0000001294591937_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294591937_p182459113812"><a name="zh-cn_topic_0000001294591937_p182459113812"></a><a name="zh-cn_topic_0000001294591937_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294591937_p132314362521"><a name="zh-cn_topic_0000001294591937_p132314362521"></a><a name="zh-cn_topic_0000001294591937_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294591937_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294591937_p423590386"><a name="zh-cn_topic_0000001294591937_p423590386"></a><a name="zh-cn_topic_0000001294591937_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294591937_p1229447954"><a name="zh-cn_topic_0000001294591937_p1229447954"></a><a name="zh-cn_topic_0000001294591937_p1229447954"></a>dims ∈ {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 4096, 3584}</p>
</td>
</tr>
</tbody>
</table>

<a name="zh-cn_topic_0000001247793230_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001247793230_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001247793230_p12559123810"><a name="zh-cn_topic_0000001247793230_p12559123810"></a><a name="zh-cn_topic_0000001247793230_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001247793230_p7112274471"><a name="zh-cn_topic_0000001247793230_p7112274471"></a><a name="zh-cn_topic_0000001247793230_p7112274471"></a>AscendIndexFlatL2(const AscendIndexFlatL2&amp;) = delete;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001247793230_p1212599383"><a name="zh-cn_topic_0000001247793230_p1212599383"></a><a name="zh-cn_topic_0000001247793230_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001247793230_p131714208358"><a name="zh-cn_topic_0000001247793230_p131714208358"></a><a name="zh-cn_topic_0000001247793230_p131714208358"></a>声明此Index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001247793230_p112195910383"><a name="zh-cn_topic_0000001247793230_p112195910383"></a><a name="zh-cn_topic_0000001247793230_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001247793230_p867213174418"><a name="zh-cn_topic_0000001247793230_p867213174418"></a><a name="zh-cn_topic_0000001247793230_p867213174418"></a><strong id="zh-cn_topic_0000001247793230_b322283735213"><a name="zh-cn_topic_0000001247793230_b322283735213"></a><a name="zh-cn_topic_0000001247793230_b322283735213"></a>const AscendIndexFlatL2&amp;</strong>：常量AscendIndexFlatL2。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001247793230_p17235973820"><a name="zh-cn_topic_0000001247793230_p17235973820"></a><a name="zh-cn_topic_0000001247793230_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001247793230_p973225082318"><a name="zh-cn_topic_0000001247793230_p973225082318"></a><a name="zh-cn_topic_0000001247793230_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001247793230_p182459113812"><a name="zh-cn_topic_0000001247793230_p182459113812"></a><a name="zh-cn_topic_0000001247793230_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001247793230_p132314362521"><a name="zh-cn_topic_0000001247793230_p132314362521"></a><a name="zh-cn_topic_0000001247793230_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793230_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001247793230_p423590386"><a name="zh-cn_topic_0000001247793230_p423590386"></a><a name="zh-cn_topic_0000001247793230_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001247793230_p182559163813"><a name="zh-cn_topic_0000001247793230_p182559163813"></a><a name="zh-cn_topic_0000001247793230_p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="zh-cn_topic_0000001294312453_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294312453_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294312453_p12559123810"><a name="zh-cn_topic_0000001294312453_p12559123810"></a><a name="zh-cn_topic_0000001294312453_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294312453_p132681218211"><a name="zh-cn_topic_0000001294312453_p132681218211"></a><a name="zh-cn_topic_0000001294312453_p132681218211"></a>virtual ~AscendIndexFlatL2()</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294312453_p1212599383"><a name="zh-cn_topic_0000001294312453_p1212599383"></a><a name="zh-cn_topic_0000001294312453_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294312453_p131714208358"><a name="zh-cn_topic_0000001294312453_p131714208358"></a><a name="zh-cn_topic_0000001294312453_p131714208358"></a>AscendIndexFlatL2的析构函数，销毁AscendIndexFlatL2对象，释放资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294312453_p112195910383"><a name="zh-cn_topic_0000001294312453_p112195910383"></a><a name="zh-cn_topic_0000001294312453_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294312453_p8451184515218"><a name="zh-cn_topic_0000001294312453_p8451184515218"></a><a name="zh-cn_topic_0000001294312453_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294312453_p17235973820"><a name="zh-cn_topic_0000001294312453_p17235973820"></a><a name="zh-cn_topic_0000001294312453_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294312453_p973225082318"><a name="zh-cn_topic_0000001294312453_p973225082318"></a><a name="zh-cn_topic_0000001294312453_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294312453_p182459113812"><a name="zh-cn_topic_0000001294312453_p182459113812"></a><a name="zh-cn_topic_0000001294312453_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294312453_p132314362521"><a name="zh-cn_topic_0000001294312453_p132314362521"></a><a name="zh-cn_topic_0000001294312453_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294312453_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294312453_p423590386"><a name="zh-cn_topic_0000001294312453_p423590386"></a><a name="zh-cn_topic_0000001294312453_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294312453_p182559163813"><a name="zh-cn_topic_0000001294312453_p182559163813"></a><a name="zh-cn_topic_0000001294312453_p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### copyFrom接口<a name="ZH-CN_TOPIC_0000001456375400"></a>

<a name="zh-cn_topic_0000001248112146_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001248112146_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001248112146_p12559123810"><a name="zh-cn_topic_0000001248112146_p12559123810"></a><a name="zh-cn_topic_0000001248112146_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001248112146_p1531315343445"><a name="zh-cn_topic_0000001248112146_p1531315343445"></a><a name="zh-cn_topic_0000001248112146_p1531315343445"></a>void copyFrom(faiss::IndexFlat *index);</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001248112146_p1212599383"><a name="zh-cn_topic_0000001248112146_p1212599383"></a><a name="zh-cn_topic_0000001248112146_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001248112146_p131714208358"><a name="zh-cn_topic_0000001248112146_p131714208358"></a><a name="zh-cn_topic_0000001248112146_p131714208358"></a>AscendIndexFlat基于一个已有的<span class="parmname" id="parmname1804125953520"><a name="parmname1804125953520"></a><a name="parmname1804125953520"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexFlatL2底库，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001248112146_p112195910383"><a name="zh-cn_topic_0000001248112146_p112195910383"></a><a name="zh-cn_topic_0000001248112146_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001248112146_p874812810555"><a name="zh-cn_topic_0000001248112146_p874812810555"></a><a name="zh-cn_topic_0000001248112146_p874812810555"></a><strong id="zh-cn_topic_0000001248112146_b976174655318"><a name="zh-cn_topic_0000001248112146_b976174655318"></a><a name="zh-cn_topic_0000001248112146_b976174655318"></a>const faiss::IndexFlat *index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001248112146_p17235973820"><a name="zh-cn_topic_0000001248112146_p17235973820"></a><a name="zh-cn_topic_0000001248112146_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001248112146_p973225082318"><a name="zh-cn_topic_0000001248112146_p973225082318"></a><a name="zh-cn_topic_0000001248112146_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001248112146_p182459113812"><a name="zh-cn_topic_0000001248112146_p182459113812"></a><a name="zh-cn_topic_0000001248112146_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001248112146_p132314362521"><a name="zh-cn_topic_0000001248112146_p132314362521"></a><a name="zh-cn_topic_0000001248112146_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001248112146_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001248112146_p423590386"><a name="zh-cn_topic_0000001248112146_p423590386"></a><a name="zh-cn_topic_0000001248112146_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001248112146_p182559163813"><a name="zh-cn_topic_0000001248112146_p182559163813"></a><a name="zh-cn_topic_0000001248112146_p182559163813"></a><span class="parmname" id="zh-cn_topic_0000001248112146_parmname159121156135315"><a name="zh-cn_topic_0000001248112146_parmname159121156135315"></a><a name="zh-cn_topic_0000001248112146_parmname159121156135315"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3584}，底库向量总数的取值范围：0 &lt;= n &lt; 1e9，metric_type参数取值为faiss::MetricType::METRIC_L2。</p>
</td>
</tr>
</tbody>
</table>


#### copyTo接口<a name="ZH-CN_TOPIC_0000001456535052"></a>

<a name="zh-cn_topic_0000001247793178_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001247793178_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001247793178_p12559123810"><a name="zh-cn_topic_0000001247793178_p12559123810"></a><a name="zh-cn_topic_0000001247793178_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001247793178_p10713954155218"><a name="zh-cn_topic_0000001247793178_p10713954155218"></a><a name="zh-cn_topic_0000001247793178_p10713954155218"></a>void copyTo(faiss::IndexFlat *index);</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001247793178_p1212599383"><a name="zh-cn_topic_0000001247793178_p1212599383"></a><a name="zh-cn_topic_0000001247793178_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001247793178_p131714208358"><a name="zh-cn_topic_0000001247793178_p131714208358"></a><a name="zh-cn_topic_0000001247793178_p131714208358"></a>将AscendIndexFlatL2的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001247793178_p112195910383"><a name="zh-cn_topic_0000001247793178_p112195910383"></a><a name="zh-cn_topic_0000001247793178_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001247793178_p874812810555"><a name="zh-cn_topic_0000001247793178_p874812810555"></a><a name="zh-cn_topic_0000001247793178_p874812810555"></a><strong id="zh-cn_topic_0000001247793178_b2644689548"><a name="zh-cn_topic_0000001247793178_b2644689548"></a><a name="zh-cn_topic_0000001247793178_b2644689548"></a>faiss::IndexFlat *index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001247793178_p17235973820"><a name="zh-cn_topic_0000001247793178_p17235973820"></a><a name="zh-cn_topic_0000001247793178_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001247793178_p973225082318"><a name="zh-cn_topic_0000001247793178_p973225082318"></a><a name="zh-cn_topic_0000001247793178_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001247793178_p182459113812"><a name="zh-cn_topic_0000001247793178_p182459113812"></a><a name="zh-cn_topic_0000001247793178_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001247793178_p132314362521"><a name="zh-cn_topic_0000001247793178_p132314362521"></a><a name="zh-cn_topic_0000001247793178_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001247793178_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001247793178_p423590386"><a name="zh-cn_topic_0000001247793178_p423590386"></a><a name="zh-cn_topic_0000001247793178_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001247793178_p182559163813"><a name="zh-cn_topic_0000001247793178_p182559163813"></a><a name="zh-cn_topic_0000001247793178_p182559163813"></a><span class="parmname" id="zh-cn_topic_0000001247793178_parmname683216143547"><a name="zh-cn_topic_0000001247793178_parmname683216143547"></a><a name="zh-cn_topic_0000001247793178_parmname683216143547"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000001456695116"></a>

<a name="zh-cn_topic_0000001294432513_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001294432513_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001294432513_p12559123810"><a name="zh-cn_topic_0000001294432513_p12559123810"></a><a name="zh-cn_topic_0000001294432513_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001294432513_p1213215268503"><a name="zh-cn_topic_0000001294432513_p1213215268503"></a><a name="zh-cn_topic_0000001294432513_p1213215268503"></a>AscendIndexFlatL2&amp; operator=(const AscendIndexFlatL2&amp;) = delete;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001294432513_p1212599383"><a name="zh-cn_topic_0000001294432513_p1212599383"></a><a name="zh-cn_topic_0000001294432513_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001294432513_p131714208358"><a name="zh-cn_topic_0000001294432513_p131714208358"></a><a name="zh-cn_topic_0000001294432513_p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001294432513_p112195910383"><a name="zh-cn_topic_0000001294432513_p112195910383"></a><a name="zh-cn_topic_0000001294432513_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001294432513_p867213174418"><a name="zh-cn_topic_0000001294432513_p867213174418"></a><a name="zh-cn_topic_0000001294432513_p867213174418"></a><strong id="zh-cn_topic_0000001294432513_b12571191511538"><a name="zh-cn_topic_0000001294432513_b12571191511538"></a><a name="zh-cn_topic_0000001294432513_b12571191511538"></a>const AscendIndexFlatL2&amp;</strong>：常量AscendIndexFlatL2。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001294432513_p17235973820"><a name="zh-cn_topic_0000001294432513_p17235973820"></a><a name="zh-cn_topic_0000001294432513_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001294432513_p973225082318"><a name="zh-cn_topic_0000001294432513_p973225082318"></a><a name="zh-cn_topic_0000001294432513_p973225082318"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001294432513_p182459113812"><a name="zh-cn_topic_0000001294432513_p182459113812"></a><a name="zh-cn_topic_0000001294432513_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001294432513_p132314362521"><a name="zh-cn_topic_0000001294432513_p132314362521"></a><a name="zh-cn_topic_0000001294432513_p132314362521"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001294432513_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001294432513_p423590386"><a name="zh-cn_topic_0000001294432513_p423590386"></a><a name="zh-cn_topic_0000001294432513_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001294432513_p182559163813"><a name="zh-cn_topic_0000001294432513_p182559163813"></a><a name="zh-cn_topic_0000001294432513_p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>



### AscendIndexSQ<a name="ZH-CN_TOPIC_0000001506614969"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456695120"></a>

AscendIndexSQ对输入向量执行Scalar Quantization。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。


#### AscendIndexSQ接口<a name="ZH-CN_TOPIC_0000001506614933"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a>AscendIndexSQ(const faiss::IndexScalarQuantizer* index, AscendIndexSQConfig config = AscendIndexSQConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexSQ的构造函数，基于一个已有的<span class="parmname" id="parmname8477263296"><a name="parmname8477263296"></a><a name="parmname8477263296"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b6104639815"><a name="b6104639815"></a><a name="b6104639815"></a>const faiss::IndexScalarQuantizer* index</strong>：CPU侧的Index资源。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1066616367111"><a name="b1066616367111"></a><a name="b1066616367111"></a>AscendIndexSQConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1938733313207"><a name="parmname1938733313207"></a><a name="parmname1938733313207"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}，sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

<a name="table207325212487"></a>
<table><tbody><tr id="row57316521481"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1731752204815"><a name="p1731752204815"></a><a name="p1731752204815"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2425655144613"><a name="p2425655144613"></a><a name="p2425655144613"></a>AscendIndexSQ(const faiss::IndexIDMap* index, AscendIndexSQConfig config = AscendIndexSQConfig());</p>
</td>
</tr>
<tr id="row1573165204811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p147395220488"><a name="p147395220488"></a><a name="p147395220488"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10738528483"><a name="p10738528483"></a><a name="p10738528483"></a>AscendIndexSQ的构造函数，基于一个已有的<span class="parmname" id="parmname853416312298"><a name="parmname853416312298"></a><a name="parmname853416312298"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row3731652104814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p197495254810"><a name="p197495254810"></a><a name="p197495254810"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b11364627323"><a name="b11364627323"></a><a name="b11364627323"></a>const faiss::IndexIDMap* index</strong>：CPU侧index资源。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b75252919212"><a name="b75252919212"></a><a name="b75252919212"></a>AscendIndexSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row37465224818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p147415211489"><a name="p147415211489"></a><a name="p147415211489"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8741452194814"><a name="p8741452194814"></a><a name="p8741452194814"></a>无</p>
</td>
</tr>
<tr id="row167475217487"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p474185234815"><a name="p474185234815"></a><a name="p474185234815"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p197455214820"><a name="p197455214820"></a><a name="p197455214820"></a>无</p>
</td>
</tr>
<tr id="row97455219484"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p974125219485"><a name="p974125219485"></a><a name="p974125219485"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p18741526488"><a name="p18741526488"></a><a name="p18741526488"></a><span class="parmname" id="parmname14521836192011"><a name="parmname14521836192011"></a><a name="parmname14521836192011"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的成员索引的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n ＜ 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}， sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1132217014918"></a>
<table><tbody><tr id="row1132250114917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1432220012499"><a name="p1432220012499"></a><a name="p1432220012499"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11207257504"><a name="p11207257504"></a><a name="p11207257504"></a>AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexSQConfig config = AscendIndexSQConfig());</p>
</td>
</tr>
<tr id="row1232215064915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p9322140114910"><a name="p9322140114910"></a><a name="p9322140114910"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p432219014915"><a name="p432219014915"></a><a name="p432219014915"></a>AscendIndexSQ的构造函数，生成维度为dims的AscendIndex（单个Index管理的一组向量的维度是唯一的），此时根据<span class="parmname" id="parmname1128518915434"><a name="parmname1128518915434"></a><a name="parmname1128518915434"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row23229044916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p732210174911"><a name="p732210174911"></a><a name="p732210174911"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16322607495"><a name="p16322607495"></a><a name="p16322607495"></a><strong id="b63116565211"><a name="b63116565211"></a><a name="b63116565211"></a>int dims</strong>：AscendIndexSQ管理的一组特征向量的维度。</p>
<p id="p995710373711"><a name="p995710373711"></a><a name="p995710373711"></a><strong id="b155310333495"><a name="b155310333495"></a><a name="b155310333495"></a>faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit</strong>，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
<p id="p163221204497"><a name="p163221204497"></a><a name="p163221204497"></a><strong id="b20474132614312"><a name="b20474132614312"></a><a name="b20474132614312"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p1132217074917"><a name="p1132217074917"></a><a name="p1132217074917"></a><strong id="b1687210287316"><a name="b1687210287316"></a><a name="b1687210287316"></a>AscendIndexSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row163222012498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1322110154919"><a name="p1322110154919"></a><a name="p1322110154919"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p73221102490"><a name="p73221102490"></a><a name="p73221102490"></a>无</p>
</td>
</tr>
<tr id="row6322190184913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1032260194919"><a name="p1032260194919"></a><a name="p1032260194919"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8322204495"><a name="p8322204495"></a><a name="p8322204495"></a>无</p>
</td>
</tr>
<tr id="row10322120124920"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p173221017492"><a name="p173221017492"></a><a name="p173221017492"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul193685551031"></a><a name="ul193685551031"></a><ul id="ul193685551031"><li>dims ∈ {64, 128, 256, 384, 512, 768}。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table16655810104919"></a>
<table><tbody><tr id="row19655810194912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16655710184912"><a name="p16655710184912"></a><a name="p16655710184912"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p8445440165114"><a name="p8445440165114"></a><a name="p8445440165114"></a>AscendIndexSQ(const AscendIndexSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row665561014492"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p10655161013492"><a name="p10655161013492"></a><a name="p10655161013492"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p365541024916"><a name="p365541024916"></a><a name="p365541024916"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row4655110114913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1365501024920"><a name="p1365501024920"></a><a name="p1365501024920"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b16480691410"><a name="b16480691410"></a><a name="b16480691410"></a>const AscendIndexSQ&amp;</strong>：AscendIndexSQ对象。</p>
</td>
</tr>
<tr id="row13655121044912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p15655111064920"><a name="p15655111064920"></a><a name="p15655111064920"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p146551210104911"><a name="p146551210104911"></a><a name="p146551210104911"></a>无</p>
</td>
</tr>
<tr id="row116554109498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1265512105496"><a name="p1265512105496"></a><a name="p1265512105496"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p46569101499"><a name="p46569101499"></a><a name="p46569101499"></a>无</p>
</td>
</tr>
<tr id="row16568103491"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p8656810104914"><a name="p8656810104914"></a><a name="p8656810104914"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p19656210134916"><a name="p19656210134916"></a><a name="p19656210134916"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table17704194534915"></a>
<table><tbody><tr id="row147041745174918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p170414524913"><a name="p170414524913"></a><a name="p170414524913"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexSQ();</p>
</td>
</tr>
<tr id="row370416455499"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p77042459498"><a name="p77042459498"></a><a name="p77042459498"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p97045456493"><a name="p97045456493"></a><a name="p97045456493"></a>AscendIndexSQ的析构函数，销毁AscendIndexSQ对象，释放资源。</p>
</td>
</tr>
<tr id="row470419456497"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p77041745124910"><a name="p77041745124910"></a><a name="p77041745124910"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row57042456497"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p470444514493"><a name="p470444514493"></a><a name="p470444514493"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p57041745194915"><a name="p57041745194915"></a><a name="p57041745194915"></a>无</p>
</td>
</tr>
<tr id="row4704845104910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p19704134513498"><a name="p19704134513498"></a><a name="p19704134513498"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p770454516493"><a name="p770454516493"></a><a name="p770454516493"></a>无</p>
</td>
</tr>
<tr id="row1870454504914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p77046459495"><a name="p77046459495"></a><a name="p77046459495"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p5704184513492"><a name="p5704184513492"></a><a name="p5704184513492"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### copyFrom接口<a name="ZH-CN_TOPIC_0000001506615037"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyFrom(const faiss::IndexScalarQuantizer* index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexSQ基于一个已有的<span class="parmname" id="parmname72491525329"><a name="parmname72491525329"></a><a name="parmname72491525329"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexSQ底库，并保持原有的AscendIndexSQ的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b113197161055"><a name="b113197161055"></a><a name="b113197161055"></a>const faiss::IndexScalarQuantizer* index</strong>：CPU侧index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1170518371159"><a name="parmname1170518371159"></a><a name="parmname1170518371159"></a>“index”</span>需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}，sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

<a name="table853716365015"></a>
<table><tbody><tr id="row1253763155012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p65375319502"><a name="p65375319502"></a><a name="p65375319502"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15371437503"><a name="p15371437503"></a><a name="p15371437503"></a>void copyFrom(const faiss::IndexIDMap* index);</p>
</td>
</tr>
<tr id="row95371733508"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p11537735509"><a name="p11537735509"></a><a name="p11537735509"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p153715311508"><a name="p153715311508"></a><a name="p153715311508"></a>AscendIndexSQ基于一个已有的<span class="parmname" id="parmname1570211587325"><a name="parmname1570211587325"></a><a name="parmname1570211587325"></a>“index”</span>拷贝到Ascend，清空当前的AscendIndexSQ底库，并保持原有的AscendIndexSQ的Device侧资源配置。</p>
</td>
</tr>
<tr id="row155371130507"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1153720316503"><a name="p1153720316503"></a><a name="p1153720316503"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1553715335013"><a name="p1553715335013"></a><a name="p1553715335013"></a><strong id="b1898128253"><a name="b1898128253"></a><a name="b1898128253"></a>const faiss::IndexIDMap*index</strong>：CPU侧index资源。</p>
</td>
</tr>
<tr id="row1253716318502"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p115377325010"><a name="p115377325010"></a><a name="p115377325010"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p153712355015"><a name="p153712355015"></a><a name="p153712355015"></a>无</p>
</td>
</tr>
<tr id="row9537203125019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p553711365019"><a name="p553711365019"></a><a name="p553711365019"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p553743135020"><a name="p553743135020"></a><a name="p553743135020"></a>无</p>
</td>
</tr>
<tr id="row55373320504"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1537143115010"><a name="p1537143115010"></a><a name="p1537143115010"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p12537193155012"><a name="p12537193155012"></a><a name="p12537193155012"></a><span class="parmname" id="parmname1290641255"><a name="parmname1290641255"></a><a name="parmname1290641255"></a>“index”</span>需要为合法有效的IndexIDMap指针，index的成员索引的维度d参数取值范围为{64, 128, 256, 384, 512, 768}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}，sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>


#### copyTo接口<a name="ZH-CN_TOPIC_0000001456695084"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1019716501395"><a name="p1019716501395"></a><a name="p1019716501395"></a>void copyTo(faiss::IndexScalarQuantizer* index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>将AscendIndexSQ的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b217675214518"><a name="b217675214518"></a><a name="b217675214518"></a>faiss::IndexScalarQuantizer* index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname74101355456"><a name="parmname74101355456"></a><a name="parmname74101355456"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

<a name="table817201512500"></a>
<table><tbody><tr id="row1517171595016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9171015145014"><a name="p9171015145014"></a><a name="p9171015145014"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexIDMap* index) const;</p>
</td>
</tr>
<tr id="row5171115145019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3177155503"><a name="p3177155503"></a><a name="p3177155503"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p51761519504"><a name="p51761519504"></a><a name="p51761519504"></a>将AscendIndexSQ的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row101711535017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p717151512506"><a name="p717151512506"></a><a name="p717151512506"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12170155508"><a name="p12170155508"></a><a name="p12170155508"></a><strong id="b18757155564"><a name="b18757155564"></a><a name="b18757155564"></a>faiss::IndexIDMap*index</strong>：CPU侧Index资源。</p>
</td>
</tr>
<tr id="row61781514507"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p417181516501"><a name="p417181516501"></a><a name="p417181516501"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1617181515015"><a name="p1617181515015"></a><a name="p1617181515015"></a>无</p>
</td>
</tr>
<tr id="row917171512503"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p11172156506"><a name="p11172156506"></a><a name="p11172156506"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p317215145014"><a name="p317215145014"></a><a name="p317215145014"></a>无</p>
</td>
</tr>
<tr id="row6179153503"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p20171415125010"><a name="p20171415125010"></a><a name="p20171415125010"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p191791513507"><a name="p191791513507"></a><a name="p191791513507"></a><span class="parmname" id="parmname169852091361"><a name="parmname169852091361"></a><a name="parmname169852091361"></a>“index”</span>需要为合法有效的IndexIDMap指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>


#### getBase接口<a name="ZH-CN_TOPIC_0000001456694928"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getBase(int deviceId, char* xb) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexSQ在特定<span class="parmname" id="parmname1731761464915"><a name="parmname1731761464915"></a><a name="parmname1731761464915"></a>“deviceId”</span>上管理的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b0928171820610"><a name="b0928171820610"></a><a name="b0928171820610"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b1959712112614"><a name="b1959712112614"></a><a name="b1959712112614"></a>char* xb</strong>：AscendIndexSQ在<span class="parmname" id="parmname132844390614"><a name="parmname132844390614"></a><a name="parmname132844390614"></a>“deviceId”</span>上存储的底库特征向量。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul126112716618"></a><a name="ul126112716618"></a><ul id="ul126112716618"><li><span class="parmname" id="parmname12140629560"><a name="parmname12140629560"></a><a name="parmname12140629560"></a>“deviceId”</span>需要为合法的设备ID。</li><li><span class="parmname" id="parmname14391592819"><a name="parmname14391592819"></a><a name="parmname14391592819"></a>“xb”</span>需要为非空指针，且长度应该为dims * BaseSize * sizeof(uint8_t)字节，否则可能出现越界读写错误并引起程序崩溃，其中BaseSize为getBaseSize函数的返回值。</li></ul>
</td>
</tr>
</tbody>
</table>


#### getBaseSize接口<a name="ZH-CN_TOPIC_0000001456854788"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>size_t getBaseSize(int deviceId) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexSQ在特定<span class="parmname" id="parmname17430117154914"><a name="parmname17430117154914"></a><a name="parmname17430117154914"></a>“deviceId”</span>上管理的特征向量数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b10691175516203"><a name="b10691175516203"></a><a name="b10691175516203"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>在特定<span class="parmname" id="parmname9112805213"><a name="parmname9112805213"></a><a name="parmname9112805213"></a>“deviceId”</span>上的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname57601417219"><a name="parmname57601417219"></a><a name="parmname57601417219"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>


#### getIdxMap接口<a name="ZH-CN_TOPIC_0000001456375152"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt;&amp; idxMap) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该AscendIndexSQ在特定<span class="parmname" id="parmname12367174413480"><a name="parmname12367174413480"></a><a name="parmname12367174413480"></a>“deviceId”</span>上管理的特征向量ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b156818551961"><a name="b156818551961"></a><a name="b156818551961"></a>int deviceId</strong>：Device侧设备ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b69151357768"><a name="b69151357768"></a><a name="b69151357768"></a>std::vector&lt;idx_t&gt; &amp;idxMap</strong>：AscendIndexSQ在<span class="parmname" id="parmname101352016714"><a name="parmname101352016714"></a><a name="parmname101352016714"></a>“deviceId”</span>上存储的底库特征向量ID 。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname47241030712"><a name="parmname47241030712"></a><a name="parmname47241030712"></a>“deviceId”</span>需要为合法的设备ID。</p>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000001456375300"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7779180105218"><a name="p7779180105218"></a><a name="p7779180105218"></a>AscendIndexSQ&amp; operator=(const AscendIndexSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b16267637049"><a name="b16267637049"></a><a name="b16267637049"></a>const AscendIndexSQ&amp;</strong>：AscendIndexSQ对象。</p>
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


#### search\_with\_filter接口<a name="ZH-CN_TOPIC_0000001810589742"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p676092910161"><a name="p676092910161"></a><a name="p676092910161"></a>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10290157145418"><a name="p10290157145418"></a><a name="p10290157145418"></a>AscendIndexSQ的特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname1390811816443"><a name="parmname1390811816443"></a><a name="parmname1390811816443"></a>“k”</span>条特征的ID。提供基于CID过滤的功能，“filters”为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b1976572871110"><a name="b1976572871110"></a><a name="b1976572871110"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b12370123112117"><a name="b12370123112117"></a><a name="b12370123112117"></a>const float *x</strong>：特征向量数据。</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b3869633101113"><a name="b3869633101113"></a><a name="b3869633101113"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p13978130112613"><a name="p13978130112613"></a><a name="p13978130112613"></a><strong id="b157981335181116"><a name="b157981335181116"></a><a name="b157981335181116"></a>const void *filters</strong>：过滤条件。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b7967538161117"><a name="b7967538161117"></a><a name="b7967538161117"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname20628131394417"><a name="parmname20628131394417"></a><a name="parmname20628131394417"></a>“k”</span>个向量间的距离值。</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b34371240191113"><a name="b34371240191113"></a><a name="b34371240191113"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname14669121564412"><a name="parmname14669121564412"></a><a name="parmname14669121564412"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul6584134771119"></a><a name="ul6584134771119"></a><ul id="ul6584134771119"><li>此处<span class="parmname" id="parmname1251142183215"><a name="parmname1251142183215"></a><a name="parmname1251142183215"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处<span class="parmname" id="parmname166023416327"><a name="parmname166023416327"></a><a name="parmname166023416327"></a>“k”</span>通常不允许超过4096。</li><li>此处指针<span class="parmname" id="parmname1281764124"><a name="parmname1281764124"></a><a name="parmname1281764124"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname10844578120"><a name="parmname10844578120"></a><a name="parmname10844578120"></a>“distances”</span>/<span class="parmname" id="parmname1668917981220"><a name="parmname1668917981220"></a><a name="parmname1668917981220"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b1939083417308"><a name="b1939083417308"></a><a name="b1939083417308"></a>k</strong> * <strong id="b8896138133019"><a name="b8896138133019"></a><a name="b8896138133019"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname18187812201214"><a name="parmname18187812201214"></a><a name="parmname18187812201214"></a>“filters”</span>需要为非空指针，且长度为n * 6的uint32_t的数组，否则可能出现越界读的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001456694932"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11977033135012"><a name="p11977033135012"></a><a name="p11977033135012"></a>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p7584153011582"><a name="p7584153011582"></a><a name="p7584153011582"></a>AscendIndexSQ的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为<strong id="b177781744812"><a name="b177781744812"></a><a name="b177781744812"></a>0</strong>、<strong id="b61111618184814"><a name="b61111618184814"></a><a name="b61111618184814"></a>1</strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong id="b18384163024810"><a name="b18384163024810"></a><a name="b18384163024810"></a>1</strong>参与，<strong id="b2070119316483"><a name="b2070119316483"></a><a name="b2070119316483"></a>0</strong>不参与。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b8509184711813"><a name="b8509184711813"></a><a name="b8509184711813"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p1332473802314"><a name="p1332473802314"></a><a name="p1332473802314"></a><strong id="b2086114494814"><a name="b2086114494814"></a><a name="b2086114494814"></a>const float *x</strong>：特征向量数据。</p>
<p id="p173513403239"><a name="p173513403239"></a><a name="p173513403239"></a><strong id="b2484155112813"><a name="b2484155112813"></a><a name="b2484155112813"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p841235065815"><a name="p841235065815"></a><a name="p841235065815"></a><strong id="b77116531187"><a name="b77116531187"></a><a name="b77116531187"></a>const void *mask：</strong>特征底库掩码。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b676467798"><a name="b676467798"></a><a name="b676467798"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname1152618119407"><a name="parmname1152618119407"></a><a name="parmname1152618119407"></a>“k”</span>个向量间的距离值。</p>
<p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b5824159193"><a name="b5824159193"></a><a name="b5824159193"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname145221464014"><a name="parmname145221464014"></a><a name="parmname145221464014"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul184581435495"></a><a name="ul184581435495"></a><ul id="ul184581435495"><li>此处<span class="parmname" id="parmname11945191043317"><a name="parmname11945191043317"></a><a name="parmname11945191043317"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处<span class="parmname" id="parmname118167143338"><a name="parmname118167143338"></a><a name="parmname118167143338"></a>“k”</span>通常不允许超过4096。</li><li>此处指针<span class="parmname" id="parmname2847104218106"><a name="parmname2847104218106"></a><a name="parmname2847104218106"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname142691946111020"><a name="parmname142691946111020"></a><a name="parmname142691946111020"></a>“distances”</span>/<span class="parmname" id="parmname2586184851015"><a name="parmname2586184851015"></a><a name="parmname2586184851015"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b101511878307"><a name="b101511878307"></a><a name="b101511878307"></a>k</strong> * <strong id="b717661318306"><a name="b717661318306"></a><a name="b717661318306"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>此处指针<span class="parmname" id="parmname199479521105"><a name="parmname199479521105"></a><a name="parmname199479521105"></a>“mask”</span>需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。</li><li>mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。</li><li>使用该接口要求底库存储在一个device中，否则可能导致过滤结果有误。</li></ul>
</td>
</tr>
</tbody>
</table>


#### train接口<a name="ZH-CN_TOPIC_0000001506414905"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void train(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对AscendIndexSQ执行训练量化器，继承AscendFaiss中的接口，并提供具体的实现。<strong id="b103001027278"><a name="b103001027278"></a><a name="b103001027278"></a>注意，执行add之前必须对Index进行train。</strong></p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b986713077"><a name="b986713077"></a><a name="b986713077"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b11961715876"><a name="b11961715876"></a><a name="b11961715876"></a>const float *x</strong>：特征向量数据。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1796193986"></a><a name="ul1796193986"></a><ul id="ul1796193986"><li>此处<span class="parmname" id="parmname4258436173317"><a name="parmname4258436173317"></a><a name="parmname4258436173317"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname14391592819"><a name="parmname14391592819"></a><a name="parmname14391592819"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>训练会统计的数据的分布，训练集比较小可能会影响查询精度。</li></ul>
</td>
</tr>
</tbody>
</table>



### AscendIndexSQConfig<a name="ZH-CN_TOPIC_0000001456375392"></a>

AscendIndexSQ需要使用对应的AscendIndexSQConfig执行对应资源的初始化。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>inline AscendIndexSQConfig()</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexSQConfig的默认构造函数，默认指定的deviceList为0（即指定NPU的第0个<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>作为AscendFaiss执行检索的异构计算平台），采用默认的资源池大小。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a>无</p>
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

<a name="table108621239568"></a>
<table><tbody><tr id="row1686242395610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p78621239565"><a name="p78621239565"></a><a name="p78621239565"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p133718164310"><a name="p133718164310"></a><a name="p133718164310"></a>inline AscendIndexSQConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t  blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row178624230566"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7862192305612"><a name="p7862192305612"></a><a name="p7862192305612"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1686232319567"><a name="p1686232319567"></a><a name="p1686232319567"></a>AscendIndexSQConfig的构造函数，生成AscendIndexSQConfig，此时根据<span class="parmname" id="parmname113412141400"><a name="parmname113412141400"></a><a name="parmname113412141400"></a>“devices”</span>中配置的值设置Device侧<span id="ph126659211576"><a name="ph126659211576"></a><a name="ph126659211576"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row886222375617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2862723165612"><a name="p2862723165612"></a><a name="p2862723165612"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p128621723155611"><a name="p128621723155611"></a><a name="p128621723155611"></a><strong id="b18990511018"><a name="b18990511018"></a><a name="b18990511018"></a>std::initializer_list&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b64851871304"><a name="b64851871304"></a><a name="b64851871304"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的<span class="parmname" id="parmname1539165725917"><a name="parmname1539165725917"></a><a name="parmname1539165725917"></a>“SQ_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p10918131681313"><a name="p10918131681313"></a><a name="p10918131681313"></a><strong id="b31638114817"><a name="b31638114817"></a><a name="b31638114817"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size，默认值为16384 * 16 = 262144，该值会影响最大可创建Index的数量与检索的性能。</p>
</td>
</tr>
<tr id="row986352311564"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p7863823135618"><a name="p7863823135618"></a><a name="p7863823135618"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1286342314562"><a name="p1286342314562"></a><a name="p1286342314562"></a>无</p>
</td>
</tr>
<tr id="row0863723185611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p208632023155619"><a name="p208632023155619"></a><a name="p208632023155619"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1386313230564"><a name="p1386313230564"></a><a name="p1386313230564"></a>无</p>
</td>
</tr>
<tr id="row486382311561"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15863423115615"><a name="p15863423115615"></a><a name="p15863423115615"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul17466113116595"></a><a name="ul17466113116595"></a><ul id="ul17466113116595"><li><span class="parmname" id="parmname58171617204"><a name="parmname58171617204"></a><a name="parmname58171617204"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname15293195861219"><a name="parmname15293195861219"></a><a name="parmname15293195861219"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue1117872319298"><a name="parmvalue1117872319298"></a><a name="parmvalue1117872319298"></a>“-1”</span>时，Device侧<span id="ph18863142385614"><a name="ph18863142385614"></a><a name="ph18863142385614"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li><li><span class="parmname" id="parmname10816837131615"><a name="parmname10816837131615"></a><a name="parmname10816837131615"></a>“blockSize”</span>可配置的值的集合为{16384 * 8，16384 * 16，16384 * 32，16384 * 64}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1735412445711"></a>
<table><tbody><tr id="row19354134175714"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1635417413572"><a name="p1635417413572"></a><a name="p1635417413572"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19780437164418"><a name="p19780437164418"></a><a name="p19780437164418"></a>inline AscendIndexSQConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t  blockSize = DEFAULT_BLOCK_SIZE)</p>
</td>
</tr>
<tr id="row93540419578"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1635420425713"><a name="p1635420425713"></a><a name="p1635420425713"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13546445718"><a name="p13546445718"></a><a name="p13546445718"></a>AscendIndexSQConfig的构造函数，生成AscendIndexSQConfig，此时根据<span class="parmname" id="parmname387713394011"><a name="parmname387713394011"></a><a name="parmname387713394011"></a>“devices”</span>中配置的值设置Device侧<span id="ph32441233165718"><a name="ph32441233165718"></a><a name="ph32441233165718"></a>昇腾AI处理器</span>资源，配置资源池大小。</p>
</td>
</tr>
<tr id="row33541741571"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1135414185711"><a name="p1135414185711"></a><a name="p1135414185711"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1135410418574"><a name="p1135410418574"></a><a name="p1135410418574"></a><strong id="b3303144712017"><a name="b3303144712017"></a><a name="b3303144712017"></a>std::vector&lt;int&gt; devices</strong>：Device侧设备ID。</p>
<p id="p13541545573"><a name="p13541545573"></a><a name="p13541545573"></a><strong id="b197022051409"><a name="b197022051409"></a><a name="b197022051409"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中定义的<span class="parmname" id="parmname439611564019"><a name="parmname439611564019"></a><a name="parmname439611564019"></a>“SQ_DEFAULT_MEM”</span>。该参数通过底库大小和search的batch数共同确定，在底库大于或等于1000万且batch数大于或等于16时建议设置1024MB。</p>
<p id="p1035454195716"><a name="p1035454195716"></a><a name="p1035454195716"></a><strong id="b12664181414917"><a name="b12664181414917"></a><a name="b12664181414917"></a>uint32_t blockSize</strong>：配置Device侧的blockSize，约束tik算子一次计算的数据量，以及底库分片存储每片存储向量的size，默认值为16384 * 16 = 262144，该值会影响最大可创建Index的数量与检索的性能。</p>
</td>
</tr>
<tr id="row2354104115713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14354184105712"><a name="p14354184105712"></a><a name="p14354184105712"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p43540455713"><a name="p43540455713"></a><a name="p43540455713"></a>无</p>
</td>
</tr>
<tr id="row1354442570"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p20354749572"><a name="p20354749572"></a><a name="p20354749572"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p163541948576"><a name="p163541948576"></a><a name="p163541948576"></a>无</p>
</td>
</tr>
<tr id="row2354174135711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p153545418579"><a name="p153545418579"></a><a name="p153545418579"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1931270712"></a><a name="ul1931270712"></a><ul id="ul1931270712"><li><span class="parmname" id="parmname639724115"><a name="parmname639724115"></a><a name="parmname639724115"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname14354144195712"><a name="parmname14354144195712"></a><a name="parmname14354144195712"></a>“resourceSize”</span>配置的值不超过4 * 1024MB（4 * 1024 * 1024 * 1024字节），当设置为<span class="parmvalue" id="parmvalue1294113452910"><a name="parmvalue1294113452910"></a><a name="parmvalue1294113452910"></a>“-1”</span>时，Device侧<span id="ph1135454195716"><a name="ph1135454195716"></a><a name="ph1135454195716"></a>昇腾AI处理器</span>资源配置为默认值128MB。</li><li><span class="parmname" id="parmname635415445713"><a name="parmname635415445713"></a><a name="parmname635415445713"></a>“blockSize”</span>可配置的值的集合为{16384 * 8，16384 * 16，16384 * 32，16384 * 64}。</li></ul>
</td>
</tr>
</tbody>
</table>


### IndexIL<a name="ZH-CN_TOPIC_0000001506414825"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456535188"></a>

IndexIL是一个基于连续内存申请机制的特征管理抽象类，服务于将下标索引作为label的检索算法，需要继承实现所有接口来使用。

要求存入底库的向量以及各个接口的query向量均为归一化后的FP16浮点数类型。（**IL**表示“Indices as Labels”。）

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。


#### AddFeatures接口<a name="ZH-CN_TOPIC_0000001506414693"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>virtual APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>向特征库插入<span class="parmname" id="parmname859219315012"><a name="parmname859219315012"></a><a name="parmname859219315012"></a>“n”</span>个指定下标索引的特征向量，如果在下标处已存在特征向量，该插入操作相当于修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b6166642122410"><a name="b6166642122410"></a><a name="b6166642122410"></a>int n</strong>：插入特征向量数目。</p>
<p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b61461844172410"><a name="b61461844172410"></a><a name="b61461844172410"></a>const float16_t *features</strong>：特征向量，长度为n * 向量维度dim。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b416612460248"><a name="b416612460248"></a><a name="b416612460248"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li>入参由该类的实现类约束。</li><li><span class="parmname" id="parmname163173251387"><a name="parmname163173251387"></a><a name="parmname163173251387"></a>“features”</span>和<span class="parmname" id="parmname171047408382"><a name="parmname171047408382"></a><a name="parmname171047408382"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### IndexIL接口<a name="ZH-CN_TOPIC_0000001456695020"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>IndexIL();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>IndexIL的构造函数，生成特征管理对象。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### \~IndexIL接口<a name="ZH-CN_TOPIC_0000001506334781"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>virtual ~IndexIL();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>IndexIL的析构函数，销毁特征管理对象。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### Finalize接口<a name="ZH-CN_TOPIC_0000001456375356"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1997210581407"><a name="p1997210581407"></a><a name="p1997210581407"></a>virtual APP_ERROR Finalize() = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>释放特征库管理资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b9837134732319"><a name="b9837134732319"></a><a name="b9837134732319"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### GetFeatures接口<a name="ZH-CN_TOPIC_0000001506495833"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>virtual APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname117615186507"><a name="parmname117615186507"></a><a name="parmname117615186507"></a>“n”</span>条指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b7444782517"><a name="b7444782517"></a><a name="b7444782517"></a>int n</strong>：获取特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b144851412192517"><a name="b144851412192517"></a><a name="b144851412192517"></a>const idx_t *indices</strong>：待查询的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b16464142810254"><a name="b16464142810254"></a><a name="b16464142810254"></a>float16_t *features</strong>：查询下标索引对应的特征向量，长度为n * 向量维度dim。在调用前需由用户自行申请内存，确保内存大小正确。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1557493018253"><a name="b1557493018253"></a><a name="b1557493018253"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li>入参由该类的实现类约束。</li><li><span class="parmname" id="parmname178666115369"><a name="parmname178666115369"></a><a name="parmname178666115369"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### GetNTotal接口<a name="ZH-CN_TOPIC_0000001456535092"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>virtual int GetNTotal() const = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p189931428115213"><a name="p189931428115213"></a><a name="p189931428115213"></a>查询当前特征库向量的最大占用空间。</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>特征向量从索引<strong id="b320263895212"><a name="b320263895212"></a><a name="b320263895212"></a>0</strong>开始插入，如果插入特征向量<span class="parmname" id="parmname3491165415711"><a name="parmname3491165415711"></a><a name="parmname3491165415711"></a>“indices”</span>连续，则<span class="parmname" id="parmname16149452675"><a name="parmname16149452675"></a><a name="parmname16149452675"></a>“ntotal”</span>等于特征向量数目，否则<span class="parmname" id="parmname895125617717"><a name="parmname895125617717"></a><a name="parmname895125617717"></a>“ntotal”</span>等于插入向量的最大索引值加1（为性能考虑，算子会批操作内存，默认将最大索引位置及之前的空间都视为有效底库向量并纳入计算），用户需要通过该接口获取index内部记录的底库总量，进而申请对应的内存空间给对应的功能接口传递参数，详细描述请参见具体接口。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a>int ntotal</strong>：请参见功能描述。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### Init接口<a name="ZH-CN_TOPIC_0000001506334657"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1939248184018"><a name="p1939248184018"></a><a name="p1939248184018"></a>virtual APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize) = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>初始化特征库参数，申请底库内存资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b156115254222"><a name="b156115254222"></a><a name="b156115254222"></a>int dim</strong>：特征向量的维度。</p>
<p id="p1889154465814"><a name="p1889154465814"></a><a name="p1889154465814"></a><strong id="b0291530172212"><a name="b0291530172212"></a><a name="b0291530172212"></a>AscendMetricType metricType</strong>： 特征距离类别：向量内积、欧氏距离、余弦相似度。</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8478173318223"><a name="b8478173318223"></a><a name="b8478173318223"></a>int capacity</strong>：底库最大容量，等于capacity * dim * sizeof(float) 字节内存数据。</p>
<p id="p12851134435520"><a name="p12851134435520"></a><a name="p12851134435520"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a>int resourceSize</strong>：提前申请Device的缓存资源，检索接口被调用时可以直接使用这里的资源，而不必调用<strong id="b53620313280"><a name="b53620313280"></a><a name="b53620313280"></a>aclrtmalloc</strong>去申请内存，达到优化加速。默认取值-1，代表按默认size申请缓存资源（128MB），可以根据检索业务的数据量和Device上的资源使用情况来更精确地配置实际需要使用的size大小。</p>
<p id="p1703214386"><a name="p1703214386"></a><a name="p1703214386"></a>例如：query的<span class="parmname" id="parmname29769985615"><a name="parmname29769985615"></a><a name="parmname29769985615"></a>“batch”</span>为<span class="parmvalue" id="parmvalue134454133566"><a name="parmvalue134454133566"></a><a name="parmvalue134454133566"></a>“64”</span>，底库总量为100万，而一个FP32数值占用4个字节，那么这里的<span class="parmname" id="parmname520812617569"><a name="parmname520812617569"></a><a name="parmname520812617569"></a>“resourceSize”</span>可以设置为： 64 * 1000000 * 4 = 256,000,000Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b18120185612228"><a name="b18120185612228"></a><a name="b18120185612228"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p392515311255"><a name="p392515311255"></a><a name="p392515311255"></a>入参由该类的实现类进行约束。</p>
</td>
</tr>
</tbody>
</table>


#### RemoveFeatures接口<a name="ZH-CN_TOPIC_0000001456534932"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>virtual APP_ERROR RemoveFeatures(int n, const idx_t *indices) = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>删除向量库中<span class="parmname" id="parmname17685161175014"><a name="parmname17685161175014"></a><a name="parmname17685161175014"></a>“n”</span>个指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b7444782517"><a name="b7444782517"></a><a name="b7444782517"></a>int n</strong>：删除特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b144851412192517"><a name="b144851412192517"></a><a name="b144851412192517"></a>const idx_t *indices</strong>：特征向量对应的下标索引。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b2046121817254"><a name="b2046121817254"></a><a name="b2046121817254"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li>入参由该类的实现类约束。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>需要为非空指针，且长度应该为n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### SetNTotal接口<a name="ZH-CN_TOPIC_0000001456375256"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>virtual APP_ERROR SetNTotal(int n) = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p57481224513"><a name="p57481224513"></a><a name="p57481224513"></a>为外部提供调整<span class="parmname" id="parmname14317663915"><a name="parmname14317663915"></a><a name="parmname14317663915"></a>“ntotal”</span>计数的接口。</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>每次增加底库向量后，Index内部虽然会根据最大插入下标更新<span class="parmname" id="parmname481913338914"><a name="parmname481913338914"></a><a name="parmname481913338914"></a>“ntotal”</span>值，但并没有记录[0, <i><span class="varname" id="varname1598113714914"><a name="varname1598113714914"></a><a name="varname1598113714914"></a>ntotal</span></i>]范围内哪些区域是无效的空间，因此<strong id="b912034151411"><a name="b912034151411"></a><a name="b912034151411"></a>RemoveFeatures</strong>操作没有改变<span class="parmname" id="parmname16611125391413"><a name="parmname16611125391413"></a><a name="parmname16611125391413"></a>“ntotal”</span>的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置<span class="parmname" id="parmname1570145018914"><a name="parmname1570145018914"></a><a name="parmname1570145018914"></a>“ntotal”</span>，这样可以在可控范围内减少算子的计算量，以提高接口性能。</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>例如：当前插入100条向量，底库索引为0~99时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部<span class="parmname" id="parmname192561317117"><a name="parmname192561317117"></a><a name="parmname192561317117"></a>“ntotal”</span>保持不变，只能设为[<i><span class="varname" id="varname1664545820118"><a name="varname1664545820118"></a><a name="varname1664545820118"></a>ntotal</span></i>, <i><span class="varname" id="varname1710201151216"><a name="varname1710201151216"></a><a name="varname1710201151216"></a>capacity</span></i>]之间的值，再次执行删除索引为90~99的底库，此时可以手动把<span class="parmname" id="parmname4538102481220"><a name="parmname4538102481220"></a><a name="parmname4538102481220"></a>“ntotal”</span>设置为[80, <i><span class="varname" id="varname17333173011213"><a name="varname17333173011213"></a><a name="varname17333173011213"></a>capacity</span></i>]之间的值，设置为<span class="parmvalue" id="parmvalue06651847101219"><a name="parmvalue06651847101219"></a><a name="parmvalue06651847101219"></a>“80”</span>时，可以使参与比对的底库数据量有效减少20条。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b8131171512139"><a name="b8131171512139"></a><a name="b8131171512139"></a>int n</strong>：由用户在业务面管理的最大底库的索引加1。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>入参由该类的实现类约束。</p>
</td>
</tr>
</tbody>
</table>



### IndexILFlat<a name="ZH-CN_TOPIC_0000001506614925"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001506414785"></a>

IndexILFlat继承自IndexIL，为纯Device侧检索方案，利用昇腾AI处理器和AI Core等资源进行各个接口的使能。程序需要在Host侧编译生成二进制文件，然后将二进制文件和相关运行时依赖部署到Device侧执行。IndexILFlat需要使用[Init](#init接口)指定对应资源的初始化，初始化完之后会申请一段完整空间用于存储底库。在使用完之后，需要调用[Finalize](#finalize接口)接口对资源进行释放。

IndexILFlat方案当前只在Atlas 推理系列产品上进行功能和性能的维护，底库和query向量由用户保证归一化，接口当前仅支持向量内积距离，具体使用方法请参见[IndexILFlat](#indexilflat)。（该算法运行成功依赖TIK算子的om文件，纯Device场景需要用户确保部署的是基于Index SDK交付件生成的om文件，需要确保om文件不被篡改。）

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。


#### AddFeatures接口<a name="ZH-CN_TOPIC_0000001456854852"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>向特征库插入<span class="parmname" id="parmname91461549125412"><a name="parmname91461549125412"></a><a name="parmname91461549125412"></a>“n”</span>个指定下标索引的特征向量，如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b645815911297"><a name="b645815911297"></a><a name="b645815911297"></a>int n</strong>：插入特征向量数目。</p>
<p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b118193183015"><a name="b118193183015"></a><a name="b118193183015"></a>const float16_t *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b119812563013"><a name="b119812563013"></a><a name="b119812563013"></a>const idx_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1835741212302"><a name="b1835741212302"></a><a name="b1835741212302"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul10674191294110"></a><a name="ul10674191294110"></a><ul id="ul10674191294110"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname423619296407"><a name="varname423619296407"></a><a name="varname423619296407"></a>capacity</span></i>)之间。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname3816191310019"><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### ComputeDistance接口<a name="ZH-CN_TOPIC_0000001456535116"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19771939144811"><a name="p19771939144811"></a><a name="p19771939144811"></a>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname11281111175619"><a name="parmname11281111175619"></a><a name="parmname11281111175619"></a>“n”</span>条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0 且table为非空指针），则输出经过映射后的距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b75131329183314"><a name="b75131329183314"></a><a name="b75131329183314"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b6862131183312"><a name="b6862131183312"></a><a name="b6862131183312"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b1041123433313"><a name="b1041123433313"></a><a name="b1041123433313"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue354895917522"><a name="parmvalue354895917522"></a><a name="parmvalue354895917522"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b376183643316"><a name="b376183643316"></a><a name="b376183643316"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname834710403338"><a name="parmname834710403338"></a><a name="parmname834710403338"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue16873151054312"><a name="parmvalue16873151054312"></a><a name="parmvalue16873151054312"></a>“48”</span>，即<span class="parmname" id="parmname135806266430"><a name="parmname135806266430"></a><a name="parmname135806266430"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1750033215518"><a name="p1750033215518"></a><a name="p1750033215518"></a><strong id="b668194911337"><a name="b668194911337"></a><a name="b668194911337"></a>float *distances</strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（<span class="parmname" id="parmname10121121717561"><a name="parmname10121121717561"></a><a name="parmname10121121717561"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname13631434155615"><a name="varname13631434155615"></a><a name="varname13631434155615"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname0810103815562"><a name="parmname0810103815562"></a><a name="parmname0810103815562"></a>“ntotal”</span>对16补齐）。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul167968714447"></a><a name="ul167968714447"></a><ul id="ul167968714447"><li><strong id="b1141410121444"><a name="b1141410121444"></a><a name="b1141410121444"></a>n</strong>：合理的n值应该在[0, <i><span class="varname" id="varname1835520162442"><a name="varname1835520162442"></a><a name="varname1835520162442"></a>capacity</span></i>]之间。</li><li><strong id="b429253634917"><a name="b429253634917"></a><a name="b429253634917"></a>distances</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname1598134614491"><a name="parmname1598134614491"></a><a name="parmname1598134614491"></a>“ntotalPad”</span>为(<i><span class="varname" id="varname1654664914915"><a name="varname1654664914915"></a><a name="varname1654664914915"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname145712539496"><a name="parmname145712539496"></a><a name="parmname145712539496"></a>“ntotal”</span>对16补齐的结果，每个query的有效比对距离存储在前<span class="parmname" id="parmname15322121501"><a name="parmname15322121501"></a><a name="parmname15322121501"></a>“ntotal”</span>的空间，补齐部分数据没有实际意义）。<p id="p0715202405018"><a name="p0715202405018"></a><a name="p0715202405018"></a>推荐使用<strong id="b1894113045017"><a name="b1894113045017"></a><a name="b1894113045017"></a>aclrtmalloc</strong>接口，可以申请到全量的物理内存来使用，优化处理时延。</p>
</li><li>传递<span class="parmname" id="parmname391111228612"><a name="parmname391111228612"></a><a name="parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="parmname19267121920619"><a name="parmname19267121920619"></a><a name="parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="b5371616181211"><a name="b5371616181211"></a><a name="b5371616181211"></a>distance</strong>进行映射：<p id="p1129513513121"><a name="p1129513513121"></a><a name="p1129513513121"></a>首先将<strong id="b13840714131216"><a name="b13840714131216"></a><a name="b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="b7555131016121"><a name="b7555131016121"></a><a name="b7555131016121"></a>f1</strong>，然后用<strong id="b199806129123"><a name="b199806129123"></a><a name="b199806129123"></a>f1</strong>乘上<span class="parmname" id="parmname14917143791"><a name="parmname14917143791"></a><a name="parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="b1399121919123"><a name="b1399121919123"></a><a name="b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="parmname266193771110"><a name="parmname266193771110"></a><a name="parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="b12230192011219"><a name="b12230192011219"></a><a name="b12230192011219"></a>score</strong>，即完成映射，将<strong id="b1622952141216"><a name="b1622952141216"></a><a name="b1622952141216"></a>score</strong>存入<span class="parmname" id="parmname106381556121113"><a name="parmname106381556121113"></a><a name="parmname106381556121113"></a>“distance”</span> 。</p>
<p id="p340315471018"><a name="p340315471018"></a><a name="p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000001456694920"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p7384196195012"><a name="p7384196195012"></a><a name="p7384196195012"></a>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>与ComputeDistance类似，区别在于ComputeDistance计算待查询向量与所有底库向量的距离，而该接口ComputeDistanceByIdx只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1178514265435"><a name="b1178514265435"></a><a name="b1178514265435"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b79742028204313"><a name="b79742028204313"></a><a name="b79742028204313"></a>const float16_t *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname1441759144217"><a name="parmname1441759144217"></a><a name="parmname1441759144217"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a>const int *num</strong>： 给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b632503394315"><a name="b632503394315"></a><a name="b632503394315"></a>const idx_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname2711154912437"><a name="parmname2711154912437"></a><a name="parmname2711154912437"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname742124364316"><a name="parmname742124364316"></a><a name="parmname742124364316"></a>“indices”</span>长度为n * max(num)。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b156251035184313"><a name="b156251035184313"></a><a name="b156251035184313"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b25863377438"><a name="b25863377438"></a><a name="b25863377438"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname136035014443"><a name="parmname136035014443"></a><a name="parmname136035014443"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue2069717180212"><a name="parmvalue2069717180212"></a><a name="parmvalue2069717180212"></a>“48”</span>，即<span class="parmname" id="parmname1997224217"><a name="parmname1997224217"></a><a name="parmname1997224217"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p057182814222"><a name="p057182814222"></a><a name="p057182814222"></a><strong id="b0194557446"><a name="b0194557446"></a><a name="b0194557446"></a>float *distances</strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大<span class="parmname" id="parmname658971354417"><a name="parmname658971354417"></a><a name="parmname658971354417"></a>“num”</span>补齐空间占用，空间长度为n * max(num)。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b140412864414"><a name="b140412864414"></a><a name="b140412864414"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname82723561324"><a name="varname82723561324"></a><a name="varname82723561324"></a>capacity</span></i>]之间。</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0， ntotal]之间。</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname343119418149"><a name="parmname343119418149"></a><a name="parmname343119418149"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### ComputeDistanceByThreshold接口<a name="ZH-CN_TOPIC_0000001506615117"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p137564208498"><a name="p137564208498"></a><a name="p137564208498"></a>APP_ERROR ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>在ComputeDistance的基础上增加了阈值筛选，只返回满足阈值条件的距离。如传递有效的映射表（tableLen &gt; 0且*table为非空指针），则distances为映射后再进行阈值过滤的结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b6232619123815"><a name="b6232619123815"></a><a name="b6232619123815"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p144875610364"><a name="p144875610364"></a><a name="p144875610364"></a><strong id="b6143623203816"><a name="b6143623203816"></a><a name="b6143623203816"></a>float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p1924692795017"><a name="p1924692795017"></a><a name="p1924692795017"></a><strong id="b183957257385"><a name="b183957257385"></a><a name="b183957257385"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname18151822141710"><a name="parmname18151822141710"></a><a name="parmname18151822141710"></a>“threshold”</span>进行过滤。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b124943270387"><a name="b124943270387"></a><a name="b124943270387"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b13344132915381"><a name="b13344132915381"></a><a name="b13344132915381"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname15667378386"><a name="parmname15667378386"></a><a name="parmname15667378386"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue729933145515"><a name="parmvalue729933145515"></a><a name="parmvalue729933145515"></a>“48”</span>，即*table指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b5394194013386"><a name="b5394194013386"></a><a name="b5394194013386"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量长度为n。</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b787564210382"><a name="b787564210382"></a><a name="b787564210382"></a>idx_t *indices</strong>：满足阈值条件的底库向量下标索引，每个query符合条件的底库数量不同，当从前往后记录所有有效的index之后，按<span class="parmname" id="parmname887025614551"><a name="parmname887025614551"></a><a name="parmname887025614551"></a>“ntotalPad”</span>补齐占用的空间，<span class="parmname" id="parmname1488799145620"><a name="parmname1488799145620"></a><a name="parmname1488799145620"></a>“indices”</span>的总长度应该为n * nTotalPad（<span class="parmname" id="parmname10121121717561"><a name="parmname10121121717561"></a><a name="parmname10121121717561"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname13631434155615"><a name="varname13631434155615"></a><a name="varname13631434155615"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname0810103815562"><a name="parmname0810103815562"></a><a name="parmname0810103815562"></a>“ntotal”</span>对16补齐）。</p>
<p id="p03841120175217"><a name="p03841120175217"></a><a name="p03841120175217"></a><strong id="b17674164983818"><a name="b17674164983818"></a><a name="b17674164983818"></a>float *distances</strong>：满足阈值条件的底库向量与待查向量距离，有效值记录方式和空间size与<span class="parmname" id="parmname1581985716568"><a name="parmname1581985716568"></a><a name="parmname1581985716568"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b292575217384"><a name="b292575217384"></a><a name="b292575217384"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul192831928125717"></a><a name="ul192831928125717"></a><ul id="ul192831928125717"><li><strong id="b657843016578"><a name="b657843016578"></a><a name="b657843016578"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname1334393525713"><a name="varname1334393525713"></a><a name="varname1334393525713"></a>capacity</span></i>]之间。</li><li><strong id="b841218507570"><a name="b841218507570"></a><a name="b841218507570"></a>indices</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname127617300585"><a name="parmname127617300585"></a><a name="parmname127617300585"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname380193518587"><a name="varname380193518587"></a><a name="varname380193518587"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname1536074119588"><a name="parmname1536074119588"></a><a name="parmname1536074119588"></a>“ntotal”</span>对16补齐的结果，第<strong id="b107114541585"><a name="b107114541585"></a><a name="b107114541585"></a>i</strong>个query比对过滤后，有效底库的索引存储在<span class="parmname" id="parmname19106128135820"><a name="parmname19106128135820"></a><a name="parmname19106128135820"></a>“ntotalPad”</span>的前*(num + i) 的空间，补齐部分数据没有实际意义）。</li><li><strong id="b19683195213576"><a name="b19683195213576"></a><a name="b19683195213576"></a>distances</strong>：需要提供的空间长度为n * ntotalPad。</li><li><span class="parmname" id="parmname315020014583"><a name="parmname315020014583"></a><a name="parmname315020014583"></a>“indices”</span>和<span class="parmname" id="parmname28841122585"><a name="parmname28841122585"></a><a name="parmname28841122585"></a>“distances”</span>推荐使用<strong id="b4371184115813"><a name="b4371184115813"></a><a name="b4371184115813"></a>aclrtmalloc</strong>接口，可以申请到全量的物理内存来使用，优化处理时延。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname95571128225"><a name="parmname95571128225"></a><a name="parmname95571128225"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### Finalize接口<a name="ZH-CN_TOPIC_0000001506414845"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR Finalize() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>释放特征库管理资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1570793612442"><a name="b1570793612442"></a><a name="b1570793612442"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### GetFeatures接口<a name="ZH-CN_TOPIC_0000001456854992"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname9635334135520"><a name="parmname9635334135520"></a><a name="parmname9635334135520"></a>“n”</span>条指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a>const idx_t *indices</strong>：需要获取的n个底库向量对应的索引值。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b043314127333"><a name="b043314127333"></a><a name="b043314127333"></a>float16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### GetNTotal接口<a name="ZH-CN_TOPIC_0000001456375336"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a>int GetNTotal() const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询当前特征库特征向量数目的理论最大值。如果插入特征向量indices连续，则ntotal等于特征向量数目。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a><strong id="b445021732816"><a name="b445021732816"></a><a name="b445021732816"></a>int ntotal</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a>int ntotal</strong>：请参见功能描述。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### IndexILFlat接口<a name="ZH-CN_TOPIC_0000001456694872"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a>IndexILFlat();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>IndexILFlat的构造函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table194381755582"></a>
<table><tbody><tr id="row1438055581"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11438155155815"><a name="p11438155155815"></a><a name="p11438155155815"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1887018312271"><a name="p1887018312271"></a><a name="p1887018312271"></a>IndexILFlat(const IndexILFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row20438551584"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p74381159583"><a name="p74381159583"></a><a name="p74381159583"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p087012313276"><a name="p087012313276"></a><a name="p087012313276"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row24385511589"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p54381519581"><a name="p54381519581"></a><a name="p54381519581"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1786920310278"><a name="p1786920310278"></a><a name="p1786920310278"></a><strong id="b1129362910278"><a name="b1129362910278"></a><a name="b1129362910278"></a>const IndexILFlat&amp;：</strong>IndexILFlat对象。</p>
</td>
</tr>
<tr id="row84387585820"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1643812512585"><a name="p1643812512585"></a><a name="p1643812512585"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p154381517589"><a name="p154381517589"></a><a name="p154381517589"></a>无</p>
</td>
</tr>
<tr id="row043813535813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1443815510581"><a name="p1443815510581"></a><a name="p1443815510581"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184381457585"><a name="p184381457585"></a><a name="p184381457585"></a>无</p>
</td>
</tr>
<tr id="row2043811515580"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1643935185813"><a name="p1643935185813"></a><a name="p1643935185813"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### \~IndexILFlat接口<a name="ZH-CN_TOPIC_0000001456375172"></a>

<a name="table11904175418"></a>
<table><tbody><tr id="row49051251216"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11905115615"><a name="p11905115615"></a><a name="p11905115615"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15905125316"><a name="p15905125316"></a><a name="p15905125316"></a>virtual ~IndexILFlat();</p>
</td>
</tr>
<tr id="row139053510117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p179056510119"><a name="p179056510119"></a><a name="p179056510119"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p59051353114"><a name="p59051353114"></a><a name="p59051353114"></a>IndexILFlat的析构函数。</p>
</td>
</tr>
<tr id="row17905135915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18905456118"><a name="p18905456118"></a><a name="p18905456118"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6905656112"><a name="p6905656112"></a><a name="p6905656112"></a>无</p>
</td>
</tr>
<tr id="row199051557118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p129051050117"><a name="p129051050117"></a><a name="p129051050117"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p109051250114"><a name="p109051250114"></a><a name="p109051250114"></a>无</p>
</td>
</tr>
<tr id="row149051757115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p169052055120"><a name="p169052055120"></a><a name="p169052055120"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1690513511119"><a name="p1690513511119"></a><a name="p1690513511119"></a>无</p>
</td>
</tr>
<tr id="row29058514119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15905151318"><a name="p15905151318"></a><a name="p15905151318"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p139051951417"><a name="p139051951417"></a><a name="p139051951417"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### Init接口<a name="ZH-CN_TOPIC_0000001456375212"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize = -1) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>初始化特征库参数，申请底库内存资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1517311219268"><a name="b1517311219268"></a><a name="b1517311219268"></a>int dim</strong>：特征向量的维度。</p>
<p id="p1889154465814"><a name="p1889154465814"></a><a name="p1889154465814"></a><strong id="b637319417265"><a name="b637319417265"></a><a name="b637319417265"></a>AscendMetricType metricType</strong>： 特征距离类别（向量内积、欧氏距离、余弦相似度）。</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8628752620"><a name="b8628752620"></a><a name="b8628752620"></a>int capacity</strong>：底库最大容量，接口会根据<span class="parmname" id="parmname16513113011414"><a name="parmname16513113011414"></a><a name="parmname16513113011414"></a>“capacity”</span>值申请capacity * dim * sizeof(fp16) 字节内存数据。</p>
<p id="p1411722401512"><a name="p1411722401512"></a><a name="p1411722401512"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a>int64_t resourceSize</strong>：提前申请Device的缓存资源，检索接口被调用时可以直接使用这里的资源，而不必调用<strong id="b64449141510"><a name="b64449141510"></a><a name="b64449141510"></a>aclrtmalloc</strong>接口去申请内存，达到优化加速。</p>
<p id="p117241413167"><a name="p117241413167"></a><a name="p117241413167"></a>默认取值<span class="parmvalue" id="parmvalue207081128161516"><a name="parmvalue207081128161516"></a><a name="parmvalue207081128161516"></a>“-1”</span>，代表按默认size申请缓存资源（128MB），可以根据检索业务的数据量和Device上的资源使用情况来更精确地配置实际需要使用的size大小。</p>
<p id="p1703214386"><a name="p1703214386"></a><a name="p1703214386"></a>例如：query的<span class="parmname" id="parmname193097270169"><a name="parmname193097270169"></a><a name="parmname193097270169"></a>“batch”</span>为<span class="parmvalue" id="parmvalue9487237111617"><a name="parmvalue9487237111617"></a><a name="parmvalue9487237111617"></a>“64”</span>，底库总量为100万，而一个FP32数值占用4个字节，那么这里的<span class="parmname" id="parmname10173124210166"><a name="parmname10173124210166"></a><a name="parmname10173124210166"></a>“resourceSize”</span>可以设置为：64 * 1000000 * 4 = 256,000,000 Byte，注意接口内部支持申请的最大缓存资源为4GB。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b13793144414268"><a name="b13793144414268"></a><a name="b13793144414268"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1768605017262"></a><a name="ul1768605017262"></a><ul id="ul1768605017262"><li>dim ∈ {32, 64, 128, 256, 384, 512, 1024}</li><li>metricType：IndexILFlat目前只实现了向量内积距离，即只支持<span class="parmvalue" id="parmvalue194512012016"><a name="parmvalue194512012016"></a><a name="parmvalue194512012016"></a>“AscendMetricType::ASCEND_METRIC_INNER_PRODUCT”</span>。</li><li>capacity：接口允许为底库申请的内存上限设为12,288,000,000Byte，同时capacity的值域约束为(0, 12000000]。<a name="ul138816512117"></a><a name="ul138816512117"></a><ul id="ul138816512117"><li>以512维、FP16类型的底库向量为例，最大支持的<span class="parmname" id="parmname1593195143016"><a name="parmname1593195143016"></a><a name="parmname1593195143016"></a>“capacity”</span>为1200万(12288000000 / (512 * sizeof(fp_16)) )。</li><li>对于256维、FP16类型的底库向量，尽管内存约束支持更大的capacity，capacity最大也只能设为1200万。</li></ul>
</li><li>resourceSize：可以配置为-1或[134217728‬，4294967296]之间的值，数值的单位为Byte，相当于[128MB，4096MB]。</li></ul>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000001897140809"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4759192852812"><a name="p4759192852812"></a><a name="p4759192852812"></a>IndexILFlat&amp; operator=(const IndexILFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5787143522812"><a name="p5787143522812"></a><a name="p5787143522812"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p175601140172811"><a name="p175601140172811"></a><a name="p175601140172811"></a><strong id="b9347444182810"><a name="b9347444182810"></a><a name="b9347444182810"></a>const IndexILFlat&amp;：</strong>IndexILFlat对象。</p>
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


#### RemoveFeatures接口<a name="ZH-CN_TOPIC_0000001506414837"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR RemoveFeatures(int n, const idx_t *indices) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>删除向量库中<span class="parmname" id="parmname15331161155517"><a name="parmname15331161155517"></a><a name="parmname15331161155517"></a>“n”</span>个指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b10676733203011"><a name="b10676733203011"></a><a name="b10676733203011"></a>int n</strong>：删除特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b1248654013016"><a name="b1248654013016"></a><a name="b1248654013016"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b727535618302"><a name="b727535618302"></a><a name="b727535618302"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85131151122111"><a name="varname85131151122111"></a><a name="varname85131151122111"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### Search接口<a name="ZH-CN_TOPIC_0000001456854856"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p95721657104813"><a name="p95721657104813"></a><a name="p95721657104813"></a>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询与query向量距离最近的<span class="parmname" id="parmname66101618135916"><a name="parmname66101618135916"></a><a name="parmname66101618135916"></a>“topk”</span>个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0 且table为非空指针），则输出映射后的距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b2797917153611"><a name="b2797917153611"></a><a name="b2797917153611"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1445642010368"><a name="b1445642010368"></a><a name="b1445642010368"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1177612213614"><a name="b1177612213614"></a><a name="b1177612213614"></a>int topk</strong>：查询向量和底库的比对距离进行排序，返回<span class="parmname" id="parmname19303182555918"><a name="parmname19303182555918"></a><a name="parmname19303182555918"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b18837162415364"><a name="b18837162415364"></a><a name="b18837162415364"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b1174653015367"><a name="b1174653015367"></a><a name="b1174653015367"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname173331533163614"><a name="parmname173331533163614"></a><a name="parmname173331533163614"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue1525151355319"><a name="parmvalue1525151355319"></a><a name="parmvalue1525151355319"></a>“48”</span>，即<span class="parmname" id="parmname78791212533"><a name="parmname78791212533"></a><a name="parmname78791212533"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b187841356153711"><a name="b187841356153711"></a><a name="b187841356153711"></a>float *distances</strong>：外部内存，与query相似度最高的<strong id="b8665314543"><a name="b8665314543"></a><a name="b8665314543"></a>topk </strong>* <strong id="b1408103115411"><a name="b1408103115411"></a><a name="b1408103115411"></a>n</strong>个底库特征向量所对应的余弦距离，长度为n * topk。</p>
<p id="p1154614405264"><a name="p1154614405264"></a><a name="p1154614405264"></a><strong id="b12933115811373"><a name="b12933115811373"></a><a name="b12933115811373"></a>idx_t *indices</strong>：外部内存，返回与query相似度最高的<span class="parmname" id="parmname1346104111594"><a name="parmname1346104111594"></a><a name="parmname1346104111594"></a>“topk”</span>个底库向量对应的下标索引，长度为n * topk。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b111527319385"><a name="b111527319385"></a><a name="b111527319385"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1346615102548"></a><a name="ul1346615102548"></a><ul id="ul1346615102548"><li><strong id="b5538111715545"><a name="b5538111715545"></a><a name="b5538111715545"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname17384527155412"><a name="varname17384527155412"></a><a name="varname17384527155412"></a>capacity</span></i>]之间。</li><li><strong id="b18681518105411"><a name="b18681518105411"></a><a name="b18681518105411"></a>topk</strong>：取值应在[0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### SearchByThreshold接口<a name="ZH-CN_TOPIC_0000001456694892"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1591144334913"><a name="p1591144334913"></a><a name="p1591144334913"></a>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1893114655310"><a name="p1893114655310"></a><a name="p1893114655310"></a>在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1378124514396"><a name="b1378124514396"></a><a name="b1378124514396"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b24212483396"><a name="b24212483396"></a><a name="b24212483396"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p12923104514555"><a name="p12923104514555"></a><a name="p12923104514555"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname166164371714"><a name="parmname166164371714"></a><a name="parmname166164371714"></a>“threshold”</span>进行过滤。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1578817211311"><a name="parmname1578817211311"></a><a name="parmname1578817211311"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname279351319407"><a name="parmname279351319407"></a><a name="parmname279351319407"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue376417339011"><a name="parmvalue376417339011"></a><a name="parmvalue376417339011"></a>“48”</span>，即<span class="parmname" id="parmname19896039903"><a name="parmname19896039903"></a><a name="parmname19896039903"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b662915439408"><a name="b662915439408"></a><a name="b662915439408"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b452884674019"><a name="b452884674019"></a><a name="b452884674019"></a>idx_t *indices</strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname1633953516114"><a name="parmname1633953516114"></a><a name="parmname1633953516114"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname1829610421613"><a name="parmname1829610421613"></a><a name="parmname1829610421613"></a>“indices”</span>总长度为n * topk。</p>
<p id="p03841120175217"><a name="p03841120175217"></a><a name="p03841120175217"></a><strong id="b1581125094017"><a name="b1581125094017"></a><a name="b1581125094017"></a>float *distances</strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与<span class="parmname" id="parmname681785019417"><a name="parmname681785019417"></a><a name="parmname681785019417"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b8300175210408"><a name="b8300175210408"></a><a name="b8300175210408"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname616535816"><a name="varname616535816"></a><a name="varname616535816"></a>capacity</span></i>]之间。</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a>topk</strong>：取值应在[0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname1431611816133"><a name="parmname1431611816133"></a><a name="parmname1431611816133"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### SetNTotal接口<a name="ZH-CN_TOPIC_0000001456854892"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR SetNTotal(int n) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p7313759183119"><a name="p7313759183119"></a><a name="p7313759183119"></a>为外部提供调整<span class="parmname" id="parmname159164443116"><a name="parmname159164443116"></a><a name="parmname159164443116"></a>“ntotal”</span>计数。</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>每次增加底库向量后，Index内部尽管会根据最大插入下标更新<span class="parmname" id="parmname114061192322"><a name="parmname114061192322"></a><a name="parmname114061192322"></a>“ntotal”</span>值，但并没有记录[0, <i><span class="varname" id="varname1917151317325"><a name="varname1917151317325"></a><a name="varname1917151317325"></a>ntotal</span></i>]范围内哪些区域是无效的空间，因此<strong id="b144482818157"><a name="b144482818157"></a><a name="b144482818157"></a>RemoveFeatures</strong>操作没有改变<span class="parmname" id="parmname1274441121512"><a name="parmname1274441121512"></a><a name="parmname1274441121512"></a>“ntotal”</span>的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置<span class="parmname" id="parmname159521818143214"><a name="parmname159521818143214"></a><a name="parmname159521818143214"></a>“ntotal”</span>，这样可以在可控范围内减少算子的计算量，以提高接口性能。</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>例如：当前插入100条向量，底库索引为0~99 时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部<span class="parmname" id="parmname974517165332"><a name="parmname974517165332"></a><a name="parmname974517165332"></a>“ntotal”</span>保持不变，只能设为[<i><span class="varname" id="varname169891216331"><a name="varname169891216331"></a><a name="varname169891216331"></a>ntotal</span></i>, <i><span class="varname" id="varname91661324163313"><a name="varname91661324163313"></a><a name="varname91661324163313"></a>capacity</span></i>]之间的值，再次执行删除索引为90~99的底库，此时可以手动把<span class="parmname" id="parmname18801143812373"><a name="parmname18801143812373"></a><a name="parmname18801143812373"></a>“ntotal”</span>设置为[80, <i><span class="varname" id="varname737175673612"><a name="varname737175673612"></a><a name="varname737175673612"></a>capacity</span></i>]之间的值，设置为<span class="parmvalue" id="parmvalue8748356153711"><a name="parmvalue8748356153711"></a><a name="parmvalue8748356153711"></a>“80”</span>时，可以使参与比对的底库数据量有效减少20条。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1215142783714"><a name="b1215142783714"></a><a name="b1215142783714"></a>int n</strong>：由用户在业务面管理的最大底库的索引加1。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</p>
</td>
</tr>
</tbody>
</table>



### AscendIndexILFlat<a name="ZH-CN_TOPIC_0000002514896041"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000002482656058"></a>

AscendIndexILFlat为ILFlat标准态场景，需要使用Init指定对应资源的初始化，初始化完成之后会申请一段完整空间用于存储底库。在使用完成之后，需要调用Finalize接口对资源进行释放。

AscendIndexILFlat仅支持使用Atlas 推理系列产品，在标准态部署方式下的向量内积距离类型。AscendIndexILFlat在使用时依赖Flat和AICPU算子，具体请参见[Flat](../user_guide.md#flat)和[AICPU](../user_guide.md#aicpu)。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。


#### AddFeatures接口<a name="ZH-CN_TOPIC_0000002514776041"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR AddFeatures(int n, const float *features);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>向特征库追加“n”个特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b645815911297"><a name="b645815911297"></a><a name="b645815911297"></a>int n</strong>：插入特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b1053019911245"><a name="b1053019911245"></a><a name="b1053019911245"></a>const float *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1835741212302"><a name="b1835741212302"></a><a name="b1835741212302"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul10674191294110"></a><a name="ul10674191294110"></a><ul id="ul10674191294110"><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname3816191310019"><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a><a name="zh-cn_topic_0000001628542464_parmname3816191310019"></a>“features”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table392463914228"></a>
<table><tbody><tr id="row17924183911228"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p159241739182219"><a name="p159241739182219"></a><a name="p159241739182219"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19924039112212"><a name="p19924039112212"></a><a name="p19924039112212"></a>APP_ERROR AddFeatures(int n, const float16_t *features);</p>
</td>
</tr>
<tr id="row13924439172216"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p792417393229"><a name="p792417393229"></a><a name="p792417393229"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1792413962211"><a name="p1792413962211"></a><a name="p1792413962211"></a>向特征库追加“n”个特征向量。</p>
</td>
</tr>
<tr id="row792418398229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1692410397224"><a name="p1692410397224"></a><a name="p1692410397224"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1192416399222"><a name="p1192416399222"></a><a name="p1192416399222"></a><strong id="b1992493913229"><a name="b1992493913229"></a><a name="b1992493913229"></a>int n</strong>：插入特征向量数目。</p>
<p id="p292416399222"><a name="p292416399222"></a><a name="p292416399222"></a><strong id="b1453142822318"><a name="b1453142822318"></a><a name="b1453142822318"></a>const float16_t *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row5924163962213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p0924439162217"><a name="p0924439162217"></a><a name="p0924439162217"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p19924103932211"><a name="p19924103932211"></a><a name="p19924103932211"></a>无</p>
</td>
</tr>
<tr id="row14924163932212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p18924143910228"><a name="p18924143910228"></a><a name="p18924143910228"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p179241397228"><a name="p179241397228"></a><a name="p179241397228"></a><strong id="b892483952211"><a name="b892483952211"></a><a name="b892483952211"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row159242391222"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p2924153992215"><a name="p2924153992215"></a><a name="p2924153992215"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul16924339172210"></a><a name="ul16924339172210"></a><ul id="ul16924339172210"><li><strong id="b1592433919225"><a name="b1592433919225"></a><a name="b1592433919225"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname109242395227"><a name="varname109242395227"></a><a name="varname109242395227"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname10924153952217"><a name="parmname10924153952217"></a><a name="parmname10924153952217"></a>“features”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### AscendIndexILFlat接口<a name="ZH-CN_TOPIC_0000002516511133"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendIndexILFlat();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexILFlat的构造函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a>无</p>
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

<a name="table161511529133912"></a>
<table><tbody><tr id="row1615110293394"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2151429113910"><a name="p2151429113910"></a><a name="p2151429113910"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16794134614447"><a name="p16794134614447"></a><a name="p16794134614447"></a>AscendIndexILFlat(const AscendIndexILFlat&amp;) = delete;</p>
</td>
</tr>
<tr id="row51517295398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p21514294391"><a name="p21514294391"></a><a name="p21514294391"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015122918399"><a name="p2015122918399"></a><a name="p2015122918399"></a>声明此Index拷贝函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row815120292398"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7151122933917"><a name="p7151122933917"></a><a name="p7151122933917"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b2450181274519"><a name="b2450181274519"></a><a name="b2450181274519"></a>const AscendIndexILFlat&amp;</strong>：AscendIndexILFlat对象。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexILFlat();</p>
</td>
</tr>
<tr id="row1926221314401"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1926218134408"><a name="p1926218134408"></a><a name="p1926218134408"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82621213184020"><a name="p82621213184020"></a><a name="p82621213184020"></a>AscendIndexILFlat的析构函数。</p>
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


#### ComputeDistance接口<a name="ZH-CN_TOPIC_0000002482736032"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19771939144811"><a name="p19771939144811"></a><a name="p19771939144811"></a>APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname11281111175619"><a name="parmname11281111175619"></a><a name="parmname11281111175619"></a>“n”</span>条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出经过映射后的距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b75131329183314"><a name="b75131329183314"></a><a name="b75131329183314"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b6862131183312"><a name="b6862131183312"></a><a name="b6862131183312"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b1041123433313"><a name="b1041123433313"></a><a name="b1041123433313"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue354895917522"><a name="parmvalue354895917522"></a><a name="parmvalue354895917522"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b376183643316"><a name="b376183643316"></a><a name="b376183643316"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname834710403338"><a name="parmname834710403338"></a><a name="parmname834710403338"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue16873151054312"><a name="parmvalue16873151054312"></a><a name="parmvalue16873151054312"></a>“48”</span>，即<span class="parmname" id="parmname135806266430"><a name="parmname135806266430"></a><a name="parmname135806266430"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1750033215518"><a name="p1750033215518"></a><a name="p1750033215518"></a><strong id="b668194911337"><a name="b668194911337"></a><a name="b668194911337"></a>float *distances</strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（<span class="parmname" id="parmname10121121717561"><a name="parmname10121121717561"></a><a name="parmname10121121717561"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname13631434155615"><a name="varname13631434155615"></a><a name="varname13631434155615"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname0810103815562"><a name="parmname0810103815562"></a><a name="parmname0810103815562"></a>“ntotal”</span>对16补齐）。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b167221751163312"><a name="b167221751163312"></a><a name="b167221751163312"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul167968714447"></a><a name="ul167968714447"></a><ul id="ul167968714447"><li><strong id="b1141410121444"><a name="b1141410121444"></a><a name="b1141410121444"></a>n</strong>：合理的n值应该在(0, <i><span class="varname" id="varname1835520162442"><a name="varname1835520162442"></a><a name="varname1835520162442"></a>capacity</span></i>]之间。</li><li><strong id="b429253634917"><a name="b429253634917"></a><a name="b429253634917"></a>distances</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname1598134614491"><a name="parmname1598134614491"></a><a name="parmname1598134614491"></a>“ntotalPad”</span>为(<i><span class="varname" id="varname1654664914915"><a name="varname1654664914915"></a><a name="varname1654664914915"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname145712539496"><a name="parmname145712539496"></a><a name="parmname145712539496"></a>“ntotal”</span>对16补齐的结果，每个query的有效比对距离存储在前<span class="parmname" id="parmname15322121501"><a name="parmname15322121501"></a><a name="parmname15322121501"></a>“ntotal”</span>的空间，补齐部分数据没有实际意义）。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table17574555124816"></a>
<table><tbody><tr id="row757435594819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p155742557480"><a name="p155742557480"></a><a name="p155742557480"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p0574255104813"><a name="p0574255104813"></a><a name="p0574255104813"></a>APP_ERROR ComputeDistance(int n, const float *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row14574135514811"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p205741955124812"><a name="p205741955124812"></a><a name="p205741955124812"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p45741955174814"><a name="p45741955174814"></a><a name="p45741955174814"></a>查询<span class="parmname" id="parmname18574205574813"><a name="parmname18574205574813"></a><a name="parmname18574205574813"></a>“n”</span>条特征向量与底库所有特征向量的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出经过映射后的距离。</p>
</td>
</tr>
<tr id="row85751555194813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p35754551488"><a name="p35754551488"></a><a name="p35754551488"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1757535519484"><a name="p1757535519484"></a><a name="p1757535519484"></a><strong id="b257555544811"><a name="b257555544811"></a><a name="b257555544811"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p1575115520489"><a name="p1575115520489"></a><a name="p1575115520489"></a><strong id="b158293141686"><a name="b158293141686"></a><a name="b158293141686"></a>const float *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p1957512553485"><a name="p1957512553485"></a><a name="p1957512553485"></a><strong id="b7575155164810"><a name="b7575155164810"></a><a name="b7575155164810"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue15575255144814"><a name="parmvalue15575255144814"></a><a name="parmvalue15575255144814"></a>“10000”</span>。</p>
<p id="p20575185520489"><a name="p20575185520489"></a><a name="p20575185520489"></a><strong id="b1657585510484"><a name="b1657585510484"></a><a name="b1657585510484"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname257575517485"><a name="parmname257575517485"></a><a name="parmname257575517485"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue19575115554814"><a name="parmvalue19575115554814"></a><a name="parmvalue19575115554814"></a>“48”</span>，即<span class="parmname" id="parmname55751655194811"><a name="parmname55751655194811"></a><a name="parmname55751655194811"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row1557595510487"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1257545518481"><a name="p1257545518481"></a><a name="p1257545518481"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p115751555184810"><a name="p115751555184810"></a><a name="p115751555184810"></a><strong id="b13575195584812"><a name="b13575195584812"></a><a name="b13575195584812"></a>float *distances</strong>：外部内存，存储查询向量与底库向量的距离，总长度应该为n * nTotalPad（<span class="parmname" id="parmname1557575514485"><a name="parmname1557575514485"></a><a name="parmname1557575514485"></a>“ntotalPad”</span>为 (<i><span class="varname" id="varname15575125515488"><a name="varname15575125515488"></a><a name="varname15575125515488"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname657555514488"><a name="parmname657555514488"></a><a name="parmname657555514488"></a>“ntotal”</span>对16补齐）。</p>
</td>
</tr>
<tr id="row7575175554817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1757575514814"><a name="p1757575514814"></a><a name="p1757575514814"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p125758557485"><a name="p125758557485"></a><a name="p125758557485"></a><strong id="b19575165519489"><a name="b19575165519489"></a><a name="b19575165519489"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row9575755204810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12575185574818"><a name="p12575185574818"></a><a name="p12575185574818"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7575125514817"></a><a name="ul7575125514817"></a><ul id="ul7575125514817"><li><strong id="b18575155510484"><a name="b18575155510484"></a><a name="b18575155510484"></a>n</strong>：合理的n值应该在(0, <i><span class="varname" id="varname1557595554819"><a name="varname1557595554819"></a><a name="varname1557595554819"></a>capacity</span></i>]之间。</li><li><strong id="b75751155114811"><a name="b75751155114811"></a><a name="b75751155114811"></a>distances</strong>：需要提供的空间长度为n * ntotalPad（<span class="parmname" id="parmname857575524810"><a name="parmname857575524810"></a><a name="parmname857575524810"></a>“ntotalPad”</span>为(<i><span class="varname" id="varname11575355194813"><a name="varname11575355194813"></a><a name="varname11575355194813"></a>ntotal </span></i>+ 15) / 16 * 16，即<span class="parmname" id="parmname3575185514489"><a name="parmname3575185514489"></a><a name="parmname3575185514489"></a>“ntotal”</span>对16补齐的结果，每个query的有效比对距离存储在前<span class="parmname" id="parmname12575175514482"><a name="parmname12575175514482"></a><a name="parmname12575175514482"></a>“ntotal”</span>的空间，补齐部分数据没有实际意义）。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li><li><span class="parmname" id="parmname457605554818"><a name="parmname457605554818"></a><a name="parmname457605554818"></a>“queries”</span>和<span class="parmname" id="parmname185761655104810"><a name="parmname185761655104810"></a><a name="parmname185761655104810"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### ComputeDistanceByIdx接口<a name="ZH-CN_TOPIC_0000002514896043"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1196718166412"><a name="p1196718166412"></a><a name="p1196718166412"></a>APP_ERROR ComputeDistanceByIdx(int n, const float *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>ComputeDistance计算待查询向量与所有底库向量的距离，而ComputeDistanceByIdx接口只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1178514265435"><a name="b1178514265435"></a><a name="b1178514265435"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b811819197314"><a name="b811819197314"></a><a name="b811819197314"></a>const float *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname1441759144217"><a name="parmname1441759144217"></a><a name="parmname1441759144217"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1572252111218"><a name="p1572252111218"></a><a name="p1572252111218"></a><strong id="b277683013439"><a name="b277683013439"></a><a name="b277683013439"></a>const int *num</strong>： 给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p6193853112116"><a name="p6193853112116"></a><a name="p6193853112116"></a><strong id="b79815523517"><a name="b79815523517"></a><a name="b79815523517"></a>const idx_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname2711154912437"><a name="parmname2711154912437"></a><a name="parmname2711154912437"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname742124364316"><a name="parmname742124364316"></a><a name="parmname742124364316"></a>“indices”</span>长度为n * max(num)。</p>
<p id="p13553919567"><a name="p13553919567"></a><a name="p13553919567"></a><strong id="b184841525865"><a name="b184841525865"></a><a name="b184841525865"></a>MEMORY_TYPE memoryType</strong>：输入输出存放位置策略，默认为MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，可选策略如下：</p>
<a name="ul125365550127"></a><a name="ul125365550127"></a><ul id="ul125365550127"><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST：输入在host，输出在host。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE：输入在device，输出在device。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST：输入在device，输出在host。</li><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE：输入在host，输出在device。</li></ul>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b156251035184313"><a name="b156251035184313"></a><a name="b156251035184313"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b25863377438"><a name="b25863377438"></a><a name="b25863377438"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname136035014443"><a name="parmname136035014443"></a><a name="parmname136035014443"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue2069717180212"><a name="parmvalue2069717180212"></a><a name="parmvalue2069717180212"></a>“48”</span>，即<span class="parmname" id="parmname1997224217"><a name="parmname1997224217"></a><a name="parmname1997224217"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p057182814222"><a name="p057182814222"></a><a name="p057182814222"></a><strong id="b0194557446"><a name="b0194557446"></a><a name="b0194557446"></a>float *distances</strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大<span class="parmname" id="parmname658971354417"><a name="parmname658971354417"></a><a name="parmname658971354417"></a>“num”</span>补齐空间占用，空间长度为n * max(num)。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b140412864414"><a name="b140412864414"></a><a name="b140412864414"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1639103913216"></a><a name="ul1639103913216"></a><ul id="ul1639103913216"><li><strong id="b4983164118215"><a name="b4983164118215"></a><a name="b4983164118215"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname82723561324"><a name="varname82723561324"></a><a name="varname82723561324"></a>capacity</span></i>]之间。</li><li><strong id="b434182710436"><a name="b434182710436"></a><a name="b434182710436"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0, ntotal]之间。</li><li><strong id="b1221646828"><a name="b1221646828"></a><a name="b1221646828"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname7520558520"><a name="varname7520558520"></a><a name="varname7520558520"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname343119418149"><a name="parmname343119418149"></a><a name="parmname343119418149"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li><li>选择memoryType存放策略时，<span class="parmname" id="parmname17419164171417"><a name="parmname17419164171417"></a><a name="parmname17419164171417"></a>“queries”</span>、<span class="parmname" id="parmname13419174191415"><a name="parmname13419174191415"></a><a name="parmname13419174191415"></a>“distances”</span>需要为对应位置指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table93703718308"></a>
<table><tbody><tr id="row20370173302"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p143716720307"><a name="p143716720307"></a><a name="p143716720307"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1037115763020"><a name="p1037115763020"></a><a name="p1037115763020"></a>APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row103719723013"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p0371207153017"><a name="p0371207153017"></a><a name="p0371207153017"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p53715711306"><a name="p53715711306"></a><a name="p53715711306"></a>ComputeDistance计算待查询向量与所有底库向量的距离，而ComputeDistanceByIdx接口只计算待查询向量与给定下标索引的底库向量之间的距离。如传递有效的映射表（tableLen &gt; 0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row123716710302"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p16371147193013"><a name="p16371147193013"></a><a name="p16371147193013"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p23711377308"><a name="p23711377308"></a><a name="p23711377308"></a><strong id="b83711372301"><a name="b83711372301"></a><a name="b83711372301"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p173714714309"><a name="p173714714309"></a><a name="p173714714309"></a><strong id="b1926533153016"><a name="b1926533153016"></a><a name="b1926533153016"></a>const float16_t *queries</strong>：待查询特征向量，有效长度为n * dim，<span class="parmname" id="parmname8371207113014"><a name="parmname8371207113014"></a><a name="parmname8371207113014"></a>“dim”</span>需与初始化时指定的dim保持一致。</p>
<p id="p1237110743020"><a name="p1237110743020"></a><a name="p1237110743020"></a><strong id="b133711723012"><a name="b133711723012"></a><a name="b133711723012"></a>const int *num</strong>： 给定每个query要比对的底库特征向量数目，长度为n。</p>
<p id="p037167153019"><a name="p037167153019"></a><a name="p037167153019"></a><strong id="b12371167113014"><a name="b12371167113014"></a><a name="b12371167113014"></a>const idx_t *indices</strong>：给定要比对的底库特征向量下标索引，每个query要比对的底库向量个数可以不同，应从前往后连续存储有效的向量索引，按照最大<span class="parmname" id="parmname33718710304"><a name="parmname33718710304"></a><a name="parmname33718710304"></a>“num”</span>补齐空间占用，<span class="parmname" id="parmname113713793017"><a name="parmname113713793017"></a><a name="parmname113713793017"></a>“indices”</span>长度为n * max(num)。</p>
<p id="p123717711303"><a name="p123717711303"></a><a name="p123717711303"></a><strong id="b1137116717301"><a name="b1137116717301"></a><a name="b1137116717301"></a>MEMORY_TYPE memoryType</strong>：输入输出存放位置策略，默认为MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST，可选策略如下：</p>
<a name="ul183711373302"></a><a name="ul183711373302"></a><ul id="ul183711373302"><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST：输入在host，输出在host。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE：输入在device，输出在device。</li><li>MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST：输入在device，输出在host。</li><li>MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE：输入在host，输出在device。</li></ul>
<p id="p173715717301"><a name="p173715717301"></a><a name="p173715717301"></a><strong id="b18371874303"><a name="b18371874303"></a><a name="b18371874303"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue2371137173014"><a name="parmvalue2371137173014"></a><a name="parmvalue2371137173014"></a>“10000”</span>。</p>
<p id="p1637117793019"><a name="p1637117793019"></a><a name="p1637117793019"></a><strong id="b153711719302"><a name="b153711719302"></a><a name="b153711719302"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname123719713013"><a name="parmname123719713013"></a><a name="parmname123719713013"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue637117153014"><a name="parmvalue637117153014"></a><a name="parmvalue637117153014"></a>“48”</span>，即<span class="parmname" id="parmname193714743018"><a name="parmname193714743018"></a><a name="parmname193714743018"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row837117183012"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p1737119715302"><a name="p1737119715302"></a><a name="p1737119715302"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p4371378300"><a name="p4371378300"></a><a name="p4371378300"></a><strong id="b137117716306"><a name="b137117716306"></a><a name="b137117716306"></a>float *distances</strong>：查询向量与选定底库向量的距离，每个query从前往后连续记录有效距离，按照最大<span class="parmname" id="parmname73715718306"><a name="parmname73715718306"></a><a name="parmname73715718306"></a>“num”</span>补齐空间占用，空间长度为n * max(num)。</p>
</td>
</tr>
<tr id="row037177153010"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p183711274309"><a name="p183711274309"></a><a name="p183711274309"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p63711776304"><a name="p63711776304"></a><a name="p63711776304"></a><strong id="b1337118763012"><a name="b1337118763012"></a><a name="b1337118763012"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row193711676307"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p137114753010"><a name="p137114753010"></a><a name="p137114753010"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul13371476304"></a><a name="ul13371476304"></a><ul id="ul13371476304"><li><strong id="b737127163015"><a name="b737127163015"></a><a name="b737127163015"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname737137183015"><a name="varname737137183015"></a><a name="varname737137183015"></a>capacity</span></i>]之间。</li><li><strong id="b103719712303"><a name="b103719712303"></a><a name="b103719712303"></a>num</strong>：由用户指定，长度为n，每个query的num值应该在[0, ntotal]之间。</li><li><strong id="b33718723018"><a name="b33718723018"></a><a name="b33718723018"></a>indices</strong>：每个特征的索引应该在[0, <i><span class="varname" id="varname6371147113013"><a name="varname6371147113013"></a><a name="varname6371147113013"></a>ntotal</span></i>)之间。</li><li>接口参数配置举例：n = 3, num[3] = {1, 3, 5}，表示3个query分别要比对的底库向量个数，max(num) = 5，则 *indices指向空间长度按照5对齐，总大小为3 * 5 * sizeof(idx_t) Byte，如{ {1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9} }。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul10372137123013"></a><a name="ul10372137123013"></a><ul id="ul10372137123013"><li><span class="parmname" id="parmname17372776306"><a name="parmname17372776306"></a><a name="parmname17372776306"></a>“indices”</span>、<span class="parmname" id="parmname437213773019"><a name="parmname437213773019"></a><a name="parmname437213773019"></a>“queries”</span>、<span class="parmname" id="parmname2037220753020"><a name="parmname2037220753020"></a><a name="parmname2037220753020"></a>“distances”</span>和<span class="parmname" id="parmname33721178301"><a name="parmname33721178301"></a><a name="parmname33721178301"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### Finalize接口<a name="ZH-CN_TOPIC_0000002482656060"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>void Finalize();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>释放特征库管理资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1570793612442"><a name="b1570793612442"></a><a name="b1570793612442"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### GetFeatures接口<a name="ZH-CN_TOPIC_0000002484074790"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeatures(int n, float *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询<span class="parmname" id="parmname9635334135520"><a name="parmname9635334135520"></a><a name="parmname9635334135520"></a>“n”</span>条指定下标索引的特征向量。输出在host。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b38573713437"><a name="b38573713437"></a><a name="b38573713437"></a>float *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table018415716495"></a>
<table><tbody><tr id="row51841657124915"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p3184757144914"><a name="p3184757144914"></a><a name="p3184757144914"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p16184957174915"><a name="p16184957174915"></a><a name="p16184957174915"></a>APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row121844578499"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1184205717496"><a name="p1184205717496"></a><a name="p1184205717496"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6184457154911"><a name="p6184457154911"></a><a name="p6184457154911"></a>查询<span class="parmname" id="parmname518485711495"><a name="parmname518485711495"></a><a name="parmname518485711495"></a>“n”</span>条指定下标索引的特征向量。输出在host。</p>
</td>
</tr>
<tr id="row20184257154913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18184657114919"><a name="p18184657114919"></a><a name="p18184657114919"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1418413578493"><a name="p1418413578493"></a><a name="p1418413578493"></a><strong id="b16184115711494"><a name="b16184115711494"></a><a name="b16184115711494"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p11184757204918"><a name="p11184757204918"></a><a name="p11184757204918"></a><strong id="b718445714916"><a name="b718445714916"></a><a name="b718445714916"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row19184195714498"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p3184175711496"><a name="p3184175711496"></a><a name="p3184175711496"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p12184657154920"><a name="p12184657154920"></a><a name="p12184657154920"></a><strong id="b12804115914544"><a name="b12804115914544"></a><a name="b12804115914544"></a>float16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。</p>
</td>
</tr>
<tr id="row1918411573494"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p19184357194919"><a name="p19184357194919"></a><a name="p19184357194919"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p14184757184914"><a name="p14184757184914"></a><a name="p14184757184914"></a><strong id="b191841957184915"><a name="b191841957184915"></a><a name="b191841957184915"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row16184195774920"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p181841657104916"><a name="p181841657104916"></a><a name="p181841657104916"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul181842571494"></a><a name="ul181842571494"></a><ul id="ul181842571494"><li><strong id="b6184105764918"><a name="b6184105764918"></a><a name="b6184105764918"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname618445754917"><a name="varname618445754917"></a><a name="varname618445754917"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b111847574493"><a name="b111847574493"></a><a name="b111847574493"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname1918410571491"><a name="varname1918410571491"></a><a name="varname1918410571491"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname2018413570492"><a name="parmname2018413570492"></a><a name="parmname2018413570492"></a>“features”</span>和<span class="parmname" id="parmname818495794917"><a name="parmname818495794917"></a><a name="parmname818495794917"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### GetFeaturesOnDevice<a name="ZH-CN_TOPIC_0000002516516843"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR GetFeaturesOnDevice (int n, float16_t *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询“n”条指定下标索引的特征向量。输出在Device。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1185433593117"><a name="b1185433593117"></a><a name="b1185433593117"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b885718291130"><a name="b885718291130"></a><a name="b885718291130"></a>float16_t *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。Device侧指针。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table15312115612410"></a>
<table><tbody><tr id="row1831211561843"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2031217561042"><a name="p2031217561042"></a><a name="p2031217561042"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10312185617418"><a name="p10312185617418"></a><a name="p10312185617418"></a>APP_ERROR GetFeaturesOnDevice (int n, float *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row123121356046"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p531245612416"><a name="p531245612416"></a><a name="p531245612416"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p431217567418"><a name="p431217567418"></a><a name="p431217567418"></a>查询“n”条指定下标索引的特征向量。输出在Device。</p>
</td>
</tr>
<tr id="row531213561245"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p431225615416"><a name="p431225615416"></a><a name="p431225615416"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p63123561342"><a name="p63123561342"></a><a name="p63123561342"></a><strong id="b13312356848"><a name="b13312356848"></a><a name="b13312356848"></a>int n</strong>：获取底库向量的个数。</p>
<p id="p1731215620414"><a name="p1731215620414"></a><a name="p1731215620414"></a><strong id="b143129561748"><a name="b143129561748"></a><a name="b143129561748"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row53126562043"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1431212564417"><a name="p1431212564417"></a><a name="p1431212564417"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p193121656244"><a name="p193121656244"></a><a name="p193121656244"></a><strong id="b1679419391057"><a name="b1679419391057"></a><a name="b1679419391057"></a>float *features</strong>：查询下标索引对应的特征向量长度为n * 向量维度dim。Device侧指针。</p>
</td>
</tr>
<tr id="row10312056346"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p73124561444"><a name="p73124561444"></a><a name="p73124561444"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p931217561412"><a name="p931217561412"></a><a name="p931217561412"></a><strong id="b1931275616410"><a name="b1931275616410"></a><a name="b1931275616410"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row123127565418"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1931214561141"><a name="p1931214561141"></a><a name="p1931214561141"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4312115612410"></a><a name="ul4312115612410"></a><ul id="ul4312115612410"><li><strong id="b331255610416"><a name="b331255610416"></a><a name="b331255610416"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname1631245618414"><a name="varname1631245618414"></a><a name="varname1631245618414"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b2312185619410"><a name="b2312185619410"></a><a name="b2312185619410"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname1831295612411"><a name="varname1831295612411"></a><a name="varname1831295612411"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname83125560419"><a name="parmname83125560419"></a><a name="parmname83125560419"></a>“features”</span>和<span class="parmname" id="parmname18312125620410"><a name="parmname18312125620410"></a><a name="parmname18312125620410"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### GetNTotal接口<a name="ZH-CN_TOPIC_0000002514776043"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p0336129171210"><a name="p0336129171210"></a><a name="p0336129171210"></a>int GetNTotal() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询当前特征库特征向量数目的理论最大值。如果插入特征向量indices连续，则ntotal等于特征向量数目。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a><strong id="b445021732816"><a name="b445021732816"></a><a name="b445021732816"></a>int ntotal</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a>int</strong>：特征向量数目的理论最大值（底库向量最大索引加1）。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### Init接口<a name="ZH-CN_TOPIC_0000002482736034"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector&lt;int&gt; &amp;deviceList, int64_t resourceSize = -1);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>AscendIndexILFlat的初始化函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1517311219268"><a name="b1517311219268"></a><a name="b1517311219268"></a>int dim</strong>：AscendIndexILFlat管理的特征向量的维度。</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8628752620"><a name="b8628752620"></a><a name="b8628752620"></a>int capacity</strong>：底库最大容量，接口会根据<span class="parmname" id="parmname16513113011414"><a name="parmname16513113011414"></a><a name="parmname16513113011414"></a>“capacity”</span>值申请capacity * dim * sizeof(fp16) 字节内存数据。</p>
<p id="p1450765311416"><a name="p1450765311416"></a><a name="p1450765311416"></a><strong id="b5404134231712"><a name="b5404134231712"></a><a name="b5404134231712"></a>faiss::MetricType metricType</strong>： 特征距离类别（向量内积、欧氏距离、余弦相似度）。</p>
<p id="p1291682015184"><a name="p1291682015184"></a><a name="p1291682015184"></a><strong id="b6916192019187"><a name="b6916192019187"></a><a name="b6916192019187"></a>const std::vector&lt;int&gt; &amp;deviceList</strong>：Device侧资源配置。</p>
<p id="p1411722401512"><a name="p1411722401512"></a><a name="p1411722401512"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a>int64_t resourceSize</strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为“-1”，表示设置为“128MB”。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b13793144414268"><a name="b13793144414268"></a><a name="b13793144414268"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1768605017262"></a><a name="ul1768605017262"></a><ul id="ul1768605017262"><li>dim ∈ {32, 64, 128, 256, 384, 512}</li><li>metricType：AscendIndexILFlat目前只实现了向量内积距离，即只支持“faiss::MetricType::METRIC_INNER_PRODUCT”。</li><li>capacity：接口允许为底库申请的内存上限设为12,288,000,000Byte，同时“capacity”的值域约束为[0, 12000000]。<a name="ul138816512117"></a><a name="ul138816512117"></a><ul id="ul138816512117"><li>以512维、FP16类型的底库向量为例，最大支持的<span class="parmname" id="parmname1593195143016"><a name="parmname1593195143016"></a><a name="parmname1593195143016"></a>“capacity”</span>为1200万(12288000000 / (512 * sizeof(fp_16)) )。</li><li>对于256维、FP16类型的底库向量，尽管内存约束支持更大的<span class="parmname" id="parmname15389183615368"><a name="parmname15389183615368"></a><a name="parmname15389183615368"></a>“capacity”</span>，<span class="parmname" id="parmname169642468367"><a name="parmname169642468367"></a><a name="parmname169642468367"></a>“capacity”</span>最大也只能设为1200万。</li></ul>
</li><li>仅支持配置单卡，暂不支持配置多卡，需满足<strong id="b16270210371"><a name="b16270210371"></a><a name="b16270210371"></a>deviceList.size() == 1</strong>。</li><li>resourceSize：可以配置为-1或[134217728，4294967296]之间的值，相当于[128MB，4096MB]。该参数通过底库大小和search的batch数共同确定，在底库大于等于1000万且batch数大于等于16时建议设置为“1024MB”。</li></ul>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000002482794858"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4759192852812"><a name="p4759192852812"></a><a name="p4759192852812"></a>AscendIndexILFlat&amp; operator=(const AscendIndexILFlat &amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5787143522812"><a name="p5787143522812"></a><a name="p5787143522812"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p175601140172811"><a name="p175601140172811"></a><a name="p175601140172811"></a><strong id="b1023216252912"><a name="b1023216252912"></a><a name="b1023216252912"></a>const AscendIndexILFlat &amp;</strong>：AscendIndexILFlat对象。</p>
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


#### RemoveFeatures接口<a name="ZH-CN_TOPIC_0000002482917750"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p411713313214"><a name="p411713313214"></a><a name="p411713313214"></a>APP_ERROR RemoveFeatures(int n, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>删除向量库中<span class="parmname" id="parmname15331161155517"><a name="parmname15331161155517"></a><a name="parmname15331161155517"></a>“n”</span>个指定下标索引的特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b10676733203011"><a name="b10676733203011"></a><a name="b10676733203011"></a>int n</strong>：删除特征向量数目。</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b1248654013016"><a name="b1248654013016"></a><a name="b1248654013016"></a>const idx_t *indices</strong>：特征向量对应的下标索引，长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b727535618302"><a name="b727535618302"></a><a name="b727535618302"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85131151122111"><a name="varname85131151122111"></a><a name="varname85131151122111"></a>ntotal</span></i>)之间，ntotal可以通过GetNTotal接口获取。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### Search接口<a name="ZH-CN_TOPIC_0000002514896045"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p95721657104813"><a name="p95721657104813"></a><a name="p95721657104813"></a>APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>查询与query向量距离最近的<span class="parmname" id="parmname66101618135916"><a name="parmname66101618135916"></a><a name="parmname66101618135916"></a>“topk”</span>个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出映射后的距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b2797917153611"><a name="b2797917153611"></a><a name="b2797917153611"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1445642010368"><a name="b1445642010368"></a><a name="b1445642010368"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1177612213614"><a name="b1177612213614"></a><a name="b1177612213614"></a>int topk</strong>：查询向量和底库的比对距离进行排序，返回<span class="parmname" id="parmname19303182555918"><a name="parmname19303182555918"></a><a name="parmname19303182555918"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b18837162415364"><a name="b18837162415364"></a><a name="b18837162415364"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b1174653015367"><a name="b1174653015367"></a><a name="b1174653015367"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname173331533163614"><a name="parmname173331533163614"></a><a name="parmname173331533163614"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue1525151355319"><a name="parmvalue1525151355319"></a><a name="parmvalue1525151355319"></a>“48”</span>，即<span class="parmname" id="parmname78791212533"><a name="parmname78791212533"></a><a name="parmname78791212533"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b187841356153711"><a name="b187841356153711"></a><a name="b187841356153711"></a>float *distances</strong>：外部内存，与query相似度最高的<strong id="b8665314543"><a name="b8665314543"></a><a name="b8665314543"></a>topk </strong>* <strong id="b1408103115411"><a name="b1408103115411"></a><a name="b1408103115411"></a>n</strong>个底库特征向量所对应的余弦距离，长度为n * topk。</p>
<p id="p1154614405264"><a name="p1154614405264"></a><a name="p1154614405264"></a><strong id="b12933115811373"><a name="b12933115811373"></a><a name="b12933115811373"></a>idx_t *indices</strong>：外部内存，返回与query相似度最高的<span class="parmname" id="parmname1346104111594"><a name="parmname1346104111594"></a><a name="parmname1346104111594"></a>“topk”</span>个底库向量对应的下标索引，长度为n * topk。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b111527319385"><a name="b111527319385"></a><a name="b111527319385"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1346615102548"></a><a name="ul1346615102548"></a><ul id="ul1346615102548"><li><strong id="b5538111715545"><a name="b5538111715545"></a><a name="b5538111715545"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname17384527155412"><a name="varname17384527155412"></a><a name="varname17384527155412"></a>capacity</span></i>]之间。</li><li><strong id="b18681518105411"><a name="b18681518105411"></a><a name="b18681518105411"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>和<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table838713119461"></a>
<table><tbody><tr id="row33871117462"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p83873114461"><a name="p83873114461"></a><a name="p83873114461"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p33878119466"><a name="p33878119466"></a><a name="p33878119466"></a>APP_ERROR Search(int n, const float *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row4388161184611"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p143889110469"><a name="p143889110469"></a><a name="p143889110469"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1388121164616"><a name="p1388121164616"></a><a name="p1388121164616"></a>查询与query向量距离最近的<span class="parmname" id="parmname138821154613"><a name="parmname138821154613"></a><a name="parmname138821154613"></a>“topk”</span>个底库下标索引和对应的距离，如传递有效的映射表（tableLen &gt; 0且table为非空指针），则输出映射后的距离。</p>
</td>
</tr>
<tr id="row1038821104617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p9388121124613"><a name="p9388121124613"></a><a name="p9388121124613"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p113881518469"><a name="p113881518469"></a><a name="p113881518469"></a><strong id="b123881015468"><a name="b123881015468"></a><a name="b123881015468"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p6388111204615"><a name="p6388111204615"></a><a name="p6388111204615"></a><strong id="b4285489460"><a name="b4285489460"></a><a name="b4285489460"></a>const float *queries</strong>：待查询特征向量，长度为n * 向量维度dim。</p>
<p id="p1738816112464"><a name="p1738816112464"></a><a name="p1738816112464"></a><strong id="b838811114614"><a name="b838811114614"></a><a name="b838811114614"></a>int topk</strong>：查询向量和底库的比对距离进行排序，返回<span class="parmname" id="parmname173881511463"><a name="parmname173881511463"></a><a name="parmname173881511463"></a>“topk”</span>条结果。</p>
<p id="p1338831154612"><a name="p1338831154612"></a><a name="p1338831154612"></a><strong id="b1438814115461"><a name="b1438814115461"></a><a name="b1438814115461"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1238861104619"><a name="parmvalue1238861104619"></a><a name="parmvalue1238861104619"></a>“10000”</span>。</p>
<p id="p183888134612"><a name="p183888134612"></a><a name="p183888134612"></a><strong id="b438816119468"><a name="b438816119468"></a><a name="b438816119468"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname133881219461"><a name="parmname133881219461"></a><a name="parmname133881219461"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue83881018461"><a name="parmvalue83881018461"></a><a name="parmvalue83881018461"></a>“48”</span>，即<span class="parmname" id="parmname1938812174616"><a name="parmname1938812174616"></a><a name="parmname1938812174616"></a>“table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row1938815124610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p13388151134618"><a name="p13388151134618"></a><a name="p13388151134618"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p538810111463"><a name="p538810111463"></a><a name="p538810111463"></a><strong id="b1338861114611"><a name="b1338861114611"></a><a name="b1338861114611"></a>float *distances</strong>：外部内存，与query相似度最高的<strong id="b10388171164619"><a name="b10388171164619"></a><a name="b10388171164619"></a>topk </strong>* <strong id="b183884194620"><a name="b183884194620"></a><a name="b183884194620"></a>n</strong>个底库特征向量所对应的余弦距离，长度为n * topk。</p>
<p id="p1638841124616"><a name="p1638841124616"></a><a name="p1638841124616"></a><strong id="b9388191154616"><a name="b9388191154616"></a><a name="b9388191154616"></a>idx_t *indices</strong>：外部内存，返回与query相似度最高的<span class="parmname" id="parmname18388131164610"><a name="parmname18388131164610"></a><a name="parmname18388131164610"></a>“topk”</span>个底库向量对应的下标索引，长度为n * topk。</p>
</td>
</tr>
<tr id="row1938811154610"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1838821174615"><a name="p1838821174615"></a><a name="p1838821174615"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p338814194614"><a name="p338814194614"></a><a name="p338814194614"></a><strong id="b1138813119469"><a name="b1138813119469"></a><a name="b1138813119469"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row8388618462"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1338815119460"><a name="p1338815119460"></a><a name="p1338815119460"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1138817154617"></a><a name="ul1138817154617"></a><ul id="ul1138817154617"><li><strong id="b3388181144610"><a name="b3388181144610"></a><a name="b3388181144610"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname838861194611"><a name="varname838861194611"></a><a name="varname838861194611"></a>capacity</span></i>]之间。</li><li><strong id="b13881714467"><a name="b13881714467"></a><a name="b13881714467"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul193898112469"></a><a name="ul193898112469"></a><ul id="ul193898112469"><li><span class="parmname" id="parmname338981164615"><a name="parmname338981164615"></a><a name="parmname338981164615"></a>“indices”</span>、<span class="parmname" id="parmname1938916174613"><a name="parmname1938916174613"></a><a name="parmname1938916174613"></a>“queries”</span>和<span class="parmname" id="parmname143891517460"><a name="parmname143891517460"></a><a name="parmname143891517460"></a>“distances”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### SearchByThreshold接口<a name="ZH-CN_TOPIC_0000002482656062"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p111481632134920"><a name="p111481632134920"></a><a name="p111481632134920"></a>APP_ERROR SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1893114655310"><a name="p1893114655310"></a><a name="p1893114655310"></a>在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p147472612548"><a name="p147472612548"></a><a name="p147472612548"></a><strong id="b1378124514396"><a name="b1378124514396"></a><a name="b1378124514396"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p92611547558"><a name="p92611547558"></a><a name="p92611547558"></a><strong id="b1840232010507"><a name="b1840232010507"></a><a name="b1840232010507"></a>const float *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p12923104514555"><a name="p12923104514555"></a><a name="p12923104514555"></a><strong id="b8381185319394"><a name="b8381185319394"></a><a name="b8381185319394"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname166164371714"><a name="parmname166164371714"></a><a name="parmname166164371714"></a>“threshold”</span>进行过滤。</p>
<p id="p660225151520"><a name="p660225151520"></a><a name="p660225151520"></a><strong id="b1245113552396"><a name="b1245113552396"></a><a name="b1245113552396"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1578817211311"><a name="parmname1578817211311"></a><a name="parmname1578817211311"></a>“topk”</span>条结果。</p>
<p id="p661173085819"><a name="p661173085819"></a><a name="p661173085819"></a><strong id="b128914571396"><a name="b128914571396"></a><a name="b128914571396"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue1760985512524"><a name="parmvalue1760985512524"></a><a name="parmvalue1760985512524"></a>“10000”</span>。</p>
<p id="p6149183495812"><a name="p6149183495812"></a><a name="p6149183495812"></a><strong id="b12391120164017"><a name="b12391120164017"></a><a name="b12391120164017"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname279351319407"><a name="parmname279351319407"></a><a name="parmname279351319407"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue376417339011"><a name="parmvalue376417339011"></a><a name="parmvalue376417339011"></a>“48”</span>，即<span class="parmname" id="parmname19896039903"><a name="parmname19896039903"></a><a name="parmname19896039903"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1664124925012"><a name="p1664124925012"></a><a name="p1664124925012"></a><strong id="b662915439408"><a name="b662915439408"></a><a name="b662915439408"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p3960124912518"><a name="p3960124912518"></a><a name="p3960124912518"></a><strong id="b15701310524"><a name="b15701310524"></a><a name="b15701310524"></a>idx_t * indices</strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname1633953516114"><a name="parmname1633953516114"></a><a name="parmname1633953516114"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname1829610421613"><a name="parmname1829610421613"></a><a name="parmname1829610421613"></a>“indices”</span>总长度为n * topk。</p>
<p id="p03841120175217"><a name="p03841120175217"></a><a name="p03841120175217"></a><strong id="b1581125094017"><a name="b1581125094017"></a><a name="b1581125094017"></a>float *distances</strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与<span class="parmname" id="parmname681785019417"><a name="parmname681785019417"></a><a name="parmname681785019417"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b8300175210408"><a name="b8300175210408"></a><a name="b8300175210408"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul54051553506"></a><a name="ul54051553506"></a><ul id="ul54051553506"><li><strong id="b1441635511013"><a name="b1441635511013"></a><a name="b1441635511013"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname616535816"><a name="varname616535816"></a><a name="varname616535816"></a>capacity</span></i>]之间。</li><li><strong id="b15675195717016"><a name="b15675195717016"></a><a name="b15675195717016"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612"><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619"><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211"><a name="zh-cn_topic_0000001456535116_b5371616181211"></a><a name="zh-cn_topic_0000001456535116_b5371616181211"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121"><a name="zh-cn_topic_0000001456535116_p1129513513121"></a><a name="zh-cn_topic_0000001456535116_p1129513513121"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216"><a name="zh-cn_topic_0000001456535116_b13840714131216"></a><a name="zh-cn_topic_0000001456535116_b13840714131216"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121"><a name="zh-cn_topic_0000001456535116_b7555131016121"></a><a name="zh-cn_topic_0000001456535116_b7555131016121"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123"><a name="zh-cn_topic_0000001456535116_b199806129123"></a><a name="zh-cn_topic_0000001456535116_b199806129123"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791"><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123"><a name="zh-cn_topic_0000001456535116_b1399121919123"></a><a name="zh-cn_topic_0000001456535116_b1399121919123"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110"><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219"><a name="zh-cn_topic_0000001456535116_b12230192011219"></a><a name="zh-cn_topic_0000001456535116_b12230192011219"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216"><a name="zh-cn_topic_0000001456535116_b1622952141216"></a><a name="zh-cn_topic_0000001456535116_b1622952141216"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113"><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018"><a name="zh-cn_topic_0000001456535116_p340315471018"></a><a name="zh-cn_topic_0000001456535116_p340315471018"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul859810511118"></a><a name="ul859810511118"></a><ul id="ul859810511118"><li><span class="parmname" id="zh-cn_topic_0000001628542464_parmname18118141717010"><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a><a name="zh-cn_topic_0000001628542464_parmname18118141717010"></a>“indices”</span>、<span class="parmname" id="parmname14425191517241"><a name="parmname14425191517241"></a><a name="parmname14425191517241"></a>“queries”</span>、<span class="parmname" id="parmname871162382410"><a name="parmname871162382410"></a><a name="parmname871162382410"></a>“distances”</span>和<span class="parmname" id="parmname1431611816133"><a name="parmname1431611816133"></a><a name="parmname1431611816133"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table910711421721"></a>
<table><tbody><tr id="row13108642623"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1610817426217"><a name="p1610817426217"></a><a name="p1610817426217"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1013445511210"><a name="p1013445511210"></a><a name="p1013445511210"></a>APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);</p>
</td>
</tr>
<tr id="row141085421821"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1510864219218"><a name="p1510864219218"></a><a name="p1510864219218"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1810818421423"><a name="p1810818421423"></a><a name="p1810818421423"></a>在Search的基础上增加了阈值筛选，只返回满足阈值条件的结果，如传递有效的映射表（tableLen&gt;0且*table为非空指针），则返回映射后的topk结果。</p>
</td>
</tr>
<tr id="row12108942725"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p131081342022"><a name="p131081342022"></a><a name="p131081342022"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21081142426"><a name="p21081142426"></a><a name="p21081142426"></a><strong id="b810811421722"><a name="b810811421722"></a><a name="b810811421722"></a>int n</strong>：待查询特征向量的数目。</p>
<p id="p16108194212219"><a name="p16108194212219"></a><a name="p16108194212219"></a><strong id="b026811345317"><a name="b026811345317"></a><a name="b026811345317"></a>const float16_t *queries</strong>：待查询特征向量，长度为n * dim。</p>
<p id="p410824217215"><a name="p410824217215"></a><a name="p410824217215"></a><strong id="b1510814425212"><a name="b1510814425212"></a><a name="b1510814425212"></a>float threshold</strong>：用于过滤的阈值，接口不做值域范围约束，如果传递映射表，则该接口先将距离映射为score，然后再按照<span class="parmname" id="parmname7108442521"><a name="parmname7108442521"></a><a name="parmname7108442521"></a>“threshold”</span>进行过滤。</p>
<p id="p121087424218"><a name="p121087424218"></a><a name="p121087424218"></a><strong id="b31086422216"><a name="b31086422216"></a><a name="b31086422216"></a>int topk</strong>：query和底库的比对距离进行排序，返回<span class="parmname" id="parmname1010817428219"><a name="parmname1010817428219"></a><a name="parmname1010817428219"></a>“topk”</span>条结果。</p>
<p id="p21081342623"><a name="p21081342623"></a><a name="p21081342623"></a><strong id="b20108134216211"><a name="b20108134216211"></a><a name="b20108134216211"></a>unsigned int tableLen</strong>：映射表长度，默认值为0，表示不做映射。目前支持配置映射表长度为<span class="parmvalue" id="parmvalue181081542727"><a name="parmvalue181081542727"></a><a name="parmvalue181081542727"></a>“10000”</span>。</p>
<p id="p51081421027"><a name="p51081421027"></a><a name="p51081421027"></a><strong id="b101081421825"><a name="b101081421825"></a><a name="b101081421825"></a>const float *table</strong>：映射表指针，指向<span class="parmname" id="parmname810816423210"><a name="parmname810816423210"></a><a name="parmname810816423210"></a>“tableLen”</span>长度的有效映射值存储空间，目前支持的冗余长度为<span class="parmvalue" id="parmvalue01080422217"><a name="parmvalue01080422217"></a><a name="parmvalue01080422217"></a>“48”</span>，即<span class="parmname" id="parmname111089425219"><a name="parmname111089425219"></a><a name="parmname111089425219"></a>“*table”</span>指向的空间长度为10048 * sizeof(float) Byte。</p>
</td>
</tr>
<tr id="row1010816424210"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p81081742228"><a name="p81081742228"></a><a name="p81081742228"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p121088421922"><a name="p121088421922"></a><a name="p121088421922"></a><strong id="b11108142324"><a name="b11108142324"></a><a name="b11108142324"></a>int *num</strong>：每条待查询特征向量满足阈值条件的底库向量数量，长度为n。</p>
<p id="p1310817428215"><a name="p1310817428215"></a><a name="p1310817428215"></a><strong id="b12433151942"><a name="b12433151942"></a><a name="b12433151942"></a>idx_t* indices</strong>：满足阈值条件的底库向量下标索引，每个query从前往后记录符合条件的距离，然后按<span class="parmname" id="parmname5108174214220"><a name="parmname5108174214220"></a><a name="parmname5108174214220"></a>“topk”</span>补齐占用空间，<span class="parmname" id="parmname201086421622"><a name="parmname201086421622"></a><a name="parmname201086421622"></a>“indices”</span>总长度为n * topk。</p>
<p id="p6108194217212"><a name="p6108194217212"></a><a name="p6108194217212"></a><strong id="b81081429211"><a name="b81081429211"></a><a name="b81081429211"></a>float *distances</strong>：满足阈值条件的底库向量与待查询向量距离，记录方式和长度与<span class="parmname" id="parmname1310824214216"><a name="parmname1310824214216"></a><a name="parmname1310824214216"></a>“indices”</span>相同。</p>
</td>
</tr>
<tr id="row1810854219219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p310810423210"><a name="p310810423210"></a><a name="p310810423210"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p810819421524"><a name="p810819421524"></a><a name="p810819421524"></a><strong id="b710815421022"><a name="b710815421022"></a><a name="b710815421022"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1110811421218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p191086421521"><a name="p191086421521"></a><a name="p191086421521"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul2010818425217"></a><a name="ul2010818425217"></a><ul id="ul2010818425217"><li><strong id="b1810834211215"><a name="b1810834211215"></a><a name="b1810834211215"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname8108164219220"><a name="varname8108164219220"></a><a name="varname8108164219220"></a>capacity</span></i>]之间。</li><li><strong id="b610974212213"><a name="b610974212213"></a><a name="b610974212213"></a>topk</strong>：取值应在(0, 1024]之间。</li><li>传递<span class="parmname" id="zh-cn_topic_0000001456535116_parmname391111228612_1"><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a><a name="zh-cn_topic_0000001456535116_parmname391111228612_1"></a>“tableLen”</span>和<span class="parmname" id="zh-cn_topic_0000001456535116_parmname19267121920619_1"><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a><a name="zh-cn_topic_0000001456535116_parmname19267121920619_1"></a>“table”</span>参数同时满足要求时，接口会对计算出来的<strong id="zh-cn_topic_0000001456535116_b5371616181211_1"><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a><a name="zh-cn_topic_0000001456535116_b5371616181211_1"></a>distance</strong>进行映射：<p id="zh-cn_topic_0000001456535116_p1129513513121_1"><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a><a name="zh-cn_topic_0000001456535116_p1129513513121_1"></a>首先将<strong id="zh-cn_topic_0000001456535116_b13840714131216_1"><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a><a name="zh-cn_topic_0000001456535116_b13840714131216_1"></a>distance</strong>值归一化为 [0, 1]之间的浮点数<strong id="zh-cn_topic_0000001456535116_b7555131016121_1"><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a><a name="zh-cn_topic_0000001456535116_b7555131016121_1"></a>f1</strong>，然后用<strong id="zh-cn_topic_0000001456535116_b199806129123_1"><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a><a name="zh-cn_topic_0000001456535116_b199806129123_1"></a>f1</strong>乘上<span class="parmname" id="zh-cn_topic_0000001456535116_parmname14917143791_1"><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a><a name="zh-cn_topic_0000001456535116_parmname14917143791_1"></a>“tableLen”</span>并取整，这样得到[0, <strong id="zh-cn_topic_0000001456535116_b1399121919123_1"><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a><a name="zh-cn_topic_0000001456535116_b1399121919123_1"></a>tableLen</strong>]之间的整数索引，再利用该整数索引作为偏移，去<span class="parmname" id="zh-cn_topic_0000001456535116_parmname266193771110_1"><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a><a name="zh-cn_topic_0000001456535116_parmname266193771110_1"></a>“table”</span>指向的内存空间取出对应的<strong id="zh-cn_topic_0000001456535116_b12230192011219_1"><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a><a name="zh-cn_topic_0000001456535116_b12230192011219_1"></a>score</strong>，即完成映射，将<strong id="zh-cn_topic_0000001456535116_b1622952141216_1"><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a><a name="zh-cn_topic_0000001456535116_b1622952141216_1"></a>score</strong>存入<span class="parmname" id="zh-cn_topic_0000001456535116_parmname106381556121113_1"><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a><a name="zh-cn_topic_0000001456535116_parmname106381556121113_1"></a>“distance”</span> 。</p>
<p id="zh-cn_topic_0000001456535116_p340315471018_1"><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a><a name="zh-cn_topic_0000001456535116_p340315471018_1"></a>索引映射公式可抽象为((CosDistance + 1) / 2) * tableLen。</p>
</li></ul>
<a name="ul01091542821"></a><a name="ul01091542821"></a><ul id="ul01091542821"><li><span class="parmname" id="parmname3109184216214"><a name="parmname3109184216214"></a><a name="parmname3109184216214"></a>“indices”</span>、<span class="parmname" id="parmname1410910421821"><a name="parmname1410910421821"></a><a name="parmname1410910421821"></a>“queries”</span>、<span class="parmname" id="parmname310910421129"><a name="parmname310910421129"></a><a name="parmname310910421129"></a>“distances”</span>和<span class="parmname" id="parmname19109104217218"><a name="parmname19109104217218"></a><a name="parmname19109104217218"></a>“num”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### SetNTotal接口<a name="ZH-CN_TOPIC_0000002514776045"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>APP_ERROR SetNTotal(int n);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p7313759183119"><a name="p7313759183119"></a><a name="p7313759183119"></a>为外部提供调整<span class="parmname" id="parmname159164443116"><a name="parmname159164443116"></a><a name="parmname159164443116"></a>“ntotal”</span>计数。</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>每次增加底库向量后，Index内部尽管会根据最大插入下标更新<span class="parmname" id="parmname114061192322"><a name="parmname114061192322"></a><a name="parmname114061192322"></a>“ntotal”</span>值，但并没有记录[0, <i><span class="varname" id="varname1917151317325"><a name="varname1917151317325"></a><a name="varname1917151317325"></a>ntotal</span></i>]范围内哪些区域是无效的空间，因此<strong id="b144482818157"><a name="b144482818157"></a><a name="b144482818157"></a>RemoveFeatures</strong>操作没有改变<span class="parmname" id="parmname1274441121512"><a name="parmname1274441121512"></a><a name="parmname1274441121512"></a>“ntotal”</span>的值。用户如果在外部明确记录了增删操作后的最大底库索引位置，可以手动设置<span class="parmname" id="parmname159521818143214"><a name="parmname159521818143214"></a><a name="parmname159521818143214"></a>“ntotal”</span>，这样可以在可控范围内减少算子的计算量，以提高接口性能。</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>例如：当前插入100条向量，底库索引为0~99 时，ntotal = 100，执行删除索引为80~90的底库，此时Index内部<span class="parmname" id="parmname974517165332"><a name="parmname974517165332"></a><a name="parmname974517165332"></a>“ntotal”</span>保持不变，只能设为[<i><span class="varname" id="varname169891216331"><a name="varname169891216331"></a><a name="varname169891216331"></a>ntotal</span></i>, <i><span class="varname" id="varname91661324163313"><a name="varname91661324163313"></a><a name="varname91661324163313"></a>capacity</span></i>]之间的值，再次执行删除索引为90~99的底库，此时可以手动把<span class="parmname" id="parmname18801143812373"><a name="parmname18801143812373"></a><a name="parmname18801143812373"></a>“ntotal”</span>设置为[80, <i><span class="varname" id="varname737175673612"><a name="varname737175673612"></a><a name="varname737175673612"></a>capacity</span></i>]之间的值，设置为<span class="parmvalue" id="parmvalue8748356153711"><a name="parmvalue8748356153711"></a><a name="parmvalue8748356153711"></a>“80”</span>时，可以使参与比对的底库数据量有效减少20条。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b1215142783714"><a name="b1215142783714"></a><a name="b1215142783714"></a>int n</strong>：由用户在业务面管理的最大底库的索引加1。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在[0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</p>
</td>
</tr>
</tbody>
</table>


#### UpdateFeatures接口<a name="ZH-CN_TOPIC_0000002516314733"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p119217478565"><a name="p119217478565"></a><a name="p119217478565"></a>APP_ERROR UpdateFeatures (int n, const float16_t *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>向特征库更新“n”个指定下标索引的特征向量，如果在下标处不存在特征向量，则添加；如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p10574435124710"><a name="p10574435124710"></a><a name="p10574435124710"></a><strong id="b18283163233118"><a name="b18283163233118"></a><a name="b18283163233118"></a>int n</strong>：插入特征向量数目。</p>
<p id="p042220329586"><a name="p042220329586"></a><a name="p042220329586"></a><strong id="b17419938175818"><a name="b17419938175818"></a><a name="b17419938175818"></a>const float16_t *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
<p id="p18422153235817"><a name="p18422153235817"></a><a name="p18422153235817"></a><strong id="b5921164425815"><a name="b5921164425815"></a><a name="b5921164425815"></a>const idx_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为n。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1352374783110"><a name="b1352374783110"></a><a name="b1352374783110"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul124151524115"></a><a name="ul124151524115"></a><ul id="ul124151524115"><li><strong id="b81701423114016"><a name="b81701423114016"></a><a name="b81701423114016"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname85248414222"><a name="varname85248414222"></a><a name="varname85248414222"></a>ntotal</span></i>)之间。</li><li><strong id="b186931931123913"><a name="b186931931123913"></a><a name="b186931931123913"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname75125539397"><a name="varname75125539397"></a><a name="varname75125539397"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname422220519356"><a name="parmname422220519356"></a><a name="parmname422220519356"></a>“features”</span>和<span class="parmname" id="parmname56641160353"><a name="parmname56641160353"></a><a name="parmname56641160353"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table19567183517113"></a>
<table><tbody><tr id="row145678353110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1056713351120"><a name="p1056713351120"></a><a name="p1056713351120"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13567835018"><a name="p13567835018"></a><a name="p13567835018"></a>APP_ERROR UpdateFeatures(int n, const float *features, const idx_t *indices);</p>
</td>
</tr>
<tr id="row1256719351818"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p175675351511"><a name="p175675351511"></a><a name="p175675351511"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p20567123520115"><a name="p20567123520115"></a><a name="p20567123520115"></a>向特征库更新“n”个指定下标索引的特征向量，如果在下标处不存在特征向量，则添加；如果在下标处已存在特征向量，则修改。</p>
</td>
</tr>
<tr id="row756713352110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1456793520110"><a name="p1456793520110"></a><a name="p1456793520110"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p2567133518120"><a name="p2567133518120"></a><a name="p2567133518120"></a><strong id="b65671335119"><a name="b65671335119"></a><a name="b65671335119"></a>int n</strong>：插入特征向量数目。</p>
<p id="p165673357114"><a name="p165673357114"></a><a name="p165673357114"></a><strong id="b3277341929"><a name="b3277341929"></a><a name="b3277341929"></a>const float *features</strong>：待插入的特征向量，长度为n * 向量维度dim。</p>
<p id="p175671135216"><a name="p175671135216"></a><a name="p175671135216"></a><strong id="b2567173515117"><a name="b2567173515117"></a><a name="b2567173515117"></a>const idx_t *indices</strong>：待插入特征向量对应的下标索引，有效长度为n。</p>
</td>
</tr>
<tr id="row456710351212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1556715352111"><a name="p1556715352111"></a><a name="p1556715352111"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1956716351718"><a name="p1956716351718"></a><a name="p1956716351718"></a>无</p>
</td>
</tr>
<tr id="row155678359112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p115677351519"><a name="p115677351519"></a><a name="p115677351519"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1056773515119"><a name="p1056773515119"></a><a name="p1056773515119"></a><strong id="b1456793518119"><a name="b1456793518119"></a><a name="b1456793518119"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row65673351912"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p9567135715"><a name="p9567135715"></a><a name="p9567135715"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1567203512115"></a><a name="ul1567203512115"></a><ul id="ul1567203512115"><li><strong id="b556783510117"><a name="b556783510117"></a><a name="b556783510117"></a>indices</strong>：每个特征的索引应在[0, <i><span class="varname" id="varname185673351619"><a name="varname185673351619"></a><a name="varname185673351619"></a>ntotal</span></i>)之间。</li><li><strong id="b1456713512117"><a name="b1456713512117"></a><a name="b1456713512117"></a>n</strong>：取值应在(0, <i><span class="varname" id="varname1756723512112"><a name="varname1756723512112"></a><a name="varname1756723512112"></a>capacity</span></i>]之间。</li><li><span class="parmname" id="parmname175671351116"><a name="parmname175671351116"></a><a name="parmname175671351116"></a>“features”</span>和<span class="parmname" id="parmname19567935111"><a name="parmname19567935111"></a><a name="parmname19567935111"></a>“indices”</span>需要为非空指针，且长度符合限制，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>



