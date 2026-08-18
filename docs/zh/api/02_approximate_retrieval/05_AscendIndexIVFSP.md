
# AscendIndexIVFSP<a name="ZH-CN_TOPIC_0000001635576081"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001635815481"></a>

昇腾原生IVFSP检索算法，使用自研矩阵近似策略，压缩特征向量后存底库，并使用自研倒排链策略选取出最可能包含Ground Truth（真实）的底库，最后使用自研检索策略在倒排链过滤后的底库进行检索得到Top K向量结果。

AscendIndexIVFSP只支持标准态场景，且只支持<term>Atlas 推理系列产品</term>。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## add接口<a name="ZH-CN_TOPIC_0000001585895568"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void add(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>向底库中添加特征向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1794683517262"><a name="p1794683517262"></a><a name="p1794683517262"></a><strong id="b016584171514"><a name="b016584171514"></a><a name="b016584171514"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p1594616353266"><a name="p1594616353266"></a><a name="p1594616353266"></a><strong id="b12572953101710"><a name="b12572953101710"></a><a name="b12572953101710"></a>const float *x</strong>：添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1572065617233"></a><a name="ul1572065617233"></a><ul id="ul1572065617233"><li>指针<span class="parmname" id="parmname241211111910"><a name="parmname241211111910"></a><a name="parmname241211111910"></a>“x”</span>的长度应该为dims * <strong id="b99141183199"><a name="b99141183199"></a><a name="b99141183199"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li>底库向量总数<span class="parmname" id="parmname913981814284"><a name="parmname913981814284"></a><a name="parmname913981814284"></a>“n”</span>通常大于0且小于1e9。</li><li>一次性add的数据量应该小于等于特征底库数据大小。</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
>- add接口不能与add\_with\_ids接口混用。
>- 使用add接口后，search结果的labels可能会重复，如果业务上对label有要求，建议使用add\_with\_ids接口。
>- add接口在小batch添加场景进行了性能优化，此场景根据数据集不同，精度会有所降低，建议在已有底库场景下用小batch添加。

## add\_with\_ids接口<a name="ZH-CN_TOPIC_0000001586055512"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>向底库中添加特征向量并指定对应的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p133861411291"><a name="p133861411291"></a><a name="p133861411291"></a><strong id="b1195072611184"><a name="b1195072611184"></a><a name="b1195072611184"></a>idx_t n</strong>：添加进底库的特征向量数量。</p>
<p id="p338634162915"><a name="p338634162915"></a><a name="p338634162915"></a><strong id="b551412342183"><a name="b551412342183"></a><a name="b551412342183"></a>const float *x</strong>：添加进底库的特征向量。</p>
<p id="p17386184182913"><a name="p17386184182913"></a><a name="p17386184182913"></a><strong id="b42592374181"><a name="b42592374181"></a><a name="b42592374181"></a>const idx_t *ids</strong>：添加进底库的特征向量对应的ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p152321022122915"><a name="p152321022122915"></a><a name="p152321022122915"></a>指针<span class="parmname" id="parmname241211111910"><a name="parmname241211111910"></a><a name="parmname241211111910"></a>“x”</span>的长度应该为dims * <strong id="b99141183199"><a name="b99141183199"></a><a name="b99141183199"></a>n</strong>，指针<span class="parmname" id="parmname186814213323"><a name="parmname186814213323"></a><a name="parmname186814213323"></a>“ids”</span>的长度应为<span class="parmname" id="parmname492873714325"><a name="parmname492873714325"></a><a name="parmname492873714325"></a>“n”</span>，否则可能出现越界读写错误并引起程序崩溃。用户需要根据自己的业务场景，保证<span class="parmname" id="parmname39625451329"><a name="parmname39625451329"></a><a name="parmname39625451329"></a>“ids”</span>的合法性，如底库中存在重复的ID，检索结果中的<span class="parmname" id="parmname1770501133315"><a name="parmname1770501133315"></a><a name="parmname1770501133315"></a>“label”</span>将无法对应具体的底库向量。</p>
<p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a><span class="parmname" id="parmname31332191105"><a name="parmname31332191105"></a><a name="parmname31332191105"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
> add\_with\_ids接口在小batch添加场景进行了性能优化，此场景根据数据集不同，精度会有所降低，建议在已有底库场景下用小batch添加。

## AscendIndexIVFSP接口<a name="ZH-CN_TOPIC_0000001585736168"></a>

> [!NOTE]
>将参数“config”传递给函数前，请根据实际情况先设置conf.handleBatch、conf.nprobe、conf.searchListSize的值（字段描述参考[公共参数](./06_AscendIndexIVFSPConfig.md#ZH-CN_TOPIC_0000001635696057)）。
>其中conf.handleBatch、conf.searchListSize值需与[IVFSP](../../05_user_guide.md#ivfsp)业务算子模型文件生成中的nprobe handle batch、search list size保持一致。
>conf.filterable（继承自[AscendIndexConfig](../01_full_retrieval/03_AscendIndexConfig.md#ascendindexconfig) ）默认为“false”，如果要使用search\_with\_filter\(\)接口，需设置**conf.filterable = true**。“conf.filterable”设置为“true”将在NPU卡上存储额外的信息，消耗更多的NPU卡上内存。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>AscendIndexIVFSP的构造函数，根据<span class="parmname" id="parmname18661204514127"><a name="parmname18661204514127"></a><a name="parmname18661204514127"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1561263817143"><a name="p1561263817143"></a><a name="p1561263817143"></a><strong id="b19612183881419"><a name="b19612183881419"></a><a name="b19612183881419"></a>int dims</strong>：AscendIndexIVFSP管理的一组特征向量的维度。</p>
<p id="p1661213891412"><a name="p1661213891412"></a><a name="p1661213891412"></a><strong id="b48633716167"><a name="b48633716167"></a><a name="b48633716167"></a>int nonzeroNum</strong>：特征向量压缩降维后非零维度个数。</p>
<p id="p17612133810143"><a name="p17612133810143"></a><a name="p17612133810143"></a><strong id="b761216381145"><a name="b761216381145"></a><a name="b761216381145"></a>int nlist</strong>：聚类中心的个数，与<a href="../../05_user_guide.md#ivfsp">IVFSP业务算子模型文件生成</a>中的&lt;centroid num&gt;参数值对应。</p>
<p id="p166121738111411"><a name="p166121738111411"></a><a name="p166121738111411"></a><strong id="b249718368317"><a name="b249718368317"></a><a name="b249718368317"></a>const char *codeBookPath</strong>：IVFSP使用的码本文件路径。</p>
<p id="p1061210384146"><a name="p1061210384146"></a><a name="p1061210384146"></a><strong id="b451311429018"><a name="b451311429018"></a><a name="b451311429018"></a>faiss::ScalarQuantizer::QuantizerType qType</strong>：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
<p id="p2617038181412"><a name="p2617038181412"></a><a name="p2617038181412"></a><strong id="b1861763817147"><a name="b1861763817147"></a><a name="b1861763817147"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。当前<span class="parmname" id="parmname858915211735"><a name="parmname858915211735"></a><a name="parmname858915211735"></a>“faiss::MetricType metric”</span>仅支持<span class="parmvalue" id="parmvalue1483205813413"><a name="parmvalue1483205813413"></a><a name="parmvalue1483205813413"></a>“METRIC_L2”</span>。</p>
<p id="p1161723812145"><a name="p1161723812145"></a><a name="p1161723812145"></a><strong id="b11550924103117"><a name="b11550924103117"></a><a name="b11550924103117"></a>AscendIndexIVFSPConfig</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul719511175195"></a><a name="ul719511175195"></a><ul id="ul719511175195"><li>训练生成码本时的&lt;dim&gt;、&lt;nonzero num&gt;、&lt;centroid num&gt; 值应该与此函数的参数<span class="parmname" id="parmname838317156357"><a name="parmname838317156357"></a><a name="parmname838317156357"></a>“dims”</span>、<span class="parmname" id="parmname15618111743514"><a name="parmname15618111743514"></a><a name="parmname15618111743514"></a>“nonzeroNum”</span>、<span class="parmname" id="parmname10133020143510"><a name="parmname10133020143510"></a><a name="parmname10133020143510"></a>“nlist”</span>对应。</li><li><span class="parmname" id="parmname35157306285"><a name="parmname35157306285"></a><a name="parmname35157306285"></a>“codeBookPath”</span>加载的码本应该与此函数的参数<span class="parmname" id="parmname13115348162817"><a name="parmname13115348162817"></a><a name="parmname13115348162817"></a>“dims”</span>、<span class="parmname" id="parmname164983568286"><a name="parmname164983568286"></a><a name="parmname164983568286"></a>“nonzeroNum”</span>、<span class="parmname" id="parmname58341316295"><a name="parmname58341316295"></a><a name="parmname58341316295"></a>“nlist”</span>对应，且程序的执行用户是码本文件的属主；且码本文件不能为软链接。</li><li>当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。</li><li><span class="parmname" id="parmname178225132016"><a name="parmname178225132016"></a><a name="parmname178225132016"></a>“nonzeroNum”</span>需为16的倍数且小于等于min(128, dims)。</li><li>metric ∈ {faiss::MetricType::METRIC_L2}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table49022324218"></a>
<table><tbody><tr id="row199021732102118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p79020325216"><a name="p79020325216"></a><a name="p79020325216"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7378142174712"><a name="p7378142174712"></a><a name="p7378142174712"></a>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &amp;codeBookSharedIdx, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</p>
</td>
</tr>
<tr id="row190216323214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p13902153202111"><a name="p13902153202111"></a><a name="p13902153202111"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1390393214211"><a name="p1390393214211"></a><a name="p1390393214211"></a>AscendIndexIVFSP的构造函数，根据<span class="parmname" id="parmname19903532122113"><a name="parmname19903532122113"></a><a name="parmname19903532122113"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row3903113252110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p89039324212"><a name="p89039324212"></a><a name="p89039324212"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p790310324213"><a name="p790310324213"></a><a name="p790310324213"></a><strong id="b209032032182118"><a name="b209032032182118"></a><a name="b209032032182118"></a>int dims</strong>：AscendIndexIVFSP管理的一组特征向量的维度。</p>
<p id="p1590310322217"><a name="p1590310322217"></a><a name="p1590310322217"></a><strong id="b99031324210"><a name="b99031324210"></a><a name="b99031324210"></a>int nonzeroNum</strong>：特征向量压缩降维后非零维度个数。</p>
<p id="p490373216218"><a name="p490373216218"></a><a name="p490373216218"></a><strong id="b390383218212"><a name="b390383218212"></a><a name="b390383218212"></a>int nlist</strong>：聚类中心的个数，与<a href="../../05_user_guide.md#ivfsp">IVFSP业务算子模型文件生成</a>中的&lt;centroid num&gt;参数值对应。</p>
<p id="p390313219218"><a name="p390313219218"></a><a name="p390313219218"></a><strong id="b116451015104820"><a name="b116451015104820"></a><a name="b116451015104820"></a>const AscendIndexIVFSP &amp;codeBookSharedIdx</strong>：共享码本的AscendIndexIVFSP对象。</p>
<p id="p1990343252111"><a name="p1990343252111"></a><a name="p1990343252111"></a><strong id="b49034325211"><a name="b49034325211"></a><a name="b49034325211"></a>faiss::ScalarQuantizer::QuantizerType qType</strong>：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
<p id="p14903132162110"><a name="p14903132162110"></a><a name="p14903132162110"></a><strong id="b119031132182114"><a name="b119031132182114"></a><a name="b119031132182114"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。当前<span class="parmname" id="parmname7903143217211"><a name="parmname7903143217211"></a><a name="parmname7903143217211"></a>“faiss::MetricType metric”</span>仅支持<span class="parmvalue" id="parmvalue1190333211216"><a name="parmvalue1190333211216"></a><a name="parmvalue1190333211216"></a>“METRIC_L2”</span>。</p>
<p id="p20903173272114"><a name="p20903173272114"></a><a name="p20903173272114"></a><strong id="b1490310325216"><a name="b1490310325216"></a><a name="b1490310325216"></a>AscendIndexIVFSPConfig</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row890313323211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1090314321219"><a name="p1090314321219"></a><a name="p1090314321219"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p6903123222115"><a name="p6903123222115"></a><a name="p6903123222115"></a>无</p>
</td>
</tr>
<tr id="row190393211210"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2903532142118"><a name="p2903532142118"></a><a name="p2903532142118"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p18903163218211"><a name="p18903163218211"></a><a name="p18903163218211"></a>无</p>
</td>
</tr>
<tr id="row4903123214219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p39031232162110"><a name="p39031232162110"></a><a name="p39031232162110"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul790383216215"></a><a name="ul790383216215"></a><ul id="ul790383216215"><li>训练生成码本时的&lt;dim&gt;、&lt;nonzero num&gt;、&lt;centroid num&gt; 值应该与此函数的参数<span class="parmname" id="parmname159038326214"><a name="parmname159038326214"></a><a name="parmname159038326214"></a>“dims”</span>、<span class="parmname" id="parmname390318322215"><a name="parmname390318322215"></a><a name="parmname390318322215"></a>“nonzeroNum”</span>、<span class="parmname" id="parmname69031632182114"><a name="parmname69031632182114"></a><a name="parmname69031632182114"></a>“nlist”</span>对应。</li><li>codeBookSharedIdx共享码本的码本配置要与当前Index的码本配置相同，且配置相同的Device资源。</li><li>当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。</li><li><span class="parmname" id="parmname13753112012420"><a name="parmname13753112012420"></a><a name="parmname13753112012420"></a>“nonzeroNum”</span>需为16的倍数且小于等于min(128, dims)。</li><li>metric ∈ {faiss::MetricType::METRIC_L2}。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table8581162710235"></a>
<table><tbody><tr id="row258119270238"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p758152711231"><a name="p758152711231"></a><a name="p758152711231"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p158117272235"><a name="p158117272235"></a><a name="p158117272235"></a>AscendIndexIVFSP (const AscendIndexIVFSP&amp;) = delete;</p>
</td>
</tr>
<tr id="row6581192742313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p358110271235"><a name="p358110271235"></a><a name="p358110271235"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2581327192318"><a name="p2581327192318"></a><a name="p2581327192318"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row858114273233"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7581162715238"><a name="p7581162715238"></a><a name="p7581162715238"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b650433582416"><a name="b650433582416"></a><a name="b650433582416"></a>const AscendIndexIVFSP&amp;</strong>：常量AscendIndexIVFSP。</p>
</td>
</tr>
<tr id="row5581152722313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p458117278231"><a name="p458117278231"></a><a name="p458117278231"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1258111270239"><a name="p1258111270239"></a><a name="p1258111270239"></a>无</p>
</td>
</tr>
<tr id="row4581162702318"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p15581202722317"><a name="p15581202722317"></a><a name="p15581202722317"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p3581127182311"><a name="p3581127182311"></a><a name="p3581127182311"></a>无</p>
</td>
</tr>
<tr id="row125811227162312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p45811227132319"><a name="p45811227132319"></a><a name="p45811227132319"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table186918413239"></a>
<table><tbody><tr id="row1386916412234"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p78691441132310"><a name="p78691441132310"></a><a name="p78691441132310"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1386914415234"><a name="p1386914415234"></a><a name="p1386914415234"></a>virtual ~AscendIndexIVFSP();</p>
</td>
</tr>
<tr id="row686920419239"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1886910414237"><a name="p1886910414237"></a><a name="p1886910414237"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p178691241122313"><a name="p178691241122313"></a><a name="p178691241122313"></a>AscendIndexIVFSP的析构函数，销毁AscendIndexIVFSP对象，释放资源。</p>
</td>
</tr>
<tr id="row28695418235"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128698415234"><a name="p128698415234"></a><a name="p128698415234"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p086914116238"><a name="p086914116238"></a><a name="p086914116238"></a>无</p>
</td>
</tr>
<tr id="row19869641142315"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18691641192314"><a name="p18691641192314"></a><a name="p18691641192314"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2869204132310"><a name="p2869204132310"></a><a name="p2869204132310"></a>无</p>
</td>
</tr>
<tr id="row6869134122317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p16869114115232"><a name="p16869114115232"></a><a name="p16869114115232"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p9869164162316"><a name="p9869164162316"></a><a name="p9869164162316"></a>无</p>
</td>
</tr>
<tr id="row3869841102310"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19869441112317"><a name="p19869441112317"></a><a name="p19869441112317"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p186914112231"><a name="p186914112231"></a><a name="p186914112231"></a>无</p>
</td>
</tr>
</tbody>
</table>

<a name="table241282321712"></a>
<table><tbody><tr id="row1441202301711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1541222315179"><a name="p1541222315179"></a><a name="p1541222315179"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p15204727163415"><a name="p15204727163415"></a><a name="p15204727163415"></a>AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());</p>
</td>
</tr>
<tr id="row84121238175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3412122315172"><a name="p3412122315172"></a><a name="p3412122315172"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p56033285334"><a name="p56033285334"></a><a name="p56033285334"></a>AscendIndexIVFSP的构造函数，根据“config”中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row164121237173"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p164121723181717"><a name="p164121723181717"></a><a name="p164121723181717"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul534674763310"></a><a name="ul534674763310"></a><ul id="ul534674763310"><li>int dims：AscendIndexIVFSP管理的一组特征向量的维度。</li><li>int nonzeroNum：特征向量压缩降维后非零维度个数。</li><li>int nlist：聚类中心的个数，与<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP业务算子模型文件生成”中的&lt;centroid num&gt;参数值对应。</li><li>faiss::ScalarQuantizer::QuantizerType qType：标量量化类型，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</li><li>faiss::MetricType metric：AscendIndex在执行特征向量相似度检索时使用的距离度量类型。当前“faiss::MetricType metric”仅支持“METRIC_L2”。</li><li>AscendIndexIVFSPConfig：Device侧资源配置。</li></ul>
</td>
</tr>
<tr id="row6413192313178"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p6413142341717"><a name="p6413142341717"></a><a name="p6413142341717"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p19203142716347"><a name="p19203142716347"></a><a name="p19203142716347"></a>无</p>
</td>
</tr>
<tr id="row341316237178"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p24131423201720"><a name="p24131423201720"></a><a name="p24131423201720"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p6203427123419"><a name="p6203427123419"></a><a name="p6203427123419"></a>无</p>
</td>
</tr>
<tr id="row7413102361711"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p6413323131715"><a name="p6413323131715"></a><a name="p6413323131715"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul162613915340"></a><a name="ul162613915340"></a><ul id="ul162613915340"><li>当dims ∈ {64, 128, 256}时，nlist∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dims ∈ {512, 768}时，nlist∈ {256, 512, 1024, 2048}。</li><li>“nonzeroNum”需为16的倍数且小于等于min(128, dims)。</li><li>metric ∈ {faiss::MetricType::METRIC_L2}。</li></ul>
</td>
</tr>
</tbody>
</table>

## loadAllData接口<a id="ZH-CN_TOPIC_0000001585736172"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void loadAllData(const char *dataPath);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>将Index结构从磁盘读入Device，包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b102253712317"><a name="b102253712317"></a><a name="b102253712317"></a>const char *dataPath：</strong>数据文件路径。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a><span class="parmvalue" id="parmvalue292811515461"><a name="parmvalue292811515461"></a><a name="parmvalue292811515461"></a>“dataPath”</span>对应的文件应该是调用saveAllData方法得到的落盘文件，程序执行用户对其有读权限；且文件不能为软链接。</p>
<p id="p1430141710323"><a name="p1430141710323"></a><a name="p1430141710323"></a>该接口无法共享码本，如需共享码本，建议使用loadAllData。</p>
</td>
</tr>
</tbody>
</table>

<a name="table115591219131513"></a>
<table><tbody><tr id="row1955918198153"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p255921981517"><a name="p255921981517"></a><a name="p255921981517"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p390319328341"><a name="p390319328341"></a><a name="p390319328341"></a>static std::shared_ptr&lt;AscendIndexIVFSP&gt; loadAllData(const AscendIndexIVFSPConfig &amp;config, const uint8_t *data, size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx = nullptr);</p>
</td>
</tr>
<tr id="row10559191931517"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1559111916158"><a name="p1559111916158"></a><a name="p1559111916158"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p144174125351"><a name="p144174125351"></a><a name="p144174125351"></a>从内存中恢复AscendIndexIVFSP对象。</p>
</td>
</tr>
<tr id="row4559219161516"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5559519101517"><a name="p5559519101517"></a><a name="p5559519101517"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul3105123212352"></a><a name="ul3105123212352"></a><ul id="ul3105123212352"><li><strong id="b3975154112358"><a name="b3975154112358"></a><a name="b3975154112358"></a>const AscendIndexIVFSPConfig &amp;config</strong>：Device侧资源配置，当前只需设置config.deviceList以及config.resourceSize即可，其他配置参数会从内存中恢复。</li><li><strong id="b8319545173518"><a name="b8319545173518"></a><a name="b8319545173518"></a>const uint8_t *data</strong>：由saveAllData方法得到的内存指针。</li><li><strong id="b1312214484354"><a name="b1312214484354"></a><a name="b1312214484354"></a>size_t dataLen</strong>：data指针的真实长度。</li><li><strong id="b125458521353"><a name="b125458521353"></a><a name="b125458521353"></a>const AscendIndexIVFSP *codeBookSharedIdx</strong>：共享码本的AscendIndexIVFSP指针，默认为nullptr，即不共享码本。</li></ul>
</td>
</tr>
<tr id="row18559201914151"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1255961941516"><a name="p1255961941516"></a><a name="p1255961941516"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1555981921519"><a name="p1555981921519"></a><a name="p1555981921519"></a>无</p>
</td>
</tr>
<tr id="row855915191150"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p6560131941520"><a name="p6560131941520"></a><a name="p6560131941520"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1671312145366"><a name="p1671312145366"></a><a name="p1671312145366"></a>从内存中恢复的AscendIndexIVFSP智能指针对象。</p>
</td>
</tr>
<tr id="row956019190157"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5560111920158"><a name="p5560111920158"></a><a name="p5560111920158"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4752731193612"></a><a name="ul4752731193612"></a><ul id="ul4752731193612"><li>data需要为非空的合法指针。</li><li>dataLen为指针data的真实长度，否则可能出现越界读写错误并引起程序崩溃。</li><li>codeBookSharedIdx共享码本的码本配置要与当前Index的码本配置相同，且配置相同的Device资源。</li></ul>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001635975413"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>AscendIndexIVFSP&amp; operator=(const AscendIndexIVFSP&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b88405472511"><a name="b88405472511"></a><a name="b88405472511"></a>const AscendIndexIVFSP&amp;</strong>：常量AscendIndexIVFSP。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001635576085"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>实现AscendIndexIVFSP删除底库中指定的特征向量的接口。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b15391516143014"><a name="b15391516143014"></a><a name="b15391516143014"></a>const faiss::IDSelector &amp;sel</strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>返回被删除的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

## reset接口<a name="ZH-CN_TOPIC_0000001635815485"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void reset() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>清空该AscendIndexIVFSP的底库向量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

## saveAllData接口<a name="ZH-CN_TOPIC_0000001635696053"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void saveAllData(const char *dataPath);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>将Index结构从Device侧写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b19557108143418"><a name="b19557108143418"></a><a name="b19557108143418"></a>const char *dataPath</strong>：保存的数据文件路径。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>用户应该保证<span class="parmvalue" id="parmvalue972214587421"><a name="parmvalue972214587421"></a><a name="parmvalue972214587421"></a>“dataPath”</span>文件路径所在的目录存在，且执行用户对目录具有写权限；出于安全加固的考虑，目录层级中不能含有软链接。</p>
<p id="p10274445174214"><a name="p10274445174214"></a><a name="p10274445174214"></a>当<span class="parmvalue" id="parmvalue4193131715434"><a name="parmvalue4193131715434"></a><a name="parmvalue4193131715434"></a>“dataPath”</span>对应的文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</p>
</td>
</tr>
</tbody>
</table>

<a name="table11876949141314"></a>
<table><tbody><tr id="row12876549141317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p20876649191320"><a name="p20876649191320"></a><a name="p20876649191320"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7338202573713"><a name="p7338202573713"></a><a name="p7338202573713"></a>void saveAllData(uint8_t *&amp;data, size_t &amp;dataLen) const;</p>
</td>
</tr>
<tr id="row1587654912137"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p148761549201313"><a name="p148761549201313"></a><a name="p148761549201313"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6203133116375"><a name="p6203133116375"></a><a name="p6203133116375"></a>将AscendIndexIVFSP对象存储至内存中。</p>
</td>
</tr>
<tr id="row17876184916136"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p38761849171311"><a name="p38761849171311"></a><a name="p38761849171311"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1333871953710"><a name="p1333871953710"></a><a name="p1333871953710"></a>无</p>
</td>
</tr>
<tr id="row7876174971312"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p138768495136"><a name="p138768495136"></a><a name="p138768495136"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1142124363714"><a name="p1142124363714"></a><a name="p1142124363714"></a><strong id="b7190348133717"><a name="b7190348133717"></a><a name="b7190348133717"></a>uint8_t *&amp;data</strong>：存储AscendIndexIVFSP数据的内存指针。</p>
<p id="p10142204317379"><a name="p10142204317379"></a><a name="p10142204317379"></a><strong id="b133501052103714"><a name="b133501052103714"></a><a name="b133501052103714"></a>size_t &amp;dataLen</strong>：data指针的真实长度。</p>
</td>
</tr>
<tr id="row487615490131"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p3876949141317"><a name="p3876949141317"></a><a name="p3876949141317"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p233621903719"><a name="p233621903719"></a><a name="p233621903719"></a>无</p>
</td>
</tr>
<tr id="row987624981313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5876149181310"><a name="p5876149181310"></a><a name="p5876149181310"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1418092193810"><a name="p1418092193810"></a><a name="p1418092193810"></a>传入的data需要为空指针，且接口返回后需要用户使用完data后通过delete来释放其内存，否则会造成内存泄漏。</p>
</td>
</tr>
</tbody>
</table>

## search接口<a name="ZH-CN_TOPIC_0000001635815489"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1587414619374"><a name="p1587414619374"></a><a name="p1587414619374"></a>void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>实现AscendIndexIVFSP特征向量查询接口，根据输入的特征向量返回最相似的“k”条特征的ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1246083018397"><a name="p1246083018397"></a><a name="p1246083018397"></a><strong id="b1246015305391"><a name="b1246015305391"></a><a name="b1246015305391"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p17460230163917"><a name="p17460230163917"></a><a name="p17460230163917"></a><strong id="b12460133016398"><a name="b12460133016398"></a><a name="b12460133016398"></a>const float *x</strong>：特征向量数据。</p>
<p id="p546073017399"><a name="p546073017399"></a><a name="p546073017399"></a><strong id="b5460630173912"><a name="b5460630173912"></a><a name="b5460630173912"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p13712185717441"><a name="p13712185717441"></a><a name="p13712185717441"></a><strong id="b15637734194512"><a name="b15637734194512"></a><a name="b15637734194512"></a>const SearchParameters *params：</strong>Faiss的可选参数，默认为“nullptr”，暂不支持该参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p9711201512411"><a name="p9711201512411"></a><a name="p9711201512411"></a><strong id="b127111415174119"><a name="b127111415174119"></a><a name="b127111415174119"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname294711157429"><a name="parmname294711157429"></a><a name="parmname294711157429"></a>“k”</span>个向量间的距离值。当有效的检索结果不足<span class="parmname" id="parmname322862410423"><a name="parmname322862410423"></a><a name="parmname322862410423"></a>“k”</span>个时，剩余无效距离用65504或-65504填充。</p>
<p id="p4711515104119"><a name="p4711515104119"></a><a name="p4711515104119"></a><strong id="b571161514419"><a name="b571161514419"></a><a name="b571161514419"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname117371282423"><a name="parmname117371282423"></a><a name="parmname117371282423"></a>“k”</span>个向量的ID。当有效的检索结果不足<span class="parmname" id="parmname176161826144211"><a name="parmname176161826144211"></a><a name="parmname176161826144211"></a>“k”</span>个时，剩余无效label用<span class="parmvalue" id="parmvalue6403131919432"><a name="parmvalue6403131919432"></a><a name="parmvalue6403131919432"></a>“-1”</span>填充。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>查询的特征向量数据<span class="parmname" id="parmname1087952510438"><a name="parmname1087952510438"></a><a name="parmname1087952510438"></a>“x”</span>的长度应该为dims * <strong id="b1279975684114"><a name="b1279975684114"></a><a name="b1279975684114"></a>n</strong>，<span class="parmname" id="parmname948613316435"><a name="parmname948613316435"></a><a name="parmname948613316435"></a>“distances”</span>以及<span class="parmname" id="parmname7858139114317"><a name="parmname7858139114317"></a><a name="parmname7858139114317"></a>“labels”</span>的长度应该为<strong id="b187993567415"><a name="b187993567415"></a><a name="b187993567415"></a>k</strong> * <strong id="b27992562413"><a name="b27992562413"></a><a name="b27992562413"></a>n</strong>，否则可能会出现越界读写的情况，引起程序的崩溃。此处<span class="parmname" id="parmname1838195014437"><a name="parmname1838195014437"></a><a name="parmname1838195014437"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9；<span class="parmname" id="parmname10590145844313"><a name="parmname10590145844313"></a><a name="parmname10590145844313"></a>“k”</span>通常不允许超过4096。</p>
</td>
</tr>
</tbody>
</table>

## search\_with\_filter接口<a name="ZH-CN_TOPIC_0000001585736176"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>AscendIndexIVFSP的特征向量查询接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname1297124204519"><a name="parmname1297124204519"></a><a name="parmname1297124204519"></a>“k”</span>条特征的ID。提供基于CID过滤的功能，“filters”为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b12607182024312"><a name="b12607182024312"></a><a name="b12607182024312"></a>idx_t n</strong>：查询的特征向量的条数。</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><strong id="b960782054317"><a name="b960782054317"></a><a name="b960782054317"></a>const float *x</strong>：特征向量数据。</p>
<p id="p176073203438"><a name="p176073203438"></a><a name="p176073203438"></a><strong id="b9607820114310"><a name="b9607820114310"></a><a name="b9607820114310"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p76071120164313"><a name="p76071120164313"></a><a name="p76071120164313"></a><strong id="b16607122014317"><a name="b16607122014317"></a><a name="b16607122014317"></a>const void *filters</strong>：过滤条件。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14659337439"><a name="p14659337439"></a><a name="p14659337439"></a><strong id="b96520338430"><a name="b96520338430"></a><a name="b96520338430"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname993415618467"><a name="parmname993415618467"></a><a name="parmname993415618467"></a>“k”</span>个向量间的距离值。</p>
<p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b76513333436"><a name="b76513333436"></a><a name="b76513333436"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname23731212124614"><a name="parmname23731212124614"></a><a name="parmname23731212124614"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li><span class="parmname" id="parmname17755173284714"><a name="parmname17755173284714"></a><a name="parmname17755173284714"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li><strong id="b2977182414719"><a name="b2977182414719"></a><a name="b2977182414719"></a>“k”</strong>通常不允许超过4096。</li><li><span class="parmname" id="parmname106134074720"><a name="parmname106134074720"></a><a name="parmname106134074720"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b189302361443"><a name="b189302361443"></a><a name="b189302361443"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname17598461489"><a name="parmname17598461489"></a><a name="parmname17598461489"></a>“distances”</span>、<span class="parmname" id="parmname1527410554487"><a name="parmname1527410554487"></a><a name="parmname1527410554487"></a>“labels”</span>需要为非空指针，且长度应该为<strong id="b3930193664416"><a name="b3930193664416"></a><a name="b3930193664416"></a>k</strong> * <strong id="b1493017362449"><a name="b1493017362449"></a><a name="b1493017362449"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname148675510478"><a name="parmname148675510478"></a><a name="parmname148675510478"></a>“filters”</span>需要为非空指针，且长度为n * 6的uint32_t的数组，否则可能出现越界读取的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## setNumProbes接口<a name="ZH-CN_TOPIC_0000001635576089"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void setNumProbes(int nprobes);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>设置检索时总的候选桶数量。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b16217144619214"><a name="b16217144619214"></a><a name="b16217144619214"></a>int nprobes</strong>：AscendIndexIVFSP的nprobe数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p24932033718"><a name="p24932033718"></a><a name="p24932033718"></a><span class="parmname" id="parmname248227153713"><a name="parmname248227153713"></a><a name="parmname248227153713"></a>“nprobes”</span>为16的倍数且符合0 &lt; nprobes ≤ nlist。</p>
</td>
</tr>
</tbody>
</table>

## setVerbose接口<a name="ZH-CN_TOPIC_0000001586055516"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>void setVerbose(bool verbose);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>设置是否显式添加特征向量到底库的进度。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1317750194518"><a name="p1317750194518"></a><a name="p1317750194518"></a><strong id="b14187193523611"><a name="b14187193523611"></a><a name="b14187193523611"></a>bool verbose</strong>：是否显式添加特征向量到底库的进度。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p111317507451"><a name="p111317507451"></a><a name="p111317507451"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1076504458"><a name="p1076504458"></a><a name="p1076504458"></a>无</p>
</td>
</tr>
</tbody>
</table>

## trainCodeBook接口<a name="ZH-CN_TOPIC_0000002148530670"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p108681253358"><a name="p108681253358"></a><a name="p108681253358"></a>void trainCodeBook(const AscendIndexCodeBookInitParams &amp;codeBookInitParams) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p5404143115354"><a name="p5404143115354"></a><a name="p5404143115354"></a>IVFSP码本训练接口。如果训练速度较慢，可能是安装OpenBLAS时限制了使用单线程，可以设置环境变量export OMP_NUM_THREADS=4 进行加速</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12524203743519"><a name="p12524203743519"></a><a name="p12524203743519"></a>const AscendIndexCodeBookInitParams &amp;codeBookInitParams：训练码本所需的初始化参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184441341163411"><a name="p184441341163411"></a><a name="p184441341163411"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1944394111342"><a name="p1944394111342"></a><a name="p1944394111342"></a>参考<a href="../02_approximate_retrieval/13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexCodeBookInitParams接口</a>。</p>
</td>
</tr>
</tbody>
</table>

## addCodeBook接口<a name="ZH-CN_TOPIC_0000002148372594"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p091033816364"><a name="p091033816364"></a><a name="p091033816364"></a>void addCodeBook(const char *codeBookPath);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p16438944153611"><a name="p16438944153611"></a><a name="p16438944153611"></a>添加训练好的码本。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p7964249173612"><a name="p7964249173612"></a><a name="p7964249173612"></a>const char *codeBookPath：码本路径。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p184441341163411"><a name="p184441341163411"></a><a name="p184441341163411"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p198901214595"><a name="p198901214595"></a><a name="p198901214595"></a>“codeBookPath”对应的文件是调用trainCodeBook方法得到的码本文件，程序执行用户对其有读权限；且文件不能为软链接。</p>
</td>
</tr>
</tbody>
</table>

## AscendIndexCodeBookInitParams接口<a name="ZH-CN_TOPIC_0000002183731529"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4159142563912"><a name="p4159142563912"></a><a name="p4159142563912"></a>AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum, std::string codeBookOutputDir, std::string learnDataPath, bool verbose);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p123622413395"><a name="p123622413395"></a><a name="p123622413395"></a>IVFSP训练码本的初始化结构体。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1044544115344"><a name="p1044544115344"></a><a name="p1044544115344"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p644474193416"><a name="p644474193416"></a><a name="p644474193416"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>参数值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p12329155424114"><a name="p12329155424114"></a><a name="p12329155424114"></a><strong id="b1039615564575"><a name="b1039615564575"></a><a name="b1039615564575"></a>int numIter</strong>：训练迭代次数参数，默认为<span class="parmvalue" id="parmvalue15177195954914"><a name="parmvalue15177195954914"></a><a name="parmvalue15177195954914"></a>“1”</span>。</p>
<p id="p632945434111"><a name="p632945434111"></a><a name="p632945434111"></a><strong id="b4257125835714"><a name="b4257125835714"></a><a name="b4257125835714"></a>int device</strong>：设备逻辑ID，默认为<span class="parmvalue" id="parmvalue15961200195018"><a name="parmvalue15961200195018"></a><a name="parmvalue15961200195018"></a>“0”</span>。</p>
<p id="p1032911541415"><a name="p1032911541415"></a><a name="p1032911541415"></a><strong id="b4898195914578"><a name="b4898195914578"></a><a name="b4898195914578"></a>float ratio</strong>：训练用原始样本的采样率，默认为<span class="parmvalue" id="parmvalue1728113155017"><a name="parmvalue1728113155017"></a><a name="parmvalue1728113155017"></a>“1.0”</span>。</p>
<p id="p15329175434116"><a name="p15329175434116"></a><a name="p15329175434116"></a><strong id="b166116165813"><a name="b166116165813"></a><a name="b166116165813"></a>int batchSize</strong>：训练时以batchSize大小执行训练。与<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP训练算子模型文件生成”中的&lt;batch_size&gt;保持一致，默认值为“32768”。</p>
<p id="p9329185464116"><a name="p9329185464116"></a><a name="p9329185464116"></a><strong id="b11743147185810"><a name="b11743147185810"></a><a name="b11743147185810"></a>int codeNum</strong>：每次最大按codeNum样本数量操作码本，必须为2的幂次。与<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节的“IVFSP训练算子模型文件生成”中的&lt;codebook_batch_size&gt;保持一致，默认为<span class="parmvalue" id="parmvalue2730106115111"><a name="parmvalue2730106115111"></a><a name="parmvalue2730106115111"></a>“32768”</span>。</p>
<p id="p1232955415418"><a name="p1232955415418"></a><a name="p1232955415418"></a><strong id="b596299165819"><a name="b596299165819"></a><a name="b596299165819"></a>std::string codeBookOutputDir</strong>：生成的码本文件输出到的目录，用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。</p>
<p id="p163291154204114"><a name="p163291154204114"></a><a name="p163291154204114"></a><strong id="b1851871235817"><a name="b1851871235817"></a><a name="b1851871235817"></a>std::string learnDataPath</strong>：训练用的原始特征文件路径，支持bin、npy格式，bin存储方式为行优先，数据类型为<span class="parmvalue" id="parmvalue31381046194918"><a name="parmvalue31381046194918"></a><a name="parmvalue31381046194918"></a>“float32”</span>。</p>
<p id="p103292545418"><a name="p103292545418"></a><a name="p103292545418"></a><strong id="b1032851425818"><a name="b1032851425818"></a><a name="b1032851425818"></a>bool verbose</strong>：是否开启额外打印信息，默认为<span class="parmname" id="parmname55226406496"><a name="parmname55226406496"></a><a name="parmname55226406496"></a>“true”</span>。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1076619519437"></a><a name="ul1076619519437"></a><ul id="ul1076619519437"><li>numIter∈ (0, 20]。</li><li>ratio∈ (0, 1.0]。</li><li>batchSize∈ (0, 32768]。</li><li>codeNum∈ (0, 32768]。</li><li>当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</li><li>在执行训练生成码本前，请先参考<a href="../../05_user_guide.md#ivfsp">IVFSP</a>生成训练算子模型文件。</li></ul>
</td>
</tr>
</tbody>
</table>

## trainCodeBookFromMem接口<a name="ZH-CN_TOPIC_0000002257319034"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p4398205434419"><a name="p4398205434419"></a><a name="p4398205434419"></a>void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>IVFSP码本训练接口。训练数据从内存中加载，如果训练速度较慢，可能是安装OpenBLAS时限制了使用单线程，可以设置环境变量export OMP_NUM_THREADS=4进行加速。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>const AscendIndexCodeBookInitFromMemParams &amp;codeBookInitFromMemParams：训练码本所需的初始化参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11397185416444"><a name="p11397185416444"></a><a name="p11397185416444"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p8383354134415"><a name="p8383354134415"></a><a name="p8383354134415"></a>了解AscendIndexCodeBookInitFromMemParams相关说明，请参见<a href="#ascendindexcodebookinitfrommemparams接口">AscendIndexCodeBookInitFromMemParams</a>。</p>
</td>
</tr>
</tbody>
</table>

## AscendIndexCodeBookInitFromMemParams接口<a name="ZH-CN_TOPIC_0000002291969193"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p7524719186"><a name="p7524719186"></a><a name="p7524719186"></a>AscendIndexCodeBookInitFromMemParams (int numIter, int device, float ratio, int batchSize, int codeNum,bool verbose,std::string codeBookOutputDir,const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>IVFSP训练码本的初始化结构体。从内存中加载训练数据。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>参数值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><strong id="b3354181515167"><a name="b3354181515167"></a><a name="b3354181515167"></a>int numIter：</strong>训练迭代次数参数，默认为“1”。</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><strong id="b13953183012163"><a name="b13953183012163"></a><a name="b13953183012163"></a>int device：</strong>设备逻辑ID，默认为“0”。</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><strong id="b229253513169"><a name="b229253513169"></a><a name="b229253513169"></a>float ratio：</strong>训练用原始样本的采样率，默认为“1.0”。</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><strong id="b1712474051618"><a name="b1712474051618"></a><a name="b1712474051618"></a>int batchSize：</strong>训练时以batchSize大小执行训练。与<a href="../../05_user_guide.md#ivfsp">IVFSP训练算子模型文件生成</a>中的&lt;batch_size&gt;保持一致，要求大于“0”，默认值为“32768”。</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><strong id="b76015463165"><a name="b76015463165"></a><a name="b76015463165"></a>int codeNum：</strong>每次最大按codeNum样本数量操作码本，必须为2的幂次。与<a href="../../05_user_guide.md#ivfsp">IVFSP训练算子模型文件生成</a>中的&lt;codebook_batch_size&gt;保持一致，要求大于0，默认为“32768”。</p>
<p id="p16645243598"><a name="p16645243598"></a><a name="p16645243598"></a><strong id="b16826155511614"><a name="b16826155511614"></a><a name="b16826155511614"></a>std::string codeBookOutputDir：</strong>生成的码本文件输出到的目录。用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。</p>
<p id="p864518435917"><a name="p864518435917"></a><a name="p864518435917"></a><strong id="b172846021716"><a name="b172846021716"></a><a name="b172846021716"></a>bool verbose：</strong>是否开启额外打印信息，默认为“true”。</p>
<p id="p146455433910"><a name="p146455433910"></a><a name="p146455433910"></a><strong id="b049205121720"><a name="b049205121720"></a><a name="b049205121720"></a>const float *memLearnData：</strong>内存中数据指针，默认为空指针。</p>
<p id="p1864544319911"><a name="p1864544319911"></a><a name="p1864544319911"></a><strong id="b3740189191719"><a name="b3740189191719"></a><a name="b3740189191719"></a>size_t memLearnDataSize：</strong>内存中数据长度，默认为0。</p>
<p id="p106451943193"><a name="p106451943193"></a><a name="p106451943193"></a><strong id="b28410155178"><a name="b28410155178"></a><a name="b28410155178"></a>bool isTrainAndAdd：</strong>是否训练码本后直接添加到Index开关，默认为false。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul129009408302"></a><a name="ul129009408302"></a><ul id="ul129009408302"><li>numIter∈ (0, 20]</li><li>ratio∈ (0, 1.0]</li><li>memLearnDataSize % dim == 0</li><li>memLearnDataSize≤25G</li></ul>
<a name="ul154204603015"></a><a name="ul154204603015"></a><ul id="ul154204603015"><li>当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</li><li>在执行训练生成码本前，请先参考<a href="../../05_user_guide.md#ivfsp">IVFSP</a>章节生成训练算子模型文件。</li></ul>
<a name="ul547975410309"></a><a name="ul547975410309"></a><ul id="ul547975410309"><li>当isTrainAndAdd为true时，码本训练好之后直接添加到Index中，不会进行落盘；</li><li>当isTrainAndAdd为false时，码本会保存到codeBookOutputDir路径下，需调用addCodeBook手动添加。</li><li>memLearnDataSize为指针memLearnData的真实长度，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>
