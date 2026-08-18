# AscendIndexIVFSQ<a name="ZH-CN_TOPIC_0000001506334625"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456694964"></a>

AscendIndexIVFSQ利用IVF来进行加速，是二级近似检索算法。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AscendIndexIVFSQ接口<a name="ZH-CN_TOPIC_0000001506414893"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a>AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQ的构造函数，基于一个已有的index创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b1580317419509"><a name="b1580317419509"></a><a name="b1580317419509"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧资源配置。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b89210445502"><a name="b89210445502"></a><a name="b89210445502"></a>AscendIndexIVFSQConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname12160171235212"><a name="parmname12160171235212"></a><a name="parmname12160171235212"></a>“index”</span>需要为合法有效的CPU Index指针。</p>
</td>
</tr>
</tbody>
</table>

<a name="table1823217151014"></a>
<table><tbody><tr id="row178231617161011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1882331711020"><a name="p1882331711020"></a><a name="p1882331711020"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p18649496410"><a name="p18649496410"></a><a name="p18649496410"></a>AscendIndexIVFSQ(int dims, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</p>
</td>
</tr>
<tr id="row8823317171017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p7823617131019"><a name="p7823617131019"></a><a name="p7823617131019"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18823117111010"><a name="p18823117111010"></a><a name="p18823117111010"></a>AscendIndexIVFSQ的构造函数，生成AscendIndexIVFSQ，此时根据<span class="parmname" id="parmname16824101217816"><a name="parmname16824101217816"></a><a name="parmname16824101217816"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row1582381741012"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128231617131015"><a name="p128231617131015"></a><a name="p128231617131015"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b342242917528"><a name="b342242917528"></a><a name="b342242917528"></a>int dims</strong>：AscendIndexIVFSQ管理的一组特征向量的维度。</p>
<p id="p169755411358"><a name="p169755411358"></a><a name="p169755411358"></a><strong id="b050293135214"><a name="b050293135214"></a><a name="b050293135214"></a>int nlist</strong>：聚类中心的个数，与算子生成脚本中的<span class="parmname" id="parmname1915820151081"><a name="parmname1915820151081"></a><a name="parmname1915820151081"></a>“coarse_centroid_num”</span>参数对应。</p>
<p id="p895114473339"><a name="p895114473339"></a><a name="p895114473339"></a><strong id="b42321338105211"><a name="b42321338105211"></a><a name="b42321338105211"></a>faiss::ScalarQuantizer::QuantizerType qtype</strong>：AscendIndexIVFSQ的量化器类型。</p>
<p id="p7823317181017"><a name="p7823317181017"></a><a name="p7823317181017"></a><strong id="b11282104020522"><a name="b11282104020522"></a><a name="b11282104020522"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p5823115014619"><a name="p5823115014619"></a><a name="p5823115014619"></a><strong id="b15262643135212"><a name="b15262643135212"></a><a name="b15262643135212"></a>bool encodeResidual</strong>：表示是否对残差编码。</p>
<p id="p168231017101016"><a name="p168231017101016"></a><a name="p168231017101016"></a><strong id="b1821144512529"><a name="b1821144512529"></a><a name="b1821144512529"></a>AscendIndexIVFSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row168231917191016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p6824121714106"><a name="p6824121714106"></a><a name="p6824121714106"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p138241317201019"><a name="p138241317201019"></a><a name="p138241317201019"></a>无</p>
</td>
</tr>
<tr id="row10824101711014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p68241317131018"><a name="p68241317131018"></a><a name="p68241317131018"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p682420176103"><a name="p682420176103"></a><a name="p682420176103"></a>无</p>
</td>
</tr>
<tr id="row5824161731013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p982431701017"><a name="p982431701017"></a><a name="p982431701017"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3234195217524"></a><a name="ul3234195217524"></a><ul id="ul3234195217524"><li>dims ∈ {64, 128, 256, 384, 512}</li><li>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>qtype = ScalarQuantizer::QuantizerType::QT_8bit，当前仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}<div class="note" id="note123188311292"><a name="note123188311292"></a><a name="note123188311292"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p2318163115919"><a name="p2318163115919"></a><a name="p2318163115919"></a>当前<span class="parmname" id="parmname1831820318915"><a name="parmname1831820318915"></a><a name="parmname1831820318915"></a>“encodeResidual”</span>在<span class="parmvalue" id="parmvalue1631933115916"><a name="parmvalue1631933115916"></a><a name="parmvalue1631933115916"></a>“metric=faiss::MetricType::METRIC_INNER_PRODUCT”</span>下，仅支持<span class="parmvalue" id="parmvalue153191931795"><a name="parmvalue153191931795"></a><a name="parmvalue153191931795"></a>“false”</span>取值，即当前并不支持对残差编码的IVFSQ方法，当取值为<span class="parmvalue" id="parmvalue1531915311293"><a name="parmvalue1531915311293"></a><a name="parmvalue1531915311293"></a>“true”</span>时能够运行成功但存在精度问题。</p>
</div></div>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table134501935171012"></a>
<table><tbody><tr id="row11451103521010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p44511935121011"><a name="p44511935121011"></a><a name="p44511935121011"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1445153561020"><a name="p1445153561020"></a><a name="p1445153561020"></a>AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);</p>
</td>
</tr>
<tr id="row1945123511015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p04512353102"><a name="p04512353102"></a><a name="p04512353102"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p8451173581017"><a name="p8451173581017"></a><a name="p8451173581017"></a>AscendIndexIVFSQ的构造函数，生成AscendIndexIVFSQ，此时根据<span class="parmname" id="parmname445173519102"><a name="parmname445173519102"></a><a name="parmname445173519102"></a>“config”</span>中配置的值设置Device侧资源。此接口不执行初始化，由子类执行初始化相关功能，后续会废弃此接口，请勿使用。</p>
</td>
</tr>
<tr id="row1645163571015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p74511835171017"><a name="p74511835171017"></a><a name="p74511835171017"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p64511935141018"><a name="p64511935141018"></a><a name="p64511935141018"></a><strong id="b54513358104"><a name="b54513358104"></a><a name="b54513358104"></a>int dims</strong>：AscendIndexIVFSQ管理的一组特征向量的维度。</p>
<p id="p13451183517103"><a name="p13451183517103"></a><a name="p13451183517103"></a><strong id="b1645153519106"><a name="b1645153519106"></a><a name="b1645153519106"></a>int nlist</strong>：聚类中心的个数，与算子生成脚本中的<span class="parmname" id="parmname1645193510102"><a name="parmname1645193510102"></a><a name="parmname1645193510102"></a>“coarse_centroid_num”</span>参数对应。</p>
<p id="p10451153591010"><a name="p10451153591010"></a><a name="p10451153591010"></a><strong id="b14451535111016"><a name="b14451535111016"></a><a name="b14451535111016"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p54513357103"><a name="p54513357103"></a><a name="p54513357103"></a><strong id="b16451435121012"><a name="b16451435121012"></a><a name="b16451435121012"></a>AscendIndexIVFSQConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row8451113510107"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p5451123541012"><a name="p5451123541012"></a><a name="p5451123541012"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p12451435121015"><a name="p12451435121015"></a><a name="p12451435121015"></a>无</p>
</td>
</tr>
<tr id="row194511735181010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p17451163513101"><a name="p17451163513101"></a><a name="p17451163513101"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p5451203551020"><a name="p5451203551020"></a><a name="p5451203551020"></a>无</p>
</td>
</tr>
<tr id="row1945183511016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1345123518101"><a name="p1345123518101"></a><a name="p1345123518101"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul15452103551014"></a><a name="ul15452103551014"></a><ul id="ul15452103551014"><li>dims ∈ {64, 128, 256, 384, 512}</li><li>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table663150151113"></a>
<table><tbody><tr id="row176440181111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p064509114"><a name="p064509114"></a><a name="p064509114"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p885213589106"><a name="p885213589106"></a><a name="p885213589106"></a>AscendIndexIVFSQ(const AscendIndexIVFSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row186417021110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p664405110"><a name="p664405110"></a><a name="p664405110"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p76470121111"><a name="p76470121111"></a><a name="p76470121111"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row964505113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p2642019118"><a name="p2642019118"></a><a name="p2642019118"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b1399954415513"><a name="b1399954415513"></a><a name="b1399954415513"></a>const AscendIndexIVFSQ&amp;</strong>：常量AscendIndexIVFSQ。</p>
</td>
</tr>
<tr id="row8641601111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p13648019116"><a name="p13648019116"></a><a name="p13648019116"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p06420013110"><a name="p06420013110"></a><a name="p06420013110"></a>无</p>
</td>
</tr>
<tr id="row1641608114"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p96418010111"><a name="p96418010111"></a><a name="p96418010111"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1264107115"><a name="p1264107115"></a><a name="p1264107115"></a>无</p>
</td>
</tr>
<tr id="row176420181110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p126412017119"><a name="p126412017119"></a><a name="p126412017119"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p16647010117"><a name="p16647010117"></a><a name="p16647010117"></a>无</p>
</td>
</tr>
</tbody>
</table>

## \~AscendIndexIVFSQ接口<a name="ZH-CN_TOPIC_0000001456534936"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexIVFSQ();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQ的析构函数，销毁AscendIndexIVFSQ对象，释放资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
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

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456375244"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1215384082314"><a name="p1215384082314"></a><a name="p1215384082314"></a>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQ基于一个已有的index拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b2023965517563"><a name="b2023965517563"></a><a name="b2023965517563"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1346620304216"><a name="p1346620304216"></a><a name="p1346620304216"></a><span class="parmname" id="parmname13234938125"><a name="parmname13234938125"></a><a name="parmname13234938125"></a>“index”</span>需要为合法有效的CPU Index指针，Index的维度d参数取值范围为{64, 128, 256, 384, 512}，</p>
<p id="p74662030922"><a name="p74662030922"></a><a name="p74662030922"></a>Index的聚类中心的个数nlist参数取值范围{1024, 2048, 4096, 8192, 16384, 32768}</p>
<p id="p154661430523"><a name="p154661430523"></a><a name="p154661430523"></a>总的候选桶数量nprobe的取值范围0 &lt; nprobe ≤ nlist</p>
<p id="p646610301721"><a name="p646610301721"></a><a name="p646610301721"></a>底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</p>
<p id="p64663301228"><a name="p64663301228"></a><a name="p64663301228"></a>sq.qtype参数仅支持“ScalarQuantizer::QuantizerType::QT_8bit”。</p>
</td>
</tr>
</tbody>
</table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001506334649"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexIVFScalarQuantizer *index) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1720318284418"><a name="p1720318284418"></a><a name="p1720318284418"></a>将AscendIndexIVFSQ的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b750912215531"><a name="b750912215531"></a><a name="b750912215531"></a>faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname95661875712"><a name="parmname95661875712"></a><a name="parmname95661875712"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001456854860"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a>AscendIndexIVFSQ&amp; operator=(const AscendIndexIVFSQ&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b115570395614"><a name="b115570395614"></a><a name="b115570395614"></a>const AscendIndexIVFSQ&amp;</strong>：常量AscendIndexIVFSQ。</p>
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

## train接口<a name="ZH-CN_TOPIC_0000001456854976"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void train(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对AscendIndexIVFSQ执行训练，继承AscendIndex中的相关接口并提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b351832435710"><a name="b351832435710"></a><a name="b351832435710"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b17199113075712"><a name="b17199113075712"></a><a name="b17199113075712"></a>const float *x</strong>：特征向量数据。</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul777123515576"></a><a name="ul777123515576"></a><ul id="ul777123515576"><li>训练采用k-means进行聚类，训练集比较小可能会影响查询精度。</li><li>此处<span class="parmname" id="parmname125783489316"><a name="parmname125783489316"></a><a name="parmname125783489316"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname95481642105713"><a name="parmname95481642105713"></a><a name="parmname95481642105713"></a>“x”</span>需要为非空指针，且长度应该为dims * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>
