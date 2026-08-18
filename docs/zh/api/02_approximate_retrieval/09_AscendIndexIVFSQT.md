# AscendIndexIVFSQT<a name="ZH-CN_TOPIC_0000001456375224"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506615005"></a>

AscendIndexIVFSQT类，包含降维算法的三级检索IVFSQ算法，需要传入两个参数指明降维前后的维度信息，要求降维后维度能整除降维前的维度。适用于1000万级底库的场景。

需要按照IVFSQT算子生成方式，生成三级检索所需算子。

该类型带有模糊聚类功能：入桶前，使用threshold参数控制模糊程度。请根据底库容量和可用内存大小设置threshold参数值，过大的threshold会引起内存不足，导致失败。<term>Atlas 200/300/500 推理产品</term>环境建议设置\[1.0, 1.1\]，<term>Atlas 推理系列产品</term>环境建议设置\[1.0, 1.5\]。搜索时建议使用**batch size = 65536**。

使用流程为：1.构建index对象；2.train数据；3.add数据；4.update数据；5.search检索数据；6.析构index对象。update后不支持继续add数据。有新数据需要进行检索时，请将原来的index对象析构后，重新按照流程使用。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## AscendIndexIVFSQT接口<a name="ZH-CN_TOPIC_0000001506495685"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p37041120111120"><a name="p37041120111120"></a><a name="p37041120111120"></a>AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQT的构造函数，基于一个已有的<span class="parmname" id="parmname102792478176"><a name="parmname102792478176"></a><a name="parmname102792478176"></a>“index”</span>创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p185955304554"><a name="p185955304554"></a><a name="p185955304554"></a><strong id="b46201291626"><a name="b46201291626"></a><a name="b46201291626"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧的Index资源。</p>
<p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b159906314214"><a name="b159906314214"></a><a name="b159906314214"></a>AscendIndexIVFSQTConfig config</strong>：Device侧资源配置。</p>
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
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1113653515219"></a><a name="ul1113653515219"></a><ul id="ul1113653515219"><li><span class="parmname" id="parmname173302381721"><a name="parmname173302381721"></a><a name="parmname173302381721"></a>“index”</span>需要为合法有效的CPU Index指针。</li><li>index-&gt;d ∈ {256}。</li><li>index-&gt;sq.d ∈ {32, 64, 128}。</li><li><span class="parmname" id="parmname144451835155310"><a name="parmname144451835155310"></a><a name="parmname144451835155310"></a>“index”</span>的维度必须大于index-&gt;sq的维度且可以被index-&gt;sq的维度整除。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table124585216195"></a>
<table><tbody><tr id="row164575271917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p134518527191"><a name="p134518527191"></a><a name="p134518527191"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p18649496410"><a name="p18649496410"></a><a name="p18649496410"></a>AscendIndexIVFSQT(int dimIn, int dimOut, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());</p>
</td>
</tr>
<tr id="row1045152101914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1245185213193"><a name="p1245185213193"></a><a name="p1245185213193"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1845115251915"><a name="p1845115251915"></a><a name="p1845115251915"></a>AscendIndexIVFSQT的构造函数，生成AscendIndexIVFSQT，此时根据<span class="parmname" id="parmname74506566178"><a name="parmname74506566178"></a><a name="parmname74506566178"></a>“config”</span>中配置的值设置Device侧资源。</p>
</td>
</tr>
<tr id="row16451352141910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p204585213198"><a name="p204585213198"></a><a name="p204585213198"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b162991621040"><a name="b162991621040"></a><a name="b162991621040"></a>int dimIn</strong>：AscendIndexIVFSQT管理的一组原始特征向量的维度。</p>
<p id="p055565814416"><a name="p055565814416"></a><a name="p055565814416"></a><strong id="b18128104346"><a name="b18128104346"></a><a name="b18128104346"></a>int dimOut</strong>：AscendIndexIVFSQT管理的一组降维目标特征向量的维度。</p>
<p id="p169755411358"><a name="p169755411358"></a><a name="p169755411358"></a><strong id="b11908056418"><a name="b11908056418"></a><a name="b11908056418"></a>int nlist</strong>：聚类中心的个数，与算子生成脚本中的<span class="parmname" id="parmname45896241598"><a name="parmname45896241598"></a><a name="parmname45896241598"></a>“coarse_centroid_num”</span>参数对应。</p>
<p id="p895114473339"><a name="p895114473339"></a><a name="p895114473339"></a><strong id="b5178209448"><a name="b5178209448"></a><a name="b5178209448"></a>faiss::ScalarQuantizer::QuantizerType qtype</strong>：AscendIndexIVFSQT的量化器类型。</p>
<p id="p174585217192"><a name="p174585217192"></a><a name="p174585217192"></a><strong id="b27578121644"><a name="b27578121644"></a><a name="b27578121644"></a>faiss::MetricType metric</strong>：AscendIndex在执行特征向量相似度检索的时候使用的距离度量类型。</p>
<p id="p345135213199"><a name="p345135213199"></a><a name="p345135213199"></a><strong id="b155089152412"><a name="b155089152412"></a><a name="b155089152412"></a>AscendIndexIVFSQTConfig config</strong>：Device侧资源配置。</p>
</td>
</tr>
<tr id="row19459527195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p94525214193"><a name="p94525214193"></a><a name="p94525214193"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p204520521198"><a name="p204520521198"></a><a name="p204520521198"></a>无</p>
</td>
</tr>
<tr id="row10451352191917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p44514527191"><a name="p44514527191"></a><a name="p44514527191"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p17451152151917"><a name="p17451152151917"></a><a name="p17451152151917"></a>无</p>
</td>
</tr>
<tr id="row154545211199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15456526194"><a name="p15456526194"></a><a name="p15456526194"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul3942173318420"></a><a name="ul3942173318420"></a><ul id="ul3942173318420"><li>dimIn ∈ {256}。</li><li>dimOut ∈ {32, 64, 128}。</li><li>nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}。</li><li>qtype = ScalarQuantizer::QuantizerType::QT_8bit，当前仅支持<span class="parmvalue" id="parmvalue35921391642"><a name="parmvalue35921391642"></a><a name="parmvalue35921391642"></a>“ScalarQuantizer::QuantizerType::QT_8bit”</span>量化器类型。</li><li>metric = faiss::MetricType::METRIC_INNER_PRODUCT （当前仅支持 <span class="parmvalue" id="parmvalue20208125320410"><a name="parmvalue20208125320410"></a><a name="parmvalue20208125320410"></a>“faiss::MetricType::METRIC_INNER_PRODUCT”</span>。）</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table68594118203"></a>
<table><tbody><tr id="row12859818205"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1885911122013"><a name="p1885911122013"></a><a name="p1885911122013"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p885213589106"><a name="p885213589106"></a><a name="p885213589106"></a>AscendIndexIVFSQT(const AscendIndexIVFSQT&amp;) = delete;</p>
</td>
</tr>
<tr id="row158592122017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p68591616209"><a name="p68591616209"></a><a name="p68591616209"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p78592113207"><a name="p78592113207"></a><a name="p78592113207"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row18859201122014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p18859141122017"><a name="p18859141122017"></a><a name="p18859141122017"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b3710328459"><a name="b3710328459"></a><a name="b3710328459"></a>const AscendIndexIVFSQT&amp;</strong>：AscendIndexIVFSQT对象。</p>
</td>
</tr>
<tr id="row28605142020"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p286001112012"><a name="p286001112012"></a><a name="p286001112012"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16860017208"><a name="p16860017208"></a><a name="p16860017208"></a>无</p>
</td>
</tr>
<tr id="row1186001132017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1586020118202"><a name="p1586020118202"></a><a name="p1586020118202"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p886015192016"><a name="p886015192016"></a><a name="p886015192016"></a>无</p>
</td>
</tr>
<tr id="row38604142015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1586012112011"><a name="p1586012112011"></a><a name="p1586012112011"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## \~AscendIndexIVFSQT接口<a name="ZH-CN_TOPIC_0000001456854984"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>virtual ~AscendIndexIVFSQT();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVFSQT的析构函数，销毁AscendIndexIVFSQT对象，释放资源。</p>
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

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456695060"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1215384082314"><a name="p1215384082314"></a><a name="p1215384082314"></a>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexIVSQT基于一个已有的<span class="parmname" id="parmname4930151917214"><a name="parmname4930151917214"></a><a name="parmname4930151917214"></a>“index”</span>拷贝到Ascend，并保持原有的AscendIndex的Device侧资源配置。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b9345217864"><a name="b9345217864"></a><a name="b9345217864"></a>const faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname1430942010615"><a name="parmname1430942010615"></a><a name="parmname1430942010615"></a>“index”</span>需要为合法有效的CPU Index指针。</p>
<a name="ul1113653515219"></a><a name="ul1113653515219"></a><ul id="ul1113653515219"><li>index-&gt;d ∈ {256}。</li><li>index-&gt;sq.d ∈ {32, 64, 128}。</li><li><span class="parmname" id="parmname144451835155310"><a name="parmname144451835155310"></a><a name="parmname144451835155310"></a>“index”</span>的维度必须大于index-&gt;sq的维度，且可以被index-&gt;sq的维度整除。</li><li>update过的对象请勿调用该接口。</li></ul>
</td>
</tr>
</tbody>
</table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001506495825"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p10713954155218"><a name="p10713954155218"></a><a name="p10713954155218"></a>void copyTo(faiss::IndexIVFScalarQuantizer *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1720318284418"><a name="p1720318284418"></a><a name="p1720318284418"></a>将AscendIndexIVFSQT的检索资源拷贝到CPU侧。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b158351007610"><a name="b158351007610"></a><a name="b158351007610"></a>faiss::IndexIVFScalarQuantizer *index</strong>：CPU侧Index资源。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a><span class="parmname" id="parmname99881731766"><a name="parmname99881731766"></a><a name="parmname99881731766"></a>“index”</span>需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</p>
</td>
</tr>
</tbody>
</table>

## fineTune接口<a name="ZH-CN_TOPIC_0000001456694860"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p193348812010"><a name="p193348812010"></a><a name="p193348812010"></a>void fineTune(size_t n, const float *x);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对中心进行微调和优化，避免分桶不均匀的问题。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p492321510014"><a name="p492321510014"></a><a name="p492321510014"></a><strong id="b101231056175619"><a name="b101231056175619"></a><a name="b101231056175619"></a>size_t n</strong>：特征向量的条数。</p>
<p id="p12314314436"><a name="p12314314436"></a><a name="p12314314436"></a><strong id="b3709358115620"><a name="b3709358115620"></a><a name="b3709358115620"></a>const float *x</strong>：特征向量数据。</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p1529214384393"><a name="p1529214384393"></a><a name="p1529214384393"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## getFuzzyK接口<a name="ZH-CN_TOPIC_0000001456855008"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>int getFuzzyK() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>获取入桶时每个向量的最大值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b8278171308"><a name="b8278171308"></a><a name="b8278171308"></a>int</strong>：每个向量入桶时的最大值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getListCodesAndIds接口<a name="ZH-CN_TOPIC_0000001687739112"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>void getListCodesAndIds(int listId, std::vector&lt;uint8_t&gt;&amp; codes, std::vector&lt;ascend_idx_t&gt;&amp; ids) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量和对应ID。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b234205219283"><a name="b234205219283"></a><a name="b234205219283"></a>int listId</strong>：AscendIndexIVFSQT的nlist中的特定nlistId。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p812472610226"><a name="p812472610226"></a><a name="p812472610226"></a><strong id="b8752144372820"><a name="b8752144372820"></a><a name="b8752144372820"></a>std::vector&lt;uint8_t&gt;&amp; codes</strong>：AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量。</p>
<p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a><strong id="b198817462287"><a name="b198817462287"></a><a name="b198817462287"></a>std::vector&lt;ascend_idx_t&gt;&amp; ids</strong>：AscendIndexIVFSQT的nlist中的特定nlistId上的特征向量ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p13621611141120"><a name="p13621611141120"></a><a name="p13621611141120"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## getListLength接口<a name="ZH-CN_TOPIC_0000001735977797"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>uint32_t getListLength(int listId) const override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>返回当前的AscendIndexIVFSQT的nlist中的特定nlistId上的长度。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b121461446192713"><a name="b121461446192713"></a><a name="b121461446192713"></a>int listId</strong>：AscendIndexIVFSQT的nlist中的特定nlistId。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>AscendIndexIVFSQT的nlist中的特定nlistId上的长度。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p13621611141120"><a name="p13621611141120"></a><a name="p13621611141120"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## getLowerBound接口<a name="ZH-CN_TOPIC_0000001506614885"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p139751803263"><a name="p139751803263"></a><a name="p139751803263"></a>int getLowerBound() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>返回二级分簇的阈值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p192391322172611"><a name="p192391322172611"></a><a name="p192391322172611"></a>二级分簇的阈值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getMergeThres接口<a name="ZH-CN_TOPIC_0000001506615073"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p898555319557"><a name="p898555319557"></a><a name="p898555319557"></a>int getMergeThres() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p476310541227"><a name="p476310541227"></a><a name="p476310541227"></a>获取合并子桶阈值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>合并子桶阈值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getQMax接口<a name="ZH-CN_TOPIC_0000001456535208"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>float getQMax() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>返回特征向量的最大值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>特征向量的最大值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getQMin接口<a name="ZH-CN_TOPIC_0000001506615029"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p114441322513"><a name="p114441322513"></a><a name="p114441322513"></a>float getQMin() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p810264716532"><a name="p810264716532"></a><a name="p810264716532"></a>返回特征向量的最小值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>特征向量的最小值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getThreshold接口<a name="ZH-CN_TOPIC_0000001506334633"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>float getThreshold() const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>获取判断向量是否入多个桶的阈值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b4330591711"><a name="b4330591711"></a><a name="b4330591711"></a>float</strong>：判断向量是否入多个桶的阈值。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>无</p>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506615085"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p11970183910121"><a name="p11970183910121"></a><a name="p11970183910121"></a>AscendIndexIVFSQT&amp; operator=(const AscendIndexIVFSQT&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b0567942255"><a name="b0567942255"></a><a name="b0567942255"></a>const AscendIndexIVFSQT&amp;</strong>：AscendIndexIVFSQT对象。</p>
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

## remove\_ids接口<a name="ZH-CN_TOPIC_0000001506615053"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>size_t remove_ids(const faiss::IDSelector &amp;sel) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>根据ID删除底库特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b8255123114917"><a name="b8255123114917"></a><a name="b8255123114917"></a>const faiss::IDSelector &amp;sel</strong>：待删除的特征向量，具体用法和定义请参考对应的Faiss中的相关说明。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.5.1 "><p id="p1866035143510"><a name="p1866035143510"></a><a name="p1866035143510"></a>返回被删除的特征向量数量。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.86%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.14%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>当前版本暂不支持该接口。</p>
</td>
</tr>
</tbody>
</table>

## reset接口<a name="ZH-CN_TOPIC_0000001506334789"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>void reset() override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p10290194315362"><a name="p10290194315362"></a><a name="p10290194315362"></a>重置索引，特征数据清零。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1414182616338"><a name="p1414182616338"></a><a name="p1414182616338"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>调用后请勿继续使用该对象。</p>
</td>
</tr>
</tbody>
</table>

## setAddTotal接口<a name="ZH-CN_TOPIC_0000001456375316"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p11341163171010"><a name="p11341163171010"></a><a name="p11341163171010"></a>void setAddTotal(size_t addTotal);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p19691924201019"><a name="p19691924201019"></a><a name="p19691924201019"></a>设置待添加的底库向量总数，默认值<span class="parmvalue" id="parmvalue199423381138"><a name="parmvalue199423381138"></a><a name="parmvalue199423381138"></a>“100000000”</span>。需要先设置<span class="parmname" id="parmname1710103964710"><a name="parmname1710103964710"></a><a name="parmname1710103964710"></a>“PreciseMemControl”</span>为<span class="parmvalue" id="parmvalue185542312415"><a name="parmvalue185542312415"></a><a name="parmvalue185542312415"></a>“true”</span>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b999816351837"><a name="b999816351837"></a><a name="b999816351837"></a>size_t addTotal</strong>：待添加的底库向量总数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p8927121410219"><a name="p8927121410219"></a><a name="p8927121410219"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## setFuzzyK接口<a name="ZH-CN_TOPIC_0000001456534940"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>void setFuzzyK(int value);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>设置入桶时每个向量的最大值。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a><strong id="b144117156596"><a name="b144117156596"></a><a name="b144117156596"></a>int value</strong>：每个向量入桶时的最大值，建议固定为默认值<span class="parmvalue" id="parmvalue19251921185917"><a name="parmvalue19251921185917"></a><a name="parmvalue19251921185917"></a>“3”</span>。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p64091154105412"><a name="p64091154105412"></a><a name="p64091154105412"></a>value的取值范围是（0,10]。</p>
</td>
</tr>
</tbody>
</table>

## setLowerBound接口<a name="ZH-CN_TOPIC_0000001506334777"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p620411196166"><a name="p620411196166"></a><a name="p620411196166"></a>void setLowerBound(int lowerBound);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p6359132829"><a name="p6359132829"></a><a name="p6359132829"></a>设置二级分簇的阈值，默认值为<span class="parmvalue" id="parmvalue1967655810458"><a name="parmvalue1967655810458"></a><a name="parmvalue1967655810458"></a>“32”</span>。</p>
<p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>若一级分簇桶中元素大于lowerBound则进行二次分簇，否则保留原状。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b173513119118"><a name="b173513119118"></a><a name="b173513119118"></a>int lowerBound</strong>：二级分簇的阈值。</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p7518112174013"><a name="p7518112174013"></a><a name="p7518112174013"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## setMemoryLimit接口<a name="ZH-CN_TOPIC_0000001506614917"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p388113312502"><a name="p388113312502"></a><a name="p388113312502"></a>void setMemoryLimit(float memoryLimit);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>设置Host内存限制，默认值为<span class="parmvalue" id="parmvalue1967655810458"><a name="parmvalue1967655810458"></a><a name="parmvalue1967655810458"></a>“32”</span>，单位“GB”。需要先设置<span class="parmname" id="parmname1710103964710"><a name="parmname1710103964710"></a><a name="parmname1710103964710"></a>“PreciseMemControl”</span>为<span class="parmvalue" id="parmvalue185542312415"><a name="parmvalue185542312415"></a><a name="parmvalue185542312415"></a>“true”</span>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b177901237145615"><a name="b177901237145615"></a><a name="b177901237145615"></a>float memoryLimit</strong>：内存限制。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p8927121410219"><a name="p8927121410219"></a><a name="p8927121410219"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## setMergeThres接口<a name="ZH-CN_TOPIC_0000001456694900"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1337496184"><a name="p1337496184"></a><a name="p1337496184"></a>void setMergeThres(int mergeThres);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p476310541227"><a name="p476310541227"></a><a name="p476310541227"></a>设置合并子桶阈值，默认值为<span class="parmvalue" id="parmvalue1967655810458"><a name="parmvalue1967655810458"></a><a name="parmvalue1967655810458"></a>“5”</span>。</p>
<p id="p1703435181217"><a name="p1703435181217"></a><a name="p1703435181217"></a>若二级分簇后某子桶中元素小于mergeThres，则合并该子桶元素至其他子桶中。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b117388471122"><a name="b117388471122"></a><a name="b117388471122"></a>int mergeThres</strong>：合并子桶阈值。</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p276517505390"><a name="p276517505390"></a><a name="p276517505390"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## setNumProbes接口<a name="ZH-CN_TOPIC_0000001736410013"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p132681218211"><a name="p132681218211"></a><a name="p132681218211"></a>void setNumProbes(int nprobes) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>设置当前的AscendIndexIVFSQT的nprobe数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b16217144619214"><a name="b16217144619214"></a><a name="b16217144619214"></a>int nprobes</strong>：AscendIndexIVFSQT的nprobe数。建议保持为默认值<span class="parmvalue" id="parmvalue1410214422510"><a name="parmvalue1410214422510"></a><a name="parmvalue1410214422510"></a>“64”</span>。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li>nprobes ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }</li><li>l2Probe ≥ nprobes, l2Probe≤ l3SegmentNum, l2Probe≤ nprobes * 64</li><li>l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }</li><li>l2Probe和l3SegmentNum的设置可参见<a href="#setsearchparams接口">setSearchParams</a>。</li><li>setNumProbes接口预计2025年9月废除，请使用<a href="#setsearchparams接口">setSearchParams</a>。</li></ul>
</td>
</tr>
</tbody>
</table>

## setPreciseMemControl接口<a name="ZH-CN_TOPIC_0000001506334681"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p534012467165"><a name="p534012467165"></a><a name="p534012467165"></a>void setPreciseMemControl(bool preciseMemControl);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p19691924201019"><a name="p19691924201019"></a><a name="p19691924201019"></a>是否精确限制Host侧的内存大小。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b9961232154"><a name="b9961232154"></a><a name="b9961232154"></a>bool preciseMemControl</strong>：默认为<span class="parmvalue" id="parmvalue185542312415"><a name="parmvalue185542312415"></a><a name="parmvalue185542312415"></a>“false”</span>，表示停用对Host侧内存大小精确限制；为<span class="parmvalue" id="parmvalue228313252247"><a name="parmvalue228313252247"></a><a name="parmvalue228313252247"></a>“true”</span>时表示启用。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p14992135715819"><a name="p14992135715819"></a><a name="p14992135715819"></a>当前版本暂不支持该接口，请勿调用。</p>
</td>
</tr>
</tbody>
</table>

## setSearchParams接口<a name="ZH-CN_TOPIC_0000002052679693"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1115121782220"><a name="p1115121782220"></a><a name="p1115121782220"></a>void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p10890925192214"><a name="p10890925192214"></a><a name="p10890925192214"></a>设置影响检索精度和性能的参数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p14571833142216"><a name="p14571833142216"></a><a name="p14571833142216"></a>int nprobe：AscendIndexIVFSQT的nprobe数。建议保持为默认值“64”。</p>
<p id="p9571533132212"><a name="p9571533132212"></a><a name="p9571533132212"></a>int l2Probe：二级检索选择子桶的数量，默认值为“48”。</p>
<p id="p8571033152213"><a name="p8571033152213"></a><a name="p8571033152213"></a>int l3SegmentNum：L3算子处理的段数，影响查找的base总数，默认值为“96”。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p963572214280"><a name="p963572214280"></a><a name="p963572214280"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li>nprobe ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }</li><li>l2Probe ≥ nprobe, l2Probe≤ l3SegmentNum, l2Probe≤ nprobe * 64</li><li>l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }</li></ul>
</td>
</tr>
</tbody>
</table>

## setSortMode接口<a name="ZH-CN_TOPIC_0000002165943965"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void setSortMode(int mode);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>设置topk排序模式。模式0为近似排序；模式1为精确排序。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.3.1 "><p id="p6307181718287"><a name="p6307181718287"></a><a name="p6307181718287"></a>int mode：topk排序模式。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.4.1 "><p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.5.1 "><p id="p22145914388"><a name="p22145914388"></a><a name="p22145914388"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.71%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.28999999999999%" headers="mcps1.1.3.6.1 "><a name="ul998918501528"></a><a name="ul998918501528"></a><ul id="ul998918501528"><li>该接口需要在Search接口之前使用。</li><li><span class="parmname" id="parmname4750135914115"><a name="parmname4750135914115"></a><a name="parmname4750135914115"></a>“mode”</span>支持模式0或模式1，默认为模式0。<a name="ul73211618141111"></a><a name="ul73211618141111"></a><ul id="ul73211618141111"><li>模式0：近似排序会截断部分topk结果，提升性能。</li><li>模式1：精确排序，会提升检索精度，牺牲部分性能。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## setThreshold接口<a name="ZH-CN_TOPIC_0000001456854808"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p894112803319"><a name="p894112803319"></a><a name="p894112803319"></a>void setThreshold(float value);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p10450142717337"><a name="p10450142717337"></a><a name="p10450142717337"></a>设置判断向量是否入多个桶的阈值，默认值为<span class="parmvalue" id="parmvalue756673153110"><a name="parmvalue756673153110"></a><a name="parmvalue756673153110"></a>“1.0”</span>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p0873202583311"><a name="p0873202583311"></a><a name="p0873202583311"></a><strong id="b07338551508"><a name="b07338551508"></a><a name="b07338551508"></a>float value</strong>：判断向量是否入多个桶的阈值，建议设置[1.0, 1.5]。由于Device侧内存存在限额，当使用内存达到限额后，会触发OOM机制，导致进程被杀死。用户可先查看Device侧的内存限额数据。（/sys/fs/cgroup/memory/usermemory/memory.limit_in_bytes），来评估添加底库的大小，若内存不充裕时，参数值建议在[1.0, 1.1]范围。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81638244335"><a name="p81638244335"></a><a name="p81638244335"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10418181714331"><a name="p10418181714331"></a><a name="p10418181714331"></a>value的取值范围是[0, fuzzyK- 1]，fuzzyK的取值请参见<a href="#getfuzzyk接口">getFuzzyK接口</a>。</p>
</td>
</tr>
</tbody>
</table>

## setUseCpuUpdate接口<a name="ZH-CN_TOPIC_0000002167379329"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p2026619114105"><a name="p2026619114105"></a><a name="p2026619114105"></a>setUseCpuUpdate(int numThreads);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p7266411101019"><a name="p7266411101019"></a><a name="p7266411101019"></a>是否使用CPU进行<a href="#update接口">update</a>。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p172659115101"><a name="p172659115101"></a><a name="p172659115101"></a><strong id="b11603133521113"><a name="b11603133521113"></a><a name="b11603133521113"></a>int numThreads</strong>：用于进行update的CPU核数，默认值为当前CPU的核数。</p>
<a name="ul76628243368"></a><a name="ul76628243368"></a><ul id="ul76628243368"><li>若当前CPU的核数&gt;96：<a name="ul11814123683920"></a><a name="ul11814123683920"></a><ul id="ul11814123683920"><li>当前CPU核数＜输入的numThreads，<strong id="b2028810524444"><a name="b2028810524444"></a><a name="b2028810524444"></a>numThreads</strong> =96；</li><li>96＜输入的numThreads≤当前CPU核数，<strong id="b324754114419"><a name="b324754114419"></a><a name="b324754114419"></a>numThreads</strong>=96；</li><li>输入的numThreads≤96，numThreads为输入值。</li></ul>
</li><li>若当前CPU的核数≤96：<a name="ul106753468457"></a><a name="ul106753468457"></a><ul id="ul106753468457"><li>当前CPU核数＜输入的numThreads ≤ 96，<strong id="b20758111294711"><a name="b20758111294711"></a><a name="b20758111294711"></a>numThreads</strong>为当前CPU核数；</li><li>0＜输入的numThreads≤当前CPU核数，<strong id="b1695581814610"><a name="b1695581814610"></a><a name="b1695581814610"></a>numThreads</strong>为输入值<strong id="b2955318204612"><a name="b2955318204612"></a><a name="b2955318204612"></a>。</strong></li></ul>
</li></ul>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.4.1 "><p id="p152641411141014"><a name="p152641411141014"></a><a name="p152641411141014"></a>无。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.5.1 "><p id="p1426321151015"><a name="p1426321151015"></a><a name="p1426321151015"></a>无。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul20955612171320"></a><a name="ul20955612171320"></a><ul id="ul20955612171320"><li><strong id="b1276762719138"><a name="b1276762719138"></a><a name="b1276762719138"></a>numThreads</strong>取值需大于0。</li><li>需要在使用<a href="#update接口">update</a>前配置。</li></ul>
</td>
</tr>
</tbody>
</table>

## train接口<a name="ZH-CN_TOPIC_0000001456375352"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void train(idx_t n, const float *x) override;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>对AscendIndexIVFSQT执行训练，继承AscendIndexIVFSQ中的相关接口并提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b141925213710"><a name="b141925213710"></a><a name="b141925213710"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b1196267978"><a name="b1196267978"></a><a name="b1196267978"></a>const float *x</strong>：特征向量数据。</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul15165212077"></a><a name="ul15165212077"></a><ul id="ul15165212077"><li>训练采用k-means进行聚类，训练集比较小可能会影响查询精度。</li><li>此处<span class="parmname" id="parmname11827122719317"><a name="parmname11827122719317"></a><a name="parmname11827122719317"></a>“n”</span>的取值范围：nlist ≤ n ≤ 7,000,000。</li><li>此处指针<span class="parmname" id="parmname19120718371"><a name="parmname19120718371"></a><a name="parmname19120718371"></a>“x”</span>需要为非空指针，且长度应该为dimIn * <strong id="b10897319132513"><a name="b10897319132513"></a><a name="b10897319132513"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## update接口<a name="ZH-CN_TOPIC_0000001506414869"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p07451129133118"><a name="p07451129133118"></a><a name="p07451129133118"></a>void update(bool cleanData = true);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>三级检索的第二级，在add完毕全部的底库数据后，执行search前，用于训练子桶中心并根据子桶中心入桶。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p6307181718287"><a name="p6307181718287"></a><a name="p6307181718287"></a>cleanData：是否清除中间数据，默认为<span class="parmvalue" id="parmvalue137057376475"><a name="parmvalue137057376475"></a><a name="parmvalue137057376475"></a>“true”</span>。</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><p id="p117332418161"><a name="p117332418161"></a><a name="p117332418161"></a>一次检索全流程中该接口只需要调用一次。</p>
</td>
</tr>
</tbody>
</table>

## updateTParams接口<a name="ZH-CN_TOPIC_0000001456854936"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.1.1 "><p id="p1418915311587"><a name="p1418915311587"></a><a name="p1418915311587"></a>void updateTParams(int l2Probe, int l3SegmentNum);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>测试时传入三级检索所需参数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.07%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b95897151286"><a name="b95897151286"></a><a name="b95897151286"></a>int l2Probe</strong>：二级检索选择子桶的数量，默认值为<span class="parmvalue" id="parmvalue81701953182510"><a name="parmvalue81701953182510"></a><a name="parmvalue81701953182510"></a>“48”</span>。</p>
<p id="p11299139310"><a name="p11299139310"></a><a name="p11299139310"></a><strong id="b439191185"><a name="b439191185"></a><a name="b439191185"></a>int l3SegmentNum</strong>：L3算子处理的段数，影响查找的base总数，默认值为<span class="parmvalue" id="parmvalue1345127261"><a name="parmvalue1345127261"></a><a name="parmvalue1345127261"></a>“96”</span>。</p>
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
<td class="cellrowborder" valign="top" width="79.93%" headers="mcps1.1.3.6.1 "><a name="ul102611833282"></a><a name="ul102611833282"></a><ul id="ul102611833282"><li>nprobe ∈{ 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 }</li><li>l2Probe ≥ nprobe, l2Probe≤ l3SegmentNum, l2Probe≤ nprobe * 64</li><li>l3SegmentNum ∈ { 24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 }</li><li>nprobe的设置可参见<a href="#setsearchparams接口">setSearchParams</a>。</li><li>updateTParams接口预计2026年9月废除，请使用<a href="#setsearchparams接口">setSearchParams</a>。</li></ul>
</td>
</tr>
</tbody>
</table>
