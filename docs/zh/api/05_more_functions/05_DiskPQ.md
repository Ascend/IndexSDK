# DiskPQ<a name="ZH-CN_TOPIC_0000002382802364"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002382647580"></a>

Index SDK提供PQ（Product Quantization）量化的训练和检索功能。PQ接口不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则可能导致功能异常。

## DiskPQParams接口<a name="ZH-CN_TOPIC_0000002382807444"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p784562919563"><a name="p784562919563"></a><a name="p784562919563"></a>DiskPQParams {</p>
<p id="p584518296562"><a name="p584518296562"></a><a name="p584518296562"></a>int pqChunks = 512;</p>
<p id="p17845192912562"><a name="p17845192912562"></a><a name="p17845192912562"></a>int funcType = 1;</p>
<p id="p1884502910567"><a name="p1884502910567"></a><a name="p1884502910567"></a>int dim = 1;</p>
<p id="p8845122945614"><a name="p8845122945614"></a><a name="p8845122945614"></a>char *pqTable = nullptr;</p>
<p id="p13845202914564"><a name="p13845202914564"></a><a name="p13845202914564"></a>uint32_t *offsets = nullptr;</p>
<p id="p28451029185617"><a name="p28451029185617"></a><a name="p28451029185617"></a>char *tablesTransposed = nullptr;</p>
<p id="p12845102915560"><a name="p12845102915560"></a><a name="p12845102915560"></a>char *centroids = nullptr;</p>
<p id="p584532905613"><a name="p584532905613"></a><a name="p584532905613"></a>}</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>PQ量化结构体。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><strong id="b3354181515167"><a name="b3354181515167"></a><a name="b3354181515167"></a>int pqChunks：</strong>表示将原始向量维度dim切分为pqChunks块。</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><strong id="b13953183012163"><a name="b13953183012163"></a><a name="b13953183012163"></a>int funcType：</strong>表示进行PQ查表距离计算时使用的计算标准。</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><strong id="b229253513169"><a name="b229253513169"></a><a name="b229253513169"></a>int dim：</strong>表示原始数据维度。</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><strong id="b1712474051618"><a name="b1712474051618"></a><a name="b1712474051618"></a>char *pqTable：</strong>表示存储码本数据的指针。默认值为nullptr。</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><strong id="b76015463165"><a name="b76015463165"></a><a name="b76015463165"></a>uint32_t *offsets：</strong>表示存储每个chunk在原始维度上起始和截止的维度。默认值为nullptr。</p>
<p id="p16645243598"><a name="p16645243598"></a><a name="p16645243598"></a><strong id="b632142913336"><a name="b632142913336"></a><a name="b632142913336"></a>char *tablesTransposed：</strong>表示存储码本数据的转置形态指针。默认值为nullptr。</p>
<p id="p864518435917"><a name="p864518435917"></a><a name="p864518435917"></a><strong id="b172846021716"><a name="b172846021716"></a><a name="b172846021716"></a>char *centroids：</strong>表示存储每个维度的平均值，用于对数据进行中心化处理。默认值为nullptr。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1368219905119"></a><a name="ul1368219905119"></a><ul id="ul1368219905119"><li>1 &lt;= pqChunks &lt;= dim。使用较小pqChunks将使用更少内存，但会带来相应的精度损失。一般情况下，推荐使用pqChunks为dim / 8或者dim / 16（均向上取整）。默认值为512。</li><li>funcType取值范围为1~3。1表示使用L2距离；2表示使用IP距离；3表示使用cosine距离。默认值为1。</li><li>1 &lt;= dim &lt;= 2000。默认值为1。</li><li>pqTable目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。</li><li>tablesTransposed目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。</li></ul>
</td>
</tr>
</tbody>
</table>

## VectorArrayData接口<a name="ZH-CN_TOPIC_0000002416326913"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p455275425615"><a name="p455275425615"></a><a name="p455275425615"></a>VectorArrayData {</p>
<p id="p055225485618"><a name="p055225485618"></a><a name="p055225485618"></a>int length;</p>
<p id="p20552454105617"><a name="p20552454105617"></a><a name="p20552454105617"></a>int maxlen;</p>
<p id="p1355211547565"><a name="p1355211547565"></a><a name="p1355211547565"></a>int dim;</p>
<p id="p15552155445611"><a name="p15552155445611"></a><a name="p15552155445611"></a>size_t itemsize;</p>
<p id="p655255413561"><a name="p655255413561"></a><a name="p655255413561"></a>char *items;</p>
<p id="p75521654195610"><a name="p75521654195610"></a><a name="p75521654195610"></a>}</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>数据封装结构体。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><strong id="b168113245547"><a name="b168113245547"></a><a name="b168113245547"></a>int length：</strong>表示结构体中存储的向量条数。</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><strong id="b13953183012163"><a name="b13953183012163"></a><a name="b13953183012163"></a>int maxlen：</strong>表示结构体中存储的最大向量条数。</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><strong id="b229253513169"><a name="b229253513169"></a><a name="b229253513169"></a>int dim：</strong>表示结构体中存储的向量维度。</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><strong id="b1712474051618"><a name="b1712474051618"></a><a name="b1712474051618"></a>size_t itemsize：</strong>保留字段，用户可以选择不设置。</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><strong id="b346491725412"><a name="b346491725412"></a><a name="b346491725412"></a>char *items：</strong>表示存储VectorArrayData中数据的指针。默认值为nullptr。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1368219905119"></a><a name="ul1368219905119"></a><ul id="ul1368219905119"><li>1 &lt;= length &lt;= 100000000。</li><li>maxlen是OpenGauss侧保留字段，非OpenGauss用户设置该值等于length值即可。</li><li>1 &lt;= dim &lt;= 2000。</li><li>对于不同接口，用户需要确保items指向不同大小的数据。</li></ul>
</td>
</tr>
</tbody>
</table>

## ComputePQTable接口<a name="ZH-CN_TOPIC_0000002416446741"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>使用sample中存储的采样底库数据计算PQ码本，并将码本相关的数据存储在参数params中的对应参数里。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b1110246113812"><a name="b1110246113812"></a><a name="b1110246113812"></a>VectorArrayData *sample：</strong>指向填充好采样底库数据的VectorArrayData实例的指针。不能为空指针。</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><strong id="b5117650173817"><a name="b5117650173817"></a><a name="b5117650173817"></a>DiskPQParams *params：</strong>指向仅包含PQ参数，未填充训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>sample数据填充要求如下：<p id="p14941713111016"><a name="p14941713111016"></a><a name="p14941713111016"></a>items指向的数据大小为(8 + dim) * length * sizeof(float)字节，即每条向量前有8字节的metadata。非OpenGauss用户使用时，需在每条向量数据添加8字节的任意数据。</p>
</li><li>params成员变量填充要求如下：<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>dim除满足上述的范围限制要求之外，还需确保与sample中对应的dim字段保持一致。</li><li>pqTable必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * 256 （256为每个chunk内的聚类数）* sizeof(float)字节。</li><li>offsets必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于(pqChunks + 1) * sizeof(uint32_t)字节。</li><li>tablesTransposed必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * 256 * sizeof(float)字节。</li><li>centroids必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * sizeof(float)字节。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## ComputeVectorPQCode接口<a name="ZH-CN_TOPIC_0000002382647584"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a>int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>使用填充好PQ数据的params，对baseData中的底库数据进行量化，并将量化数据写入pqCode指向的缓存区中。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b89499121237"><a name="b89499121237"></a><a name="b89499121237"></a>VectorArrayData *baseData：</strong>指向填充好底库数据的VectorArrayData实例的指针。不能为空指针。用户可以根据自身内存的限制，在外层决定baseData中底库数据的大小。</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><strong id="b617081772316"><a name="b617081772316"></a><a name="b617081772316"></a>const DiskPQParams *params：</strong>指向填充好PQ参数和训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b293063113232"><a name="b293063113232"></a><a name="b293063113232"></a>uint8_t *pqCode：</strong>接收返回的压缩好的底库向量的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>baseData数据填充要求如下：<p id="p71308396175"><a name="p71308396175"></a><a name="p71308396175"></a>items指向的数据大小为length * dim * sizeof(float)字节。注意此处与ComputePQTable接口不同， 无需在每条数据前填充代替metadata的数据。</p>
</li><li>params成员变量填充要求如下：<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>dim除满足上述的范围限制要求之外，还需确保与sample中对应的dim字段保持一致。</li><li>pqTable必须指向内存大小为dim * 256 * sizeof(float)字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>offsets必须指向内存大小为(pqChunks + 1) * sizeof(uint32_t)字节数的offsets数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>对tablesTransposed填充值无要求。</li><li>centroids必须指向内存大小为dim * sizeof(float)字节数的centroids数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li></ul>
</li><li>用户需保证pqCode指向的空间大小至少有length * pqChunks字节数。其中，length为VectorArrayData参数；pqChunks为DiskPQParams参数。</li></ul>
</td>
</tr>
</tbody>
</table>

## GetPQDistanceTable接口<a name="ZH-CN_TOPIC_0000002382807448"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a>int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>使用填充好PQ数据的params，对vec指向的query数据进行ADC PQ距离计算，并将PQ距离表写入pqDistanceTable指向的缓存区中。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b510342493616"><a name="b510342493616"></a><a name="b510342493616"></a>char *vec：</strong>指向待计算的query数据的指针。</p>
<p id="p86231040131614"><a name="p86231040131614"></a><a name="p86231040131614"></a><strong id="b23454284363"><a name="b23454284363"></a><a name="b23454284363"></a>const DiskPQParams *params：</strong>指向填充好PQ参数和训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b1548973317364"><a name="b1548973317364"></a><a name="b1548973317364"></a>float *pqDistanceTable：</strong>接收返回的query与每个chunk内每个centroid距离的指针。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>用户需保证vec指向的空间大小至少有dim * sizeof(float)字节数。目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。</li><li>params成员变量填充要求如下：<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>pqTable指向值无要求。</li><li>offsets必须指向内存大小为(pqChunks + 1) * sizeof(uint32_t)字节数的offsets数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>tablesTransposed必须指向内存大小为dim * 256 * sizeof(float)字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>centroids必须指向内存大小为dim * sizeof(float)字节数的centroids数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li></ul>
</li><li>用户需保证pqDistanceTable指向的空间大小至少有pqChunks * 256 * sizeof(float)字节数。</li></ul>
</td>
</tr>
</tbody>
</table>

## GetPQDistance接口<a name="ZH-CN_TOPIC_0000002416326917"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a>int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &amp;pqDistance);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p169090129309"><a name="p169090129309"></a><a name="p169090129309"></a>使用basecode指向的底库向量对应的压缩码字数据和GetPQDistanceTable接口中获取的pqDistanceTable，计算query与该底库向量的PQ距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1533165363012"><a name="p1533165363012"></a><a name="p1533165363012"></a><strong id="b10761103265015"><a name="b10761103265015"></a><a name="b10761103265015"></a>const uint8_t *basecode：</strong>指向一个底库向量对应的压缩码字数据的指针。</p>
<p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b1811010371509"><a name="b1811010371509"></a><a name="b1811010371509"></a>const DiskPQParams *params：</strong>指向填充好pqChunks数值的DiskPQParams实例的指针。不能为空指针。</p>
<p id="p18804047171617"><a name="p18804047171617"></a><a name="p18804047171617"></a><strong id="b144854125013"><a name="b144854125013"></a><a name="b144854125013"></a>const float *pqDistanceTable：</strong>指向query对应的ADC PQ距离表的指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b19388104715016"><a name="b19388104715016"></a><a name="b19388104715016"></a>float &amp;pqDistance：</strong>接收最终输出的PQ距离的引用值。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>用户需保证basecode指向的数据大小至少有pqChunks个字节。</li><li>在params中，仅需填充pqChunks值，且与basecode中提到的pqChunks值对应。</li><li>用户需保证pqDistanceTable指向的数据大小至少有pqChunks * 256 * sizeof(float)字节数。</li><li>接口中不会在使用前对pqDistance置零，pqDistance最终结果为原pqDistance值 + 输出的query与basecode的PQ距离，因此推荐输入值为0。</li></ul>
</td>
</tr>
</tbody>
</table>
