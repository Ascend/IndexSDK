# AscendCloner<a name="ZH-CN_TOPIC_0000001506334577"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456375412"></a>

Index SDK提供了将NPU上的检索Index资源拷贝到CPU侧Faiss的操作，拷贝过程发生在内存中，原始NPU的Index上加载的数据会被拷贝到CPU侧的内存中，方便用户在CPU上使用相同的底库执行检索。

> [!NOTE]
>部分版本的Faiss中提供了将内存中的Index落盘（内存中的数据保存到本地硬盘）的方法，用户在基于Index SDK和Faiss处理某些敏感数据时需要特别注意提供对应的权限控制和加密保护。

## index\_ascend\_to\_cpu接口<a name="ZH-CN_TOPIC_0000001506334821"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p01708458179"><a name="p01708458179"></a><a name="p01708458179"></a>faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据Ascend上的检索index资源，拷贝生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b1522416109533"><a name="b1522416109533"></a><a name="b1522416109533"></a>const faiss::Index *ascend_index</strong>：Ascend上的Index资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p3947640162619"><a name="p3947640162619"></a><a name="p3947640162619"></a>使用完毕该接口返回的Index指针后请注意delete掉此指针，释放对应的空间。</p>
</td>
</tr>
</tbody>
</table>

## index\_cpu\_to\_ascend接口<a name="ZH-CN_TOPIC_0000001456695032"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1915092217197"><a name="p1915092217197"></a><a name="p1915092217197"></a>faiss::Index *index_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b1464710223531"><a name="b1464710223531"></a><a name="b1464710223531"></a>std::initializer_list&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p142883052016"><a name="p142883052016"></a><a name="p142883052016"></a><strong id="b18287724145316"><a name="b18287724145316"></a><a name="b18287724145316"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p15735115410199"><a name="p15735115410199"></a><a name="p15735115410199"></a><strong id="b883722705317"><a name="b883722705317"></a><a name="b883722705317"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul37673419536"></a><a name="ul37673419536"></a><ul id="ul37673419536"><li>使用完毕该接口返回的Index指针后请注意delete掉此指针，释放对应的空间。</li><li><span class="parmname" id="parmname11908183575319"><a name="parmname11908183575319"></a><a name="parmname11908183575319"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname1260113705320"><a name="parmname1260113705320"></a><a name="parmname1260113705320"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table22143401019"></a>
<table><tbody><tr id="row122113471017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p0263414104"><a name="p0263414104"></a><a name="p0263414104"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12818145232014"><a name="p12818145232014"></a><a name="p12818145232014"></a>faiss::Index *index_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row1629341102"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212340103"><a name="p1212340103"></a><a name="p1212340103"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82153419105"><a name="p82153419105"></a><a name="p82153419105"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row10211342101"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p182234111012"><a name="p182234111012"></a><a name="p182234111012"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p122133411107"><a name="p122133411107"></a><a name="p122133411107"></a><strong id="b1675612491543"><a name="b1675612491543"></a><a name="b1675612491543"></a>std::vector&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p02113412104"><a name="p02113412104"></a><a name="p02113412104"></a><strong id="b1646135211547"><a name="b1646135211547"></a><a name="b1646135211547"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p521734121010"><a name="p521734121010"></a><a name="p521734121010"></a><strong id="b8416205519544"><a name="b8416205519544"></a><a name="b8416205519544"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row20253411010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p12734121017"><a name="p12734121017"></a><a name="p12734121017"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p22203431014"><a name="p22203431014"></a><a name="p22203431014"></a>无</p>
</td>
</tr>
<tr id="row15210342102"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12283431010"><a name="p12283431010"></a><a name="p12283431010"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132173418105"><a name="p132173418105"></a><a name="p132173418105"></a>生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row112143411101"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5210340101"><a name="p5210340101"></a><a name="p5210340101"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1030011345514"></a><a name="ul1030011345514"></a><ul id="ul1030011345514"><li>使用完毕该接口返回的Index指针后请注意delete掉此指针，释放对应的空间。</li><li><span class="parmname" id="parmname41511577559"><a name="parmname41511577559"></a><a name="parmname41511577559"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname1362238115514"><a name="parmname1362238115514"></a><a name="parmname1362238115514"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>

## index\_int8\_ascend\_to\_cpu接口<a name="ZH-CN_TOPIC_0000001506414761"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p01708458179"><a name="p01708458179"></a><a name="p01708458179"></a>faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据Ascend上的INT8的检索Index资源，拷贝生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b6155153813552"><a name="b6155153813552"></a><a name="b6155153813552"></a>const AscendIndexInt8 *index</strong>：Ascend上的Index资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20237164255510"></a><a name="ul20237164255510"></a><ul id="ul20237164255510"><li>使用完毕该接口返回的Index指针后请注意delete此指针，释放对应的空间。</li><li><span class="parmname" id="parmname1298514443554"><a name="parmname1298514443554"></a><a name="parmname1298514443554"></a>“index”</span>需要为合法有效的AscendIndexInt8指针。</li></ul>
</td>
</tr>
</tbody>
</table>

## index\_int8\_cpu\_to\_ascend接口<a name="ZH-CN_TOPIC_0000001456375248"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19640556122110"><a name="p19640556122110"></a><a name="p19640556122110"></a>AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p894441552217"><a name="p894441552217"></a><a name="p894441552217"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b3395198135618"><a name="b3395198135618"></a><a name="b3395198135618"></a>std::initializer_list&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p142883052016"><a name="p142883052016"></a><a name="p142883052016"></a><strong id="b183851411105619"><a name="b183851411105619"></a><a name="b183851411105619"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p15735115410199"><a name="p15735115410199"></a><a name="p15735115410199"></a><strong id="b851411319563"><a name="b851411319563"></a><a name="b851411319563"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul142782285619"></a><a name="ul142782285619"></a><ul id="ul142782285619"><li>使用完毕该接口返回的Index指针后请注意delete此指针，释放对应的空间。</li><li><span class="parmname" id="parmname454423155617"><a name="parmname454423155617"></a><a name="parmname454423155617"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname12364173425615"><a name="parmname12364173425615"></a><a name="parmname12364173425615"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table161071151116"></a>
<table><tbody><tr id="row0610181121116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p166107115116"><a name="p166107115116"></a><a name="p166107115116"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p561010115113"><a name="p561010115113"></a><a name="p561010115113"></a>AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row1161011116113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p461011116112"><a name="p461011116112"></a><a name="p461011116112"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6610181101117"><a name="p6610181101117"></a><a name="p6610181101117"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row166109141110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14610131191111"><a name="p14610131191111"></a><a name="p14610131191111"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p36107118118"><a name="p36107118118"></a><a name="p36107118118"></a><strong id="b411385010568"><a name="b411385010568"></a><a name="b411385010568"></a>std::vector&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p1561010110117"><a name="p1561010110117"></a><a name="p1561010110117"></a><strong id="b7139524560"><a name="b7139524560"></a><a name="b7139524560"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p561031161112"><a name="p561031161112"></a><a name="p561031161112"></a><strong id="b0814145395615"><a name="b0814145395615"></a><a name="b0814145395615"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row1461012111111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p36107113117"><a name="p36107113117"></a><a name="p36107113117"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p161071131117"><a name="p161071131117"></a><a name="p161071131117"></a>无</p>
</td>
</tr>
<tr id="row14611161171112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p106113111111"><a name="p106113111111"></a><a name="p106113111111"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16111161110"><a name="p16111161110"></a><a name="p16111161110"></a>生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row146118119111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5611101151114"><a name="p5611101151114"></a><a name="p5611101151114"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20972759205619"></a><a name="ul20972759205619"></a><ul id="ul20972759205619"><li>使用完毕该接口返回的Index指针后请注意delete此指针，释放对应的空间。</li><li><span class="parmname" id="parmname152210225710"><a name="parmname152210225710"></a><a name="parmname152210225710"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname13280475716"><a name="parmname13280475716"></a><a name="parmname13280475716"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>
