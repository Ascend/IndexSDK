# AscendNNInference<a name="ZH-CN_TOPIC_0000001456375320"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456535204"></a>

通过神经网络执行推理。

## AscendNNInference接口<a name="ZH-CN_TOPIC_0000001456854780"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendNNInference(std::vector&lt;int&gt; deviceList, const char* model, uint64_t modelSize);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendNNInference的构造函数，生成AscendNNInference，此时根据<span class="parmname" id="parmname8437181062119"><a name="parmname8437181062119"></a><a name="parmname8437181062119"></a>“deviceList”</span>中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源以及模型路径等。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b18268454414"><a name="b18268454414"></a><a name="b18268454414"></a>std::vector&lt;int&gt; deviceList</strong>：Device侧设备ID。</p>
<p id="p187869217128"><a name="p187869217128"></a><a name="p187869217128"></a><strong id="b9239165094116"><a name="b9239165094116"></a><a name="b9239165094116"></a>const char* model</strong>：深度神经网络推理模型。</p>
<p id="p11661833191215"><a name="p11661833191215"></a><a name="p11661833191215"></a><strong id="b165044526414"><a name="b165044526414"></a><a name="b165044526414"></a>uint64_t modelSize</strong>：深度神经网络推理模型的大小。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul246474615523"></a><a name="ul246474615523"></a><ul id="ul246474615523"><li>deviceList取值范围(0, 32]。</li><li><span class="parmname" id="parmname17928145516416"><a name="parmname17928145516416"></a><a name="parmname17928145516416"></a>“model”</span>需要为合法有效的深度神经网络推理模型的内存指针，大小为<span class="parmname" id="parmname13470125964114"><a name="parmname13470125964114"></a><a name="parmname13470125964114"></a>“modelSize”</span>，modelSize取值范围为(0, 128MB]，参数不匹配可能造成模型实例化或推理失败。非法的模型可能会对系统造成危害，请确保模型的来源合法有效。<a name="ul29631955112419"></a><a name="ul29631955112419"></a><ul id="ul29631955112419"><li>dimsIn ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li>dimsOut ∈ {32, 64, 96, 128, 256}。</li><li>batches ∈ {1, 2, 4, 8, 16, 32, 64, 128}</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1246213101873"></a>
<table><tbody><tr id="row1462121015717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p84621510171"><a name="p84621510171"></a><a name="p84621510171"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p17980142452716"><a name="p17980142452716"></a><a name="p17980142452716"></a>AscendNNInference(const AscendNNInference&amp;) = delete;</p>
</td>
</tr>
<tr id="row164624102073"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p12462161014713"><a name="p12462161014713"></a><a name="p12462161014713"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1462161010718"><a name="p1462161010718"></a><a name="p1462161010718"></a>声明此AscendNNInference拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row5462101013718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p04623106716"><a name="p04623106716"></a><a name="p04623106716"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b05634141428"><a name="b05634141428"></a><a name="b05634141428"></a>const AscendNNInference&amp;</strong>：常量AscendNNInference。</p>
</td>
</tr>
<tr id="row12462610671"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p94628106715"><a name="p94628106715"></a><a name="p94628106715"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p946213108719"><a name="p946213108719"></a><a name="p946213108719"></a>无</p>
</td>
</tr>
<tr id="row194623101670"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1546320101573"><a name="p1546320101573"></a><a name="p1546320101573"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1346321016716"><a name="p1346321016716"></a><a name="p1346321016716"></a>无</p>
</td>
</tr>
<tr id="row104631810275"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1046391017711"><a name="p1046391017711"></a><a name="p1046391017711"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## \~AscendNNInference接口<a name="ZH-CN_TOPIC_0000001506495737"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a>~AscendNNInference();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendNNInference的析构函数，销毁AscendNNInference对象，释放资源。</p>
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

## getDimBatch接口<a name="ZH-CN_TOPIC_0000001506334797"></a>

<a name="zh-cn_topic_0000001287392566_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001287392566_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001287392566_p12559123810"><a name="zh-cn_topic_0000001287392566_p12559123810"></a><a name="zh-cn_topic_0000001287392566_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001287392566_p13466158111313"><a name="zh-cn_topic_0000001287392566_p13466158111313"></a><a name="zh-cn_topic_0000001287392566_p13466158111313"></a>int getDimBatch() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001287392566_p1212599383"><a name="zh-cn_topic_0000001287392566_p1212599383"></a><a name="zh-cn_topic_0000001287392566_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001287392566_p131714208358"><a name="zh-cn_topic_0000001287392566_p131714208358"></a><a name="zh-cn_topic_0000001287392566_p131714208358"></a>获取模型的单次推理的样本或查询向量的数量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001287392566_p112195910383"><a name="zh-cn_topic_0000001287392566_p112195910383"></a><a name="zh-cn_topic_0000001287392566_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001287392566_p1963814585141"><a name="zh-cn_topic_0000001287392566_p1963814585141"></a><a name="zh-cn_topic_0000001287392566_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001287392566_p17235973820"><a name="zh-cn_topic_0000001287392566_p17235973820"></a><a name="zh-cn_topic_0000001287392566_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001287392566_p8451184515218"><a name="zh-cn_topic_0000001287392566_p8451184515218"></a><a name="zh-cn_topic_0000001287392566_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001287392566_p182459113812"><a name="zh-cn_topic_0000001287392566_p182459113812"></a><a name="zh-cn_topic_0000001287392566_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001287392566_p132314362521"><a name="zh-cn_topic_0000001287392566_p132314362521"></a><a name="zh-cn_topic_0000001287392566_p132314362521"></a>模型单次推理的样本或查询向量的数量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001287392566_p423590386"><a name="zh-cn_topic_0000001287392566_p423590386"></a><a name="zh-cn_topic_0000001287392566_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001287392566_p991611401713"><a name="zh-cn_topic_0000001287392566_p991611401713"></a><a name="zh-cn_topic_0000001287392566_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getInputType接口<a name="ZH-CN_TOPIC_0000001456854776"></a>

<a name="zh-cn_topic_0000001340072289_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001340072289_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001340072289_p12559123810"><a name="zh-cn_topic_0000001340072289_p12559123810"></a><a name="zh-cn_topic_0000001340072289_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001340072289_p13466158111313"><a name="zh-cn_topic_0000001340072289_p13466158111313"></a><a name="zh-cn_topic_0000001340072289_p13466158111313"></a>int getInputType() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001340072289_p1212599383"><a name="zh-cn_topic_0000001340072289_p1212599383"></a><a name="zh-cn_topic_0000001340072289_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001340072289_p131714208358"><a name="zh-cn_topic_0000001340072289_p131714208358"></a><a name="zh-cn_topic_0000001340072289_p131714208358"></a>获取模型的输入数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001340072289_p112195910383"><a name="zh-cn_topic_0000001340072289_p112195910383"></a><a name="zh-cn_topic_0000001340072289_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001340072289_p1963814585141"><a name="zh-cn_topic_0000001340072289_p1963814585141"></a><a name="zh-cn_topic_0000001340072289_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001340072289_p17235973820"><a name="zh-cn_topic_0000001340072289_p17235973820"></a><a name="zh-cn_topic_0000001340072289_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001340072289_p8451184515218"><a name="zh-cn_topic_0000001340072289_p8451184515218"></a><a name="zh-cn_topic_0000001340072289_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001340072289_p182459113812"><a name="zh-cn_topic_0000001340072289_p182459113812"></a><a name="zh-cn_topic_0000001340072289_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001340072289_p132314362521"><a name="zh-cn_topic_0000001340072289_p132314362521"></a><a name="zh-cn_topic_0000001340072289_p132314362521"></a>模型的输入数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001340072289_p423590386"><a name="zh-cn_topic_0000001340072289_p423590386"></a><a name="zh-cn_topic_0000001340072289_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001340072289_p991611401713"><a name="zh-cn_topic_0000001340072289_p991611401713"></a><a name="zh-cn_topic_0000001340072289_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getOutputType接口<a name="ZH-CN_TOPIC_0000001456854868"></a>

<a name="zh-cn_topic_0000001340232437_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001340232437_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001340232437_p12559123810"><a name="zh-cn_topic_0000001340232437_p12559123810"></a><a name="zh-cn_topic_0000001340232437_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001340232437_p13466158111313"><a name="zh-cn_topic_0000001340232437_p13466158111313"></a><a name="zh-cn_topic_0000001340232437_p13466158111313"></a>int getOutputType() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001340232437_p1212599383"><a name="zh-cn_topic_0000001340232437_p1212599383"></a><a name="zh-cn_topic_0000001340232437_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001340232437_p131714208358"><a name="zh-cn_topic_0000001340232437_p131714208358"></a><a name="zh-cn_topic_0000001340232437_p131714208358"></a>获取模型的输出数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001340232437_p112195910383"><a name="zh-cn_topic_0000001340232437_p112195910383"></a><a name="zh-cn_topic_0000001340232437_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001340232437_p1963814585141"><a name="zh-cn_topic_0000001340232437_p1963814585141"></a><a name="zh-cn_topic_0000001340232437_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001340232437_p17235973820"><a name="zh-cn_topic_0000001340232437_p17235973820"></a><a name="zh-cn_topic_0000001340232437_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001340232437_p8451184515218"><a name="zh-cn_topic_0000001340232437_p8451184515218"></a><a name="zh-cn_topic_0000001340232437_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001340232437_p182459113812"><a name="zh-cn_topic_0000001340232437_p182459113812"></a><a name="zh-cn_topic_0000001340232437_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001340232437_p132314362521"><a name="zh-cn_topic_0000001340232437_p132314362521"></a><a name="zh-cn_topic_0000001340232437_p132314362521"></a>模型的输出数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001340232437_p423590386"><a name="zh-cn_topic_0000001340232437_p423590386"></a><a name="zh-cn_topic_0000001340232437_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001340232437_p991611401713"><a name="zh-cn_topic_0000001340232437_p991611401713"></a><a name="zh-cn_topic_0000001340232437_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getDimIn接口<a name="ZH-CN_TOPIC_0000001456535128"></a>

<a name="zh-cn_topic_0000001287712442_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001287712442_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001287712442_p12559123810"><a name="zh-cn_topic_0000001287712442_p12559123810"></a><a name="zh-cn_topic_0000001287712442_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001287712442_p13466158111313"><a name="zh-cn_topic_0000001287712442_p13466158111313"></a><a name="zh-cn_topic_0000001287712442_p13466158111313"></a>int getDimIn() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001287712442_p1212599383"><a name="zh-cn_topic_0000001287712442_p1212599383"></a><a name="zh-cn_topic_0000001287712442_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001287712442_p131714208358"><a name="zh-cn_topic_0000001287712442_p131714208358"></a><a name="zh-cn_topic_0000001287712442_p131714208358"></a>获取模型的输入数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001287712442_p112195910383"><a name="zh-cn_topic_0000001287712442_p112195910383"></a><a name="zh-cn_topic_0000001287712442_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001287712442_p1963814585141"><a name="zh-cn_topic_0000001287712442_p1963814585141"></a><a name="zh-cn_topic_0000001287712442_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001287712442_p17235973820"><a name="zh-cn_topic_0000001287712442_p17235973820"></a><a name="zh-cn_topic_0000001287712442_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001287712442_p8451184515218"><a name="zh-cn_topic_0000001287712442_p8451184515218"></a><a name="zh-cn_topic_0000001287712442_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001287712442_p182459113812"><a name="zh-cn_topic_0000001287712442_p182459113812"></a><a name="zh-cn_topic_0000001287712442_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001287712442_p132314362521"><a name="zh-cn_topic_0000001287712442_p132314362521"></a><a name="zh-cn_topic_0000001287712442_p132314362521"></a>输入数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001287712442_p423590386"><a name="zh-cn_topic_0000001287712442_p423590386"></a><a name="zh-cn_topic_0000001287712442_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001287712442_p991611401713"><a name="zh-cn_topic_0000001287712442_p991611401713"></a><a name="zh-cn_topic_0000001287712442_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>

## getDimOut接口<a name="ZH-CN_TOPIC_0000001456695056"></a>

<a name="zh-cn_topic_0000001287552486_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001287552486_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001287552486_p12559123810"><a name="zh-cn_topic_0000001287552486_p12559123810"></a><a name="zh-cn_topic_0000001287552486_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001287552486_p13466158111313"><a name="zh-cn_topic_0000001287552486_p13466158111313"></a><a name="zh-cn_topic_0000001287552486_p13466158111313"></a>int getDimOut() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001287552486_p1212599383"><a name="zh-cn_topic_0000001287552486_p1212599383"></a><a name="zh-cn_topic_0000001287552486_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001287552486_p131714208358"><a name="zh-cn_topic_0000001287552486_p131714208358"></a><a name="zh-cn_topic_0000001287552486_p131714208358"></a>获取模型的输出数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001287552486_p112195910383"><a name="zh-cn_topic_0000001287552486_p112195910383"></a><a name="zh-cn_topic_0000001287552486_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001287552486_p1963814585141"><a name="zh-cn_topic_0000001287552486_p1963814585141"></a><a name="zh-cn_topic_0000001287552486_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001287552486_p17235973820"><a name="zh-cn_topic_0000001287552486_p17235973820"></a><a name="zh-cn_topic_0000001287552486_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001287552486_p8451184515218"><a name="zh-cn_topic_0000001287552486_p8451184515218"></a><a name="zh-cn_topic_0000001287552486_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001287552486_p182459113812"><a name="zh-cn_topic_0000001287552486_p182459113812"></a><a name="zh-cn_topic_0000001287552486_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001287552486_p132314362521"><a name="zh-cn_topic_0000001287552486_p132314362521"></a><a name="zh-cn_topic_0000001287552486_p132314362521"></a>模型的输出数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001287552486_p423590386"><a name="zh-cn_topic_0000001287552486_p423590386"></a><a name="zh-cn_topic_0000001287552486_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001287552486_p991611401713"><a name="zh-cn_topic_0000001287552486_p991611401713"></a><a name="zh-cn_topic_0000001287552486_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>

## infer接口<a name="ZH-CN_TOPIC_0000001506495709"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a>void infer(size_t n, const char* inputData, char* outputData) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据网络模型执行推理。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.3.1 "><p id="p1963814585141"><a name="p1963814585141"></a><a name="p1963814585141"></a><strong id="b33632034317"><a name="b33632034317"></a><a name="b33632034317"></a>size_t n</strong>：待执行推理的输入数量。</p>
<p id="p1633753171511"><a name="p1633753171511"></a><a name="p1633753171511"></a><strong id="b12723522435"><a name="b12723522435"></a><a name="b12723522435"></a>const char* inputData</strong>：待执行推理的特征向量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.4.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b29431145430"><a name="b29431145430"></a><a name="b29431145430"></a>char* outputData</strong>：执行推理得到的特征向量结果。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.6.1 "><a name="ul58419974316"></a><a name="ul58419974316"></a><ul id="ul58419974316"><li>此处<span class="parmname" id="parmname1589434893110"><a name="parmname1589434893110"></a><a name="parmname1589434893110"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname1017013169439"><a name="parmname1017013169439"></a><a name="parmname1017013169439"></a>“inputData”</span>需要为非空指针，且长度应该为dimIn * <strong id="b1658485663114"><a name="b1658485663114"></a><a name="b1658485663114"></a>n</strong>，<span class="parmname" id="parmname10352171914316"><a name="parmname10352171914316"></a><a name="parmname10352171914316"></a>“outputData”</span>需要为非空指针，且长度应该为dimOut * <strong id="b952335293118"><a name="b952335293118"></a><a name="b952335293118"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001456535156"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p17980142452716"><a name="p17980142452716"></a><a name="p17980142452716"></a>AscendNNInference&amp; operator=(const AscendNNInference&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b19703385424"><a name="b19703385424"></a><a name="b19703385424"></a>const AscendNNInference&amp;</strong>：常量AscendNNInference。</p>
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
