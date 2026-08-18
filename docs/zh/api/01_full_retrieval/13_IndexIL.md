# IndexIL<a name="ZH-CN_TOPIC_0000001506414825"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456535188"></a>

IndexIL是一个基于连续内存申请机制的特征管理抽象类，服务于将下标索引作为label的检索算法，需要继承实现所有接口来使用。

要求存入底库的向量以及各个接口的query向量均为归一化后的FP16浮点数类型。（**IL**表示“Indices as Labels”。）

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

## AddFeatures接口<a name="ZH-CN_TOPIC_0000001506414693"></a>

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

## IndexIL接口<a name="ZH-CN_TOPIC_0000001456695020"></a>

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

## \~IndexIL接口<a name="ZH-CN_TOPIC_0000001506334781"></a>

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

## Finalize接口<a name="ZH-CN_TOPIC_0000001456375356"></a>

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

## GetFeatures接口<a name="ZH-CN_TOPIC_0000001506495833"></a>

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

## GetNTotal接口<a name="ZH-CN_TOPIC_0000001456535092"></a>

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

## Init接口<a name="ZH-CN_TOPIC_0000001506334657"></a>

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

## RemoveFeatures接口<a name="ZH-CN_TOPIC_0000001456534932"></a>

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

## SetNTotal接口<a name="ZH-CN_TOPIC_0000001456375256"></a>

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
