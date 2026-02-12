## 属性过滤检索<a name="ZH-CN_TOPIC_0000001482844454"></a>

### AscendIndexTS<a name="ZH-CN_TOPIC_0000001507640105"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001507879785"></a>

时空库功能类接口。添加底库特征时，每条特征可以配置一个属性FeatureAttr，执行检索功能时每一批query向量可以配置一个过滤器AttrFilter，该过滤器首先对全量的底库进行筛选并与符合条件的向量进行比对。

当前支持以下算法：

-   二值化特征检索（汉明距离）：使用前需要手动生成[BinaryFlat](../user_guide.md#binaryflat)、[Mask](../user_guide.md#mask)算子并移动到对应的“modelpath”目录中。
-   Int8Flat（cos距离）、FP16Flat（IP距离）、Int8Flat（L2距离）：使用前需要手动生成[Mask](../user_guide.md#mask)算子并移动到对应的“modelpath”目录中。
-   支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。


#### AddFeature接口<a name="ZH-CN_TOPIC_0000001458360182"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p645895412284"><a name="p645895412284"></a><a name="p645895412284"></a>APP_ERROR AddFeature(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const uint8_t *customAttr = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>添加特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p1994144463213"><a name="p1994144463213"></a><a name="p1994144463213"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a>int64_t count</strong>：待添加的特征数量。</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a>const void *features</strong>：待添加的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型；FP16Flat距离为float类型。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1733711363162"><a name="b1733711363162"></a><a name="b1733711363162"></a>const FeatureAttr *attributes</strong>：待添加的特征属性，具体请参见<a href="#ZH-CN_TOPIC_0000001507967381">FeatureAttr</a>。</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b11413152220159"><a name="b11413152220159"></a><a name="b11413152220159"></a>const int64_t *labels</strong>：待添加的特征Label，使用上需要保证Label在Index实例中的唯一性。</p>
<p id="p1087211592617"><a name="p1087211592617"></a><a name="p1087211592617"></a><strong id="b1387213364459"><a name="b1387213364459"></a><a name="b1387213364459"></a>const uint8_t *customAttr</strong>：待添加的用户自定义特征属性。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname817612520010"><a name="parmname817612520010"></a><a name="parmname817612520010"></a>“count”</span>单次取值在[1, 1e6]区间，底库最大值1e9。</li><li><span class="parmname" id="parmname13390711317"><a name="parmname13390711317"></a><a name="parmname13390711317"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname186181734113"><a name="parmname186181734113"></a><a name="parmname186181734113"></a>“attributes”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname16572106417"><a name="parmname16572106417"></a><a name="parmname16572106417"></a>“labels”</span>长度为count，各元素不重复且都不在底库中，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname17305193215019"><a name="parmname17305193215019"></a><a name="parmname17305193215019"></a>“customAttr”</span>取值为空指针或者长度为count * customAttrLen，否则可能出现越界读写错误并引起程序崩溃；customAttrLen在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>或<a href="#ZH-CN_TOPIC_0000002013206217">InitWithExtraVal</a>设置。</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE] 说明 
>AddFeature不能与AddWithExtraVal接口混用。

#### AddFeatureByIndice接口<a name="ZH-CN_TOPIC_0000002411433020"></a>

> [!NOTE] 说明
>
>- AddFeatureByIndice接口不能和AddFeature、AddWithExtraVal接口混用。
>- 使用AddFeatureByIndice接口按位置添加底库之后，不能使用GetExtraValAttrByLabel等依赖Label的接口，AddFeatureByIndice和GetFeatureByIndice需配套使用。

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p4977325172214"><a name="p4977325172214"></a><a name="p4977325172214"></a>APP_ERROR AddFeatureByIndice(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *indices, const ExtraValAttr *extraVal = nullptr, const uint8_t *customAttr = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p84931034102319"><a name="p84931034102319"></a><a name="p84931034102319"></a>按照位置来添加底库特征。此接口当前只支持FlatIP和Int8Flat（cos距离）。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p16400852182313"><a name="p16400852182313"></a><a name="p16400852182313"></a><strong id="b12681434245"><a name="b12681434245"></a><a name="b12681434245"></a>int64_t count：</strong>待添加的特征数量。</p>
<p id="p1640015524237"><a name="p1640015524237"></a><a name="p1640015524237"></a><strong id="b3961719172416"><a name="b3961719172416"></a><a name="b3961719172416"></a>const void *features：</strong>待添加的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型；FP16Flat距离为float类型。</p>
<p id="p9400155214239"><a name="p9400155214239"></a><a name="p9400155214239"></a><strong id="b117145534246"><a name="b117145534246"></a><a name="b117145534246"></a>const FeatureAttr *attributes：</strong>待添加的特征属性。</p>
<p id="p940013526232"><a name="p940013526232"></a><a name="p940013526232"></a><strong id="b192668514250"><a name="b192668514250"></a><a name="b192668514250"></a>const int64_t *indices：</strong>待添加的特征在底库中的位置。</p>
<p id="p740065211237"><a name="p740065211237"></a><a name="p740065211237"></a><strong id="b11521184710255"><a name="b11521184710255"></a><a name="b11521184710255"></a>const ExtraValAttr *extraVal：</strong>待添加的附加特征属性。</p>
<p id="p2400165210237"><a name="p2400165210237"></a><a name="p2400165210237"></a><strong id="b2498951122512"><a name="b2498951122512"></a><a name="b2498951122512"></a>const uint8_t *customAttr：</strong>待添加的用户自定义特征属性。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><a name="ul101361314193612"></a><a name="ul101361314193612"></a><ul id="ul101361314193612"><li><span class="parmname" id="parmname5466122363620"><a name="parmname5466122363620"></a><a name="parmname5466122363620"></a>“count”</span>单次取值在[1, 1e6]区间，底库最大值1e9。</li><li><span class="parmname" id="parmname115333973614"><a name="parmname115333973614"></a><a name="parmname115333973614"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname128741704378"><a name="parmname128741704378"></a><a name="parmname128741704378"></a>“attributes”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname17821171853719"><a name="parmname17821171853719"></a><a name="parmname17821171853719"></a>“indices”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。需为严格递增且非负的值，值小于底库数量时表示替换，值大于等于底库数量时表示新增，此时值需要连续的。</li><li><span class="parmname" id="parmname842117618384"><a name="parmname842117618384"></a><a name="parmname842117618384"></a>“extraVal”</span>取值为空指针或者长度为count，否则可能出现越界读写错误并引起程序崩溃。取值为空指针时表示不需要添加附加属性。</li><li><span class="parmname" id="parmname4303122993814"><a name="parmname4303122993814"></a><a name="parmname4303122993814"></a>“customAttr”</span>取值为空指针或者长度为count * customAttrLen，否则可能出现越界读写错误并引起程序崩溃。取值为空指针时表示不需要添加自定义属性。</li></ul>
</td>
</tr>
</tbody>
</table>

#### AddWithExtraVal接口<a name="ZH-CN_TOPIC_0000001976650872"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p737617493548"><a name="p737617493548"></a><a name="p737617493548"></a>APP_ERROR AddWithExtraVal(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const ExtraValAttr *extraVal, const uint8_t *customAttr = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>添加附加属性特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1994144463213"><a name="p1994144463213"></a><a name="p1994144463213"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a>int64_t count</strong>：待添加的特征数量。</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a>const void *features</strong>：待添加的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1733711363162"><a name="b1733711363162"></a><a name="b1733711363162"></a>const FeatureAttr *attributes</strong>：待添加的特征属性，具体请参见<a href="#ZH-CN_TOPIC_0000001507967381">FeatureAttr</a>。</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b11413152220159"><a name="b11413152220159"></a><a name="b11413152220159"></a>const int64_t *labels</strong>：待添加的特征Label，使用上需要保证Label在Index实例中的唯一性。</p>
<p id="p14651655103714"><a name="p14651655103714"></a><a name="p14651655103714"></a><strong id="b48516115382"><a name="b48516115382"></a><a name="b48516115382"></a>const ExtraValAttr *extraVal</strong>：待添加的附加特征属性，具体请参见<a href="#ZH-CN_TOPIC_0000002013198657">ExtraValAttr</a>。</p>
<p id="p1087211592617"><a name="p1087211592617"></a><a name="p1087211592617"></a><strong id="b9522152852710"><a name="b9522152852710"></a><a name="b9522152852710"></a>const uint8_t *customAttr</strong>：待添加的用户自定义特征属性。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname817612520010"><a name="parmname817612520010"></a><a name="parmname817612520010"></a>“count”</span>单次取值在[1, 1e6]区间，底库最大值1e9。</li><li><span class="parmname" id="parmname13390711317"><a name="parmname13390711317"></a><a name="parmname13390711317"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname186181734113"><a name="parmname186181734113"></a><a name="parmname186181734113"></a>“attributes”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname16572106417"><a name="parmname16572106417"></a><a name="parmname16572106417"></a>“labels”</span>长度为count，各元素不重复且都不在底库中，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1116854619574"><a name="parmname1116854619574"></a><a name="parmname1116854619574"></a>“extraVal”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname17305193215019"><a name="parmname17305193215019"></a><a name="parmname17305193215019"></a>“customAttr”</span>取值为空指针或者count * customAttrLen，否则可能出现越界读写错误并引起程序崩溃；customAttrLen在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>或<a href="#ZH-CN_TOPIC_0000002013206217">InitWithExtraVal</a>设置。</li></ul>
</td>
</tr>
</tbody>
</table>

#### AscendIndexTS接口<a name="ZH-CN_TOPIC_0000001458200394"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a>AscendIndexTS() = default;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>AscendIndexTS的构造函数。</p>
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

<a name="table91172211633"></a>
<table><tbody><tr id="row21174211319"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2117202114312"><a name="p2117202114312"></a><a name="p2117202114312"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1642685832412"><a name="p1642685832412"></a><a name="p1642685832412"></a>AscendIndexTS(const AscendIndexTS &amp;) = delete;</p>
</td>
</tr>
<tr id="row81173211232"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p611714212318"><a name="p611714212318"></a><a name="p611714212318"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row61178219313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p131171921232"><a name="p131171921232"></a><a name="p131171921232"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b20557153917269"><a name="b20557153917269"></a><a name="b20557153917269"></a>const AscendIndexTS &amp;</strong>：AscendIndexTS对象。</p>
</td>
</tr>
<tr id="row14117162110310"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p91177215313"><a name="p91177215313"></a><a name="p91177215313"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1011742114316"><a name="p1011742114316"></a><a name="p1011742114316"></a>无</p>
</td>
</tr>
<tr id="row1911772112317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1511742113312"><a name="p1511742113312"></a><a name="p1511742113312"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8117221239"><a name="p8117221239"></a><a name="p8117221239"></a>无</p>
</td>
</tr>
<tr id="row141174211735"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1411719210317"><a name="p1411719210317"></a><a name="p1411719210317"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### \~AscendIndexTS接口<a name="ZH-CN_TOPIC_0000001507760865"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a>virtual ~AscendIndexTS() = default;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>AscendIndexTS的析构函数，销毁特征管理对象。</p>
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

#### DeleteFeatureByLabel接口<a name="ZH-CN_TOPIC_0000001458200398"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p71517584492"><a name="p71517584492"></a><a name="p71517584492"></a>APP_ERROR DeleteFeatureByLabel(int64_t count, const int64_t *labels);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>批量移除指定Label的特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1018221113350"><a name="p1018221113350"></a><a name="p1018221113350"></a><strong id="b518231114356"><a name="b518231114356"></a><a name="b518231114356"></a>int64_t count</strong>：待移除的特征数量。</p>
<p id="p0703157114817"><a name="p0703157114817"></a><a name="p0703157114817"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a>const int64_t *labels</strong>：特征Label。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname19528134136"><a name="parmname19528134136"></a><a name="parmname19528134136"></a>“count”</span>取值在[1, 1e6]区间。</li><li><span class="parmname" id="parmname9631106131"><a name="parmname9631106131"></a><a name="parmname9631106131"></a>“labels”</span>长度为count，各元素不重复且在底库实际存在，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### DeleteFeatureByToken接口<a id="ZH-CN_TOPIC_0000001458680018"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p610285755019"><a name="p610285755019"></a><a name="p610285755019"></a>APP_ERROR DeleteFeatureByToken(int64_t count, const uint32_t *tokens);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p121015103512"><a name="p121015103512"></a><a name="p121015103512"></a>批量移除指定Token ID的特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p960115103339"><a name="p960115103339"></a><a name="p960115103339"></a><strong id="b29989101715"><a name="b29989101715"></a><a name="b29989101715"></a>int64_t count</strong>：待删除的Token数量。</p>
<p id="p871394423613"><a name="p871394423613"></a><a name="p871394423613"></a><strong id="b2814139201"><a name="b2814139201"></a><a name="b2814139201"></a>const uint32_t *tokens</strong>：Token对应的ID。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul981181435015"></a><a name="ul981181435015"></a><ul id="ul981181435015"><li><span class="parmname" id="parmname19528134136"><a name="parmname19528134136"></a><a name="parmname19528134136"></a>“count”</span>取值在[1, 1e6]区间。</li><li><span class="parmname" id="parmname11410241112"><a name="parmname11410241112"></a><a name="parmname11410241112"></a>“tokens”</span>的长度为count，待移除的<span class="parmname" id="parmname1981881512012"><a name="parmname1981881512012"></a><a name="parmname1981881512012"></a>“tokens”</span>需要在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### FastDeleteFeatureByIndice接口<a name="ZH-CN_TOPIC_0000002445152089"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p6761137134011"><a name="p6761137134011"></a><a name="p6761137134011"></a>APP_ERROR FastDeleteFeatureByIndice(int64_t count, const int64_t *indices);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p188718287407"><a name="p188718287407"></a><a name="p188718287407"></a>按照位置来删除底库特征。此接口只支持TSFlatIP和TSInt8FlatCos的附加相似度场景。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p830817388409"><a name="p830817388409"></a><a name="p830817388409"></a><strong id="b7314134311402"><a name="b7314134311402"></a><a name="b7314134311402"></a>int64_t count：</strong>待删除的特征数量。</p>
<p id="p030813385400"><a name="p030813385400"></a><a name="p030813385400"></a><strong id="b280944614401"><a name="b280944614401"></a><a name="b280944614401"></a>const int64_t *indices：</strong>待删除的特征在底库中的位置。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1991445165319"></a><a name="ul1991445165319"></a><ul id="ul1991445165319"><li><span class="parmname" id="parmname6685111165411"><a name="parmname6685111165411"></a><a name="parmname6685111165411"></a>“count”</span>需要大于0，小于等于底库数量。</li><li><span class="parmname" id="parmname1479610219557"><a name="parmname1479610219557"></a><a name="parmname1479610219557"></a>“indices”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。值需大于等于0，小于底库数量。</li></ul>
</td>
</tr>
</tbody>
</table>

#### FastDeleteFeatureByRange接口<a name="ZH-CN_TOPIC_0000002445960745"></a>

<a name="table18950829154115"></a>
<table><tbody><tr id="row12950172911415"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p1950162910415"><a name="p1950162910415"></a><a name="p1950162910415"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1124533454213"><a name="p1124533454213"></a><a name="p1124533454213"></a>APP_ERROR FastDeleteFeatureByRange(int64_t start, int64_t count);</p>
</td>
</tr>
<tr id="row295015292419"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p179503291419"><a name="p179503291419"></a><a name="p179503291419"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p9695104615427"><a name="p9695104615427"></a><a name="p9695104615427"></a>从start位置来批量删除count个底库特征。此接口只支持TSFlatIP和TSInt8FlatCos的附加相似度场景。</p>
</td>
</tr>
<tr id="row119504292414"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p09501729114112"><a name="p09501729114112"></a><a name="p09501729114112"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p114134413433"><a name="p114134413433"></a><a name="p114134413433"></a><strong id="b124422894315"><a name="b124422894315"></a><a name="b124422894315"></a>int64_t start：</strong>批量删除的特征起始位置。</p>
<p id="p1741320420433"><a name="p1741320420433"></a><a name="p1741320420433"></a><strong id="b1149672012430"><a name="b1149672012430"></a><a name="b1149672012430"></a>int64_t count：</strong>批量删除的特征数量。</p>
</td>
</tr>
<tr id="row1795015292412"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p4950102984115"><a name="p4950102984115"></a><a name="p4950102984115"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p4950102910412"><a name="p4950102910412"></a><a name="p4950102910412"></a>无</p>
</td>
</tr>
<tr id="row1295062964115"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p12950102912412"><a name="p12950102912412"></a><a name="p12950102912412"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p995022912414"><a name="p995022912414"></a><a name="p995022912414"></a><strong id="b19950029124112"><a name="b19950029124112"></a><a name="b19950029124112"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row8950132974112"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p2095102954118"><a name="p2095102954118"></a><a name="p2095102954118"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul121991916566"></a><a name="ul121991916566"></a><ul id="ul121991916566"><li><span class="parmname" id="parmname968162718568"><a name="parmname968162718568"></a><a name="parmname968162718568"></a>“start”</span>需要大于等于0，小于底库数量。</li><li><span class="parmname" id="parmname1141313911563"><a name="parmname1141313911563"></a><a name="parmname1141313911563"></a>“count”</span>需要大于0，小于等于底库数量。</li><li><span class="parmname" id="parmname128761351195616"><a name="parmname128761351195616"></a><a name="parmname128761351195616"></a>“start”</span>与<span class="parmname" id="parmname11829205813568"><a name="parmname11829205813568"></a><a name="parmname11829205813568"></a>“count”</span>的和小于等于底库数量。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetBaseByRange接口<a name="ZH-CN_TOPIC_0000001818301380"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p312319560281"><a name="p312319560281"></a><a name="p312319560281"></a>APP_ERROR GetBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>基于范围查询底库。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b168047345217"><a name="b168047345217"></a><a name="b168047345217"></a>uint32_t offset</strong>：获取底库特征初始偏移值。</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b19447237624"><a name="b19447237624"></a><a name="b19447237624"></a>uint32_t num</strong>：特征数量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a>int64_t *labels</strong>：特征Label。</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a>void *features</strong>：特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型；FP16Flat距离为float类型。</p>
<p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b2042017420403"><a name="b2042017420403"></a><a name="b2042017420403"></a>FeatureAttr *attributes</strong>：特征属性。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.6.1 "><a name="ul1652965944519"></a><a name="ul1652965944519"></a><ul id="ul1652965944519"><li>0＜ offset ≤8.0e8</li><li>0＜ num ≤8.0e8</li><li>offset＋num ≤ ntotal</li><li><span class="parmname" id="parmname204541533928"><a name="parmname204541533928"></a><a name="parmname204541533928"></a>“labels”</span>长度为num，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1443963511216"><a name="parmname1443963511216"></a><a name="parmname1443963511216"></a>“features”</span>长度为num * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1481821245019"><a name="parmname1481821245019"></a><a name="parmname1481821245019"></a>“attributes”</span>长度为num，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetBaseByRangeWithExtraVal接口<a name="ZH-CN_TOPIC_0000001976495686"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p1695861915578"><a name="p1695861915578"></a><a name="p1695861915578"></a>APP_ERROR GetBaseByRangeWithExtraVal(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes, ExtraValAttr *extraVal) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>基于范围查询带附加属性的底库。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b138871842534"><a name="b138871842534"></a><a name="b138871842534"></a>uint32_t offset</strong>：获取底库特征初始偏移值。</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b13710463319"><a name="b13710463319"></a><a name="b13710463319"></a>uint32_t num</strong>：特征数量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a>int64_t *labels</strong>：特征Label。</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a>void *features</strong>：特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型。</p>
<p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b2042017420403"><a name="b2042017420403"></a><a name="b2042017420403"></a>FeatureAttr *attributes</strong>：特征属性。</p>
<p id="p584735635516"><a name="p584735635516"></a><a name="p584735635516"></a><strong id="b024346145617"><a name="b024346145617"></a><a name="b024346145617"></a>ExtraValAttr *extraVal</strong>：附加属性。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.6.1 "><a name="ul1652965944519"></a><a name="ul1652965944519"></a><ul id="ul1652965944519"><li>0&lt;=offset&lt;8.0e8</li><li>0＜ num ≤8.0e8</li><li>offset＋num ≤ ntotal</li><li><span class="parmname" id="parmname204541533928"><a name="parmname204541533928"></a><a name="parmname204541533928"></a>“labels”</span>长度为num，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1443963511216"><a name="parmname1443963511216"></a><a name="parmname1443963511216"></a>“features”</span>长度为num * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1481821245019"><a name="parmname1481821245019"></a><a name="parmname1481821245019"></a>“attributes”</span>长度为num，否则可能出现越界读写错误并引起程序崩溃。</li><li><strong id="b13132123625719"><a name="b13132123625719"></a><a name="b13132123625719"></a><span class="parmname" id="parmname5995457205713"><a name="parmname5995457205713"></a><a name="parmname5995457205713"></a>“extraVal”</span></strong>长度为num，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetBaseMask接口<a name="ZH-CN_TOPIC_0000002445112157"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1257715237445"><a name="p1257715237445"></a><a name="p1257715237445"></a>APP_ERROR GetBaseMask(int64_t count, uint8_t *mask);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1228610371441"><a name="p1228610371441"></a><a name="p1228610371441"></a>获取底库是否被快速删除的标志。如果某个bit位上为0，表示该位置的底库被删除了，是无效的。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p15465289457"><a name="p15465289457"></a><a name="p15465289457"></a><strong id="b166204535358"><a name="b166204535358"></a><a name="b166204535358"></a>int64_t count：</strong>mask数组有效长度。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p2022018482356"><a name="p2022018482356"></a><a name="p2022018482356"></a><strong id="b99311008366"><a name="b99311008366"></a><a name="b99311008366"></a>uint8_t *mask：</strong>标记底库是否被删除的数组。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul11392164455912"></a><a name="ul11392164455912"></a><ul id="ul11392164455912"><li><span class="parmname" id="parmname15731754125910"><a name="parmname15731754125910"></a><a name="parmname15731754125910"></a>“count”</span>值需为[1, ceil(ntotal/8)]，否则可能出现越界读写错误并引起程序崩溃。其中，ntotal为底库特征数量。</li><li><span class="parmname" id="parmname1571614575596"><a name="parmname1571614575596"></a><a name="parmname1571614575596"></a>“mask”</span>长度需大于等于count，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetCustomAttrByBlockId接口<a name="ZH-CN_TOPIC_0000001736682593"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2398651133314"><a name="p2398651133314"></a><a name="p2398651133314"></a>APP_ERROR GetCustomAttrByBlockId(uint32_t blockId, uint8_t *&amp;customAttr) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p121015103512"><a name="p121015103512"></a><a name="p121015103512"></a>获取指定blockId的自定义属性。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p960115103339"><a name="p960115103339"></a><a name="p960115103339"></a><strong id="b64714843417"><a name="b64714843417"></a><a name="b64714843417"></a>uint32_t blockId</strong>：待获取的blockId。</p>
<p id="p871394423613"><a name="p871394423613"></a><a name="p871394423613"></a><strong id="b1231449164011"><a name="b1231449164011"></a><a name="b1231449164011"></a>uint8_t *&amp;customAttr</strong>：Device侧的用户自定义特征属性。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p2142161113218"><a name="p2142161113218"></a><a name="p2142161113218"></a><span class="parmname" id="parmname6113951121"><a name="parmname6113951121"></a><a name="parmname6113951121"></a>“customAttr”</span>长度为customAttrBlockSize * customAttrLen，否则可能出现越界读写错误并引起程序崩溃。customAttrBlockSize和customAttrLen在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>或<a href="#ZH-CN_TOPIC_0000002013206217">InitWithExtraVal</a>设置。</p>
</td>
</tr>
</tbody>
</table>

#### GetExtraValAttrByLabel接口<a name="ZH-CN_TOPIC_0000001976655414"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1723162213589"><a name="p1723162213589"></a><a name="p1723162213589"></a>APP_ERROR GetExtraValAttrByLabel(int64_t count, const int64_t *labels, ExtraValAttr *extraVal) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取指定Label特征的附加属性。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p92901840173714"><a name="p92901840173714"></a><a name="p92901840173714"></a><strong id="b274355824717"><a name="b274355824717"></a><a name="b274355824717"></a>int64_t count</strong>：获取特征的数量。</p>
<p id="p4672543173915"><a name="p4672543173915"></a><a name="p4672543173915"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a>const int64_t *labels</strong>：特征Label。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b12012305910"><a name="b12012305910"></a><a name="b12012305910"></a>ExtraValAttr *extraVal</strong>：附加属性。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname526614311120"><a name="parmname526614311120"></a><a name="parmname526614311120"></a>“count”</span>取值在[1, 1e6]区间。</li><li><span class="parmname" id="parmname204541533928"><a name="parmname204541533928"></a><a name="parmname204541533928"></a>“labels”</span>长度为count，各元素不重复且在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。如输入的<span class="parmname" id="parmname12554195016103"><a name="parmname12554195016103"></a><a name="parmname12554195016103"></a>“labels”</span>不存在底库中，接口返回的附加属性中，<span class="parmname" id="parmname5939135241411"><a name="parmname5939135241411"></a><a name="parmname5939135241411"></a>“val”</span>为<span class="parmvalue" id="parmvalue7121125615148"><a name="parmvalue7121125615148"></a><a name="parmvalue7121125615148"></a>“INT16_MIN”</span>。</li><li><span class="parmname" id="parmname2885152415137"><a name="parmname2885152415137"></a><a name="parmname2885152415137"></a>“extraVal”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetFeatureAttrByLabel接口<a name="ZH-CN_TOPIC_0000001594544301"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p129110403377"><a name="p129110403377"></a><a name="p129110403377"></a>APP_ERROR GetFeatureAttrByLabel(int64_t count, const int64_t *labels, FeatureAttr *attributes) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取指定Label特征的属性。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p92901840173714"><a name="p92901840173714"></a><a name="p92901840173714"></a><strong id="b274355824717"><a name="b274355824717"></a><a name="b274355824717"></a>int64_t count</strong>：获取特征的数量。</p>
<p id="p4672543173915"><a name="p4672543173915"></a><a name="p4672543173915"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a>const int64_t *labels</strong>：特征Label。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b2042017420403"><a name="b2042017420403"></a><a name="b2042017420403"></a>FeatureAttr *attributes</strong>：特征属性。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname526614311120"><a name="parmname526614311120"></a><a name="parmname526614311120"></a>“count”</span>取值在[1, 1e6]区间。</li><li><span class="parmname" id="parmname204541533928"><a name="parmname204541533928"></a><a name="parmname204541533928"></a>“labels”</span>长度为count，各元素不重复且在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。如输入的<span class="parmname" id="parmname12554195016103"><a name="parmname12554195016103"></a><a name="parmname12554195016103"></a>“labels”</span>不存在底库中，接口返回的特征属性中，<span class="parmname" id="parmname5939135241411"><a name="parmname5939135241411"></a><a name="parmname5939135241411"></a>“time”</span>为<span class="parmvalue" id="parmvalue7121125615148"><a name="parmvalue7121125615148"></a><a name="parmvalue7121125615148"></a>“INT32_MIN”</span>，<span class="parmname" id="parmname477017482136"><a name="parmname477017482136"></a><a name="parmname477017482136"></a>“tokenId”</span>为<span class="parmvalue" id="parmvalue198681557181415"><a name="parmvalue198681557181415"></a><a name="parmvalue198681557181415"></a>“UINT32_MAX”</span>。</li><li><span class="parmname" id="parmname9326453141512"><a name="parmname9326453141512"></a><a name="parmname9326453141512"></a>“attributes”</span>的长度为count，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetFeatureByIndice接口<a name="ZH-CN_TOPIC_0000002411592888"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p987171553412"><a name="p987171553412"></a><a name="p987171553412"></a>APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels = nullptr, void *features = nullptr, FeatureAttr *attributes = nullptr, ExtraValAttr *extraVal = nullptr) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p14469247153411"><a name="p14469247153411"></a><a name="p14469247153411"></a>按照位置来获取底库特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p204841192352"><a name="p204841192352"></a><a name="p204841192352"></a><strong id="b217916181358"><a name="b217916181358"></a><a name="b217916181358"></a>int64_t count：</strong>待获取的特征数量。</p>
<p id="p648439143515"><a name="p648439143515"></a><a name="p648439143515"></a><strong id="b1925652253519"><a name="b1925652253519"></a><a name="b1925652253519"></a>const int64_t *indices：</strong>待获取的特征在底库中的位置。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p184841691354"><a name="p184841691354"></a><a name="p184841691354"></a><strong id="b6544142883511"><a name="b6544142883511"></a><a name="b6544142883511"></a>int64_t *labels：</strong>待获取的特征对应的label。</p>
<p id="p16484892356"><a name="p16484892356"></a><a name="p16484892356"></a><strong id="b1177389353"><a name="b1177389353"></a><a name="b1177389353"></a>void *features：</strong>待获取的特征向量。</p>
<p id="p1448489143512"><a name="p1448489143512"></a><a name="p1448489143512"></a><strong id="b157791442193515"><a name="b157791442193515"></a><a name="b157791442193515"></a>FeatureAttr *attributes：</strong>待获取的特征时空属性。</p>
<p id="p124841495355"><a name="p124841495355"></a><a name="p124841495355"></a><strong id="b1896914513357"><a name="b1896914513357"></a><a name="b1896914513357"></a>ExtraValAttr *extraVal：</strong>待获取的特征额外属性。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><a name="ul1365255411394"></a><a name="ul1365255411394"></a><ul id="ul1365255411394"><li><span class="parmname" id="parmname1672452818408"><a name="parmname1672452818408"></a><a name="parmname1672452818408"></a>“count”</span>单次取值在[1, 1e6]区间。</li><li><span class="parmname" id="parmname9525183974014"><a name="parmname9525183974014"></a><a name="parmname9525183974014"></a>“indices”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。值需大于等于0，小于底库数量。</li><li><span class="parmname" id="parmname11492197414"><a name="parmname11492197414"></a><a name="parmname11492197414"></a>“labels”</span>为<span class="parmvalue" id="parmvalue1553619154115"><a name="parmvalue1553619154115"></a><a name="parmvalue1553619154115"></a>“nullptr”</span>时表示不用获取，或者长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname19820115116417"><a name="parmname19820115116417"></a><a name="parmname19820115116417"></a>“features”</span>为<span class="parmvalue" id="parmvalue14150359114115"><a name="parmvalue14150359114115"></a><a name="parmvalue14150359114115"></a>“nullptr”</span>时表示不用获取，或者长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname158121424134216"><a name="parmname158121424134216"></a><a name="parmname158121424134216"></a>“attributes”</span>为<span class="parmvalue" id="parmvalue562163013421"><a name="parmvalue562163013421"></a><a name="parmvalue562163013421"></a>“nullptr”</span>时表示不用获取，或者长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1585165624219"><a name="parmname1585165624219"></a><a name="parmname1585165624219"></a>“extraVal”</span>为<span class="parmvalue" id="parmvalue5852165910421"><a name="parmvalue5852165910421"></a><a name="parmvalue5852165910421"></a>“nullptr”</span>时表示不用获取，或者长度为count，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetFeatureByLabel接口<a name="ZH-CN_TOPIC_0000001507879789"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p410994614718"><a name="p410994614718"></a><a name="p410994614718"></a>APP_ERROR GetFeatureByLabel(int64_t count, const int64_t *labels, void *features);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取指定Label的特征。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p6644175415341"><a name="p6644175415341"></a><a name="p6644175415341"></a><strong id="b274355824717"><a name="b274355824717"></a><a name="b274355824717"></a>int64_t count</strong>：获取特征的数量。</p>
<p id="p0703157114817"><a name="p0703157114817"></a><a name="p0703157114817"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a>const int64_t *labels</strong>：特征Label。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p136503012544"><a name="p136503012544"></a><a name="p136503012544"></a><strong id="b818316576311"><a name="b818316576311"></a><a name="b818316576311"></a>void *features</strong>：根据指定Label获取的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat距离为float类型。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname526614311120"><a name="parmname526614311120"></a><a name="parmname526614311120"></a>“count”</span>取值在[1, 1e6]区间。</li><li><span class="parmname" id="parmname204541533928"><a name="parmname204541533928"></a><a name="parmname204541533928"></a>“labels”</span>长度为count，各元素不重复且在底库中实际存在，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1443963511216"><a name="parmname1443963511216"></a><a name="parmname1443963511216"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### GetFeatureNum接口<a name="ZH-CN_TOPIC_0000001544946953"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p312319560281"><a name="p312319560281"></a><a name="p312319560281"></a>APP_ERROR GetFeatureNum(int64_t *totalNum)；</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取该Index实例中的特征条数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p0703157114817"><a name="p0703157114817"></a><a name="p0703157114817"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p1658819198309"><a name="p1658819198309"></a><a name="p1658819198309"></a><strong id="b10588219113017"><a name="b10588219113017"></a><a name="b10588219113017"></a>int64_t *totalNum</strong>：底库中特征的数量。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p20431123883017"><a name="p20431123883017"></a><a name="p20431123883017"></a>无</p>
</td>
</tr>
</tbody>
</table>

#### Init接口<a id="ZH-CN_TOPIC_0000001458680014"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1177454219213"><a name="p1177454219213"></a><a name="p1177454219213"></a>APP_ERROR Init(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, AlgorithmType algType = AlgorithmType::FLAT_COS_INT8, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实例初始化函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b57154112218"><a name="b57154112218"></a><a name="b57154112218"></a>uint32_t deviceId</strong>：Index使用的设备ID。</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b1137774510218"><a name="b1137774510218"></a><a name="b1137774510218"></a>uint32_t dim</strong>：底库向量的维度。</p>
<p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b1192712462213"><a name="b1192712462213"></a><a name="b1192712462213"></a>uint32_t tokenNum</strong>：当前时空库Token的最大数量，需要和生成对应的Mask算子Token数量一致。</p>
<p id="p13858143610478"><a name="p13858143610478"></a><a name="p13858143610478"></a><strong id="b12145453824"><a name="b12145453824"></a><a name="b12145453824"></a>AlgorithmType algType</strong>：底层使用的距离比对算法，默认为<span class="parmvalue" id="parmvalue69446331337"><a name="parmvalue69446331337"></a><a name="parmvalue69446331337"></a>“AlgorithmType::FLAT_COS_INT8”</span>，可选算法参见如下。</p>
<a name="ul7984114704715"></a><a name="ul7984114704715"></a><ul id="ul7984114704715"><li><span class="parmvalue" id="parmvalue19768835143"><a name="parmvalue19768835143"></a><a name="parmvalue19768835143"></a>“AlgorithmType::FLAT_HAMMING”</span>：二值化特征检索（汉明距离）。</li><li><span class="parmvalue" id="parmvalue4540183574817"><a name="parmvalue4540183574817"></a><a name="parmvalue4540183574817"></a>“AlgorithmType::FLAT_COS_INT8”</span>：Int8Flat（cos距离）。</li><li><span class="parmvalue" id="parmvalue598518449264"><a name="parmvalue598518449264"></a><a name="parmvalue598518449264"></a>“AlgorithmType::FLAT_L2_INT8”</span>：Int8Flat（L2距离）。</li><li><span class="parmvalue" id="parmvalue28001772171"><a name="parmvalue28001772171"></a><a name="parmvalue28001772171"></a>“AlgorithmType::FLAT_IP_FP16”</span>：FP16Flat（IP距离）。</li><li><span class="parmvalue" id="parmvalue334582520494"><a name="parmvalue334582520494"></a><a name="parmvalue334582520494"></a>“AlgorithmType::FLAT_HPP_COS_INT8”</span>：Int8Flat（cos距离）。</li></ul>
<div class="p" id="p511515103547"><a name="p511515103547"></a><a name="p511515103547"></a><strong id="b18177046143817"><a name="b18177046143817"></a><a name="b18177046143817"></a>MemoryStrategy memoryStrategy</strong>：底层使用的内存策略，默认为<span class="parmvalue" id="parmvalue132807333394"><a name="parmvalue132807333394"></a><a name="parmvalue132807333394"></a>“MemoryStrategy::PURE_DEVICE_MEMORY”</span>，可选策略参见如下。<a name="ul6481121833912"></a><a name="ul6481121833912"></a><ul id="ul6481121833912"><li>MemoryStrategy::PURE_DEVICE_MEMORY：纯Device内存策略。</li><li>MemoryStrategy::HETERO_MEMORY：异构内存策略。</li><li>MemoryStrategy::HPP：HPP的异构内存策略。</li></ul>
</div>
<p id="p14841631173010"><a name="p14841631173010"></a><a name="p14841631173010"></a><strong id="b1383382923017"><a name="b1383382923017"></a><a name="b1383382923017"></a>customAttrLen</strong>：自定义属性长度。</p>
<p id="p133252511308"><a name="p133252511308"></a><a name="p133252511308"></a><strong id="b92382483110"><a name="b92382483110"></a><a name="b92382483110"></a>customAttrBlockSize</strong>：自定义属性blocksize的大小。</p>
<p id="p195521819124"><a name="p195521819124"></a><a name="p195521819124"></a><strong id="b19493104084719"><a name="b19493104084719"></a><a name="b19493104084719"></a>maxFeatureRowCount：</strong>底库最大向量条数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p157591724172420"><a name="p157591724172420"></a><a name="p157591724172420"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1233201818419"></a><a name="ul1233201818419"></a><ul id="ul1233201818419"><li>需要紧跟在构造函数后调用。</li><li><span class="parmname" id="parmname129911241505"><a name="parmname129911241505"></a><a name="parmname129911241505"></a>“deviceId”</span>为有效的设备ID，设置范围为[0, 1024]。</li><li><span class="parmname" id="parmname1512023074919"><a name="parmname1512023074919"></a><a name="parmname1512023074919"></a>“tokenNum”</span>设置范围为(0, 3e5]。</li><li>对于二值化特征检索（汉明距离）算法，dim ∈ {256, 512, 1024}。</li><li>对于Int8Flat（cos距离、L2距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}；对于FP16Flat（IP距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li><span class="parmname" id="parmname121853561610"><a name="parmname121853561610"></a><a name="parmname121853561610"></a>“memoryStrategy::HETERO_MEMORY”</span>当前只支持“AlgorithmType::FLAT_COS_INT8”算法。</li><li><span class="parmname" id="parmname20787447125320"><a name="parmname20787447125320"></a><a name="parmname20787447125320"></a>“customAttrLen”</span>设置范围为[0, 32]，默认值为<span class="parmvalue" id="parmvalue297865405619"><a name="parmvalue297865405619"></a><a name="parmvalue297865405619"></a>“0”</span>，设置为<span class="parmvalue" id="parmvalue932814529584"><a name="parmvalue932814529584"></a><a name="parmvalue932814529584"></a>“0”</span>时表示无自定义属性。</li><li><span class="parmname" id="parmname8753142175415"><a name="parmname8753142175415"></a><a name="parmname8753142175415"></a>“customAttrBlockSize”</span>设置范围为[0, 262144*64]，需要为1024*256的整数倍。默认值为<span class="parmvalue" id="parmvalue124141463574"><a name="parmvalue124141463574"></a><a name="parmvalue124141463574"></a>“0”</span>，设置为<span class="parmvalue" id="parmvalue834813319594"><a name="parmvalue834813319594"></a><a name="parmvalue834813319594"></a>“0”</span>时表示无自定义属性。</li><li><span class="parmname" id="parmname435216805314"><a name="parmname435216805314"></a><a name="parmname435216805314"></a>“maxFeatureRowCount”</span>设置范围为[262144 * 64, 262144 * 550 *3]，需要为256的整数倍。默认值为uint64的最大值。该参数只在<span class="parmname" id="parmname7634205023214"><a name="parmname7634205023214"></a><a name="parmname7634205023214"></a>“MemoryStrategy memoryStrategy”</span>设置为<span class="parmvalue" id="parmvalue10568101063610"><a name="parmvalue10568101063610"></a><a name="parmvalue10568101063610"></a>“MemoryStrategy::HPP”</span>时有效。</li><li>当<strong id="b44445251175"><a name="b44445251175"></a><a name="b44445251175"></a>MemoryStrategy memoryStrategy</strong>设置为<span class="parmvalue" id="parmvalue1054510476713"><a name="parmvalue1054510476713"></a><a name="parmvalue1054510476713"></a>“MemoryStrategy::HPP”</span>时，Host侧的可用内存需要大于等于250GB、空闲CPU物理核数需要大于等于15核，且目前仅支持256维向量的检索。</li></ul>
</td>
</tr>
</tbody>
</table>

#### InitWithExtraVal接口<a id="ZH-CN_TOPIC_0000002013206217"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p201951631145314"><a name="p201951631145314"></a><a name="p201951631145314"></a>APP_ERROR InitWithExtraVal(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, AlgorithmType algType = AlgorithmType::FLAT_HAMMING, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>实例带附加属性的初始化函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b57154112218"><a name="b57154112218"></a><a name="b57154112218"></a>uint32_t deviceId</strong>：Index使用的设备ID。</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b1137774510218"><a name="b1137774510218"></a><a name="b1137774510218"></a>uint32_t dim</strong>：底库向量的维度。</p>
<p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b1192712462213"><a name="b1192712462213"></a><a name="b1192712462213"></a>uint32_t tokenNum</strong>：当前时空库Token的最大数量，需要和生成对应的Mask算子Token数量一致。</p>
<p id="p19371341202917"><a name="p19371341202917"></a><a name="p19371341202917"></a><strong id="b88105533316"><a name="b88105533316"></a><a name="b88105533316"></a>uint64_t resources</strong>：共享内存大小。</p>
<p id="p13858143610478"><a name="p13858143610478"></a><a name="p13858143610478"></a><strong id="b12145453824"><a name="b12145453824"></a><a name="b12145453824"></a>AlgorithmType algType</strong>：底层使用的距离比对算法，默认为<span class="parmvalue" id="parmvalue69446331337"><a name="parmvalue69446331337"></a><a name="parmvalue69446331337"></a>“AlgorithmType::FLAT_HAMMING”。</span>可选算法参见如下。</p>
<a name="ul7984114704715"></a><a name="ul7984114704715"></a><ul id="ul7984114704715"><li><span class="parmvalue" id="parmvalue19768835143"><a name="parmvalue19768835143"></a><a name="parmvalue19768835143"></a>“AlgorithmType::FLAT_HAMMING”</span>：二值化特征检索（汉明距离）。</li><li><span class="parmvalue" id="parmvalue4540183574817"><a name="parmvalue4540183574817"></a><a name="parmvalue4540183574817"></a>“AlgorithmType::FLAT_COS_INT8”</span>：Int8Flat（cos距离）。</li></ul>
<div class="p" id="p511515103547"><a name="p511515103547"></a><a name="p511515103547"></a><strong id="b18177046143817"><a name="b18177046143817"></a><a name="b18177046143817"></a>MemoryStrategy memoryStrategy</strong>：底层使用的内存策略，默认为<span class="parmvalue" id="parmvalue132807333394"><a name="parmvalue132807333394"></a><a name="parmvalue132807333394"></a>“MemoryStrategy::PURE_DEVICE_MEMORY”</span>，可选策略参见如下。<a name="ul6481121833912"></a><a name="ul6481121833912"></a><ul id="ul6481121833912"><li>MemoryStrategy::PURE_DEVICE_MEMORY：纯Device内存策略。</li><li>MemoryStrategy::HETERO_MEMORY：异构内存策略。</li></ul>
</div>
<p id="p14841631173010"><a name="p14841631173010"></a><a name="p14841631173010"></a><strong id="b1383382923017"><a name="b1383382923017"></a><a name="b1383382923017"></a>customAttrLen</strong>：自定义属性长度。</p>
<p id="p133252511308"><a name="p133252511308"></a><a name="p133252511308"></a><strong id="b92382483110"><a name="b92382483110"></a><a name="b92382483110"></a>customAttrBlockSize</strong>：自定义属性blocksize的大小。</p>
<p id="p195521819124"><a name="p195521819124"></a><a name="p195521819124"></a><strong id="b19493104084719"><a name="b19493104084719"></a><a name="b19493104084719"></a>maxFeatureRowCount：</strong>底库最大向量条数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1233201818419"></a><a name="ul1233201818419"></a><ul id="ul1233201818419"><li>需要紧跟在构造函数后调用。</li><li><span class="parmname" id="parmname129911241505"><a name="parmname129911241505"></a><a name="parmname129911241505"></a>“deviceId”</span>为有效的设备ID，设置范围为[0, 1024]。</li><li><span class="parmname" id="parmname1512023074919"><a name="parmname1512023074919"></a><a name="parmname1512023074919"></a>“tokenNum”</span>设置范围为(0, 3e5]。</li><li><span class="parmname" id="parmname17504112111345"><a name="parmname17504112111345"></a><a name="parmname17504112111345"></a>“uint64_t resources”</span>合法范围为[1*1024*1024*1024, 32*1024*1024*1024]，使用附加属性时推荐申请4GB。</li><li>对于二值化特征检索（汉明距离）算法，dim ∈ {256, 512, 1024}。</li><li>对于Int8Flat（cos距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}</li><li><span class="parmname" id="parmname20787447125320"><a name="parmname20787447125320"></a><a name="parmname20787447125320"></a>“customAttrLen”</span>设置范围为[0, 32]，默认值为<span class="parmvalue" id="parmvalue297865405619"><a name="parmvalue297865405619"></a><a name="parmvalue297865405619"></a>“0”</span>，设置为<span class="parmvalue" id="parmvalue932814529584"><a name="parmvalue932814529584"></a><a name="parmvalue932814529584"></a>“0”</span>时表示无自定义属性。</li><li><span class="parmname" id="parmname8753142175415"><a name="parmname8753142175415"></a><a name="parmname8753142175415"></a>“customAttrBlockSize”</span>设置范围为[0, 262144*64]，需要为1024*256的整数倍。默认值为<span class="parmvalue" id="parmvalue124141463574"><a name="parmvalue124141463574"></a><a name="parmvalue124141463574"></a>“0”</span>，设置为<span class="parmvalue" id="parmvalue834813319594"><a name="parmvalue834813319594"></a><a name="parmvalue834813319594"></a>“0”</span>时表示无自定义属性。</li><li><span class="parmname" id="parmname435216805314"><a name="parmname435216805314"></a><a name="parmname435216805314"></a>“maxFeatureRowCount”</span>附加属性不支持HPP，默认为uint64的最大值。</li></ul>
</td>
</tr>
</tbody>
</table>

#### InitWithQuantify接口<a name="ZH-CN_TOPIC_0000002458673509"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2056224963619"><a name="p2056224963619"></a><a name="p2056224963619"></a>APP_ERROR InitWithQuantify(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, const float *scale, AlgorithmType algType = AlgorithmType::FLAT_IP_FP16, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>底库向量化初始化接口。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p11571112514013"><a name="p11571112514013"></a><a name="p11571112514013"></a><strong id="b3472938164012"><a name="b3472938164012"></a><a name="b3472938164012"></a>uint32_t deviceId：</strong>Index使用的设备ID。</p>
<p id="p957192544013"><a name="p957192544013"></a><a name="p957192544013"></a><strong id="b118141148114017"><a name="b118141148114017"></a><a name="b118141148114017"></a>uint32_t dim：</strong>底库向量的维度。</p>
<p id="p1857110253409"><a name="p1857110253409"></a><a name="p1857110253409"></a><strong id="b1497619074111"><a name="b1497619074111"></a><a name="b1497619074111"></a>uint32_t tokenNum：</strong>当前时空库Token的最大数量，需要和生成对应的Mask算子Token数量一致。</p>
<p id="p457112584010"><a name="p457112584010"></a><a name="p457112584010"></a><strong id="b153364211415"><a name="b153364211415"></a><a name="b153364211415"></a>uint64_t resources：</strong>共享内存大小。</p>
<p id="p1257132594013"><a name="p1257132594013"></a><a name="p1257132594013"></a><strong id="b567513917419"><a name="b567513917419"></a><a name="b567513917419"></a>const float *scale：</strong>底库向量化缩放因子。缩放因子和底库相乘后转化为int8_t类型。</p>
<p id="p1357118253407"><a name="p1357118253407"></a><a name="p1357118253407"></a><strong id="b116721651194112"><a name="b116721651194112"></a><a name="b116721651194112"></a>AlgorithmType algType：</strong>底层使用的距离比对算法。默认为“AlgorithmType:: FLAT_IP_FP16”，表示FP16Flat（IP距离），目前仅支持AlgorithmType:: FLAT_IP_FP16算法。</p>
<p id="p668831612439"><a name="p668831612439"></a><a name="p668831612439"></a><strong id="b8861135274318"><a name="b8861135274318"></a><a name="b8861135274318"></a>uint32_t customAttrLen：</strong>自定义属性长度。</p>
<p id="p1649111854311"><a name="p1649111854311"></a><a name="p1649111854311"></a><strong id="b146321459440"><a name="b146321459440"></a><a name="b146321459440"></a>uint32_t customAttrBlockSize：</strong>自定义属性blocksize的大小。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1233201818419"></a><a name="ul1233201818419"></a><ul id="ul1233201818419"><li>需要紧跟在构造函数后调用。</li><li><span class="parmname" id="parmname129911241505"><a name="parmname129911241505"></a><a name="parmname129911241505"></a>“deviceId”</span>为有效的设备ID，设置范围为[0, 1024]。</li><li><span class="parmname" id="parmname1512023074919"><a name="parmname1512023074919"></a><a name="parmname1512023074919"></a>“tokenNum”</span>设置范围为(0, 3e5]。</li><li><span class="parmname" id="parmname17504112111345"><a name="parmname17504112111345"></a><a name="parmname17504112111345"></a>“uint64_t resources”</span>合法范围为大于0小于等于4*1024*1024*1024。</li><li>Scale反量化时需要进行除运算，不能接近0；Scale中因子绝对值大于等于1e-6f。</li><li>对于FP16Flat（IP距离）算法，dim ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li>当前只支持FP16Flat（IP距离）算法的非共享模式。</li><li>本接口和AddFeatureByIndice配套使用。</li><li><span class="parmname" id="parmname20787447125320"><a name="parmname20787447125320"></a><a name="parmname20787447125320"></a>“customAttrLen”</span>设置范围为[0, 32]，默认值为<span class="parmvalue" id="parmvalue297865405619"><a name="parmvalue297865405619"></a><a name="parmvalue297865405619"></a>“0”</span>，设置为<span class="parmvalue" id="parmvalue932814529584"><a name="parmvalue932814529584"></a><a name="parmvalue932814529584"></a>“0”</span>时表示无自定义属性。</li><li><span class="parmname" id="parmname8753142175415"><a name="parmname8753142175415"></a><a name="parmname8753142175415"></a>“customAttrBlockSize”</span>设置范围为[0, 262144*64]，需要为1024*256的整数倍。默认值为<span class="parmvalue" id="parmvalue124141463574"><a name="parmvalue124141463574"></a><a name="parmvalue124141463574"></a>“0”</span>，设置为<span class="parmvalue" id="parmvalue834813319594"><a name="parmvalue834813319594"></a><a name="parmvalue834813319594"></a>“0”</span>时表示无自定义属性。</li></ul>
</td>
</tr>
</tbody>
</table>

#### operator = 接口<a name="ZH-CN_TOPIC_0000001507959881"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p26981713282"><a name="p26981713282"></a><a name="p26981713282"></a>AscendIndexTS &amp;operator = (const AscendIndexTS &amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b6551145181714"><a name="b6551145181714"></a><a name="b6551145181714"></a>const AscendIndexTS &amp;</strong>：常量AscendIndexTS。</p>
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

#### Search接口<a name="ZH-CN_TOPIC_0000001507640109"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1722511394427"><a name="p1722511394427"></a><a name="p1722511394427"></a>APP_ERROR Search(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, bool enableTimeFilter = true);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>计算输入特征和经过AttrFilter过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p2360715171619"><a name="p2360715171619"></a><a name="p2360715171619"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a>uint32_t count</strong>：待比较的特征数量。</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a>const void *features</strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat为float类型。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b161793465431"><a name="b161793465431"></a><a name="b161793465431"></a>const AttrFilter *attrFilter</strong>：属性过滤信息，具体请参见<a href="#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。</p>
<p id="p638962561711"><a name="p638962561711"></a><a name="p638962561711"></a><strong id="b72082417127"><a name="b72082417127"></a><a name="b72082417127"></a>bool shareAttrFilter</strong>：不同query是否共享一个mask。</p>
<p id="p263216178448"><a name="p263216178448"></a><a name="p263216178448"></a><strong id="b9237122811443"><a name="b9237122811443"></a><a name="b9237122811443"></a>uint32_t topk</strong>：计算余弦距离后需要保存的TopK大小。</p>
<p id="p198449441221"><a name="p198449441221"></a><a name="p198449441221"></a><strong id="b194349562317"><a name="b194349562317"></a><a name="b194349562317"></a>bool enableTimeFilter</strong>：时间戳属性过滤开关，默认为<span class="parmvalue" id="parmvalue6379218248"><a name="parmvalue6379218248"></a><a name="parmvalue6379218248"></a>“true”</span>，当<strong id="b12428111462413"><a name="b12428111462413"></a><a name="b12428111462413"></a>enableTimeFilter = false</strong>时，不进行时间戳属性的过滤。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p33413632111"><a name="p33413632111"></a><a name="p33413632111"></a><strong id="b170516359452"><a name="b170516359452"></a><a name="b170516359452"></a>int64_t *labels</strong>：TopK特征的Label。</p>
<p id="p1861116507441"><a name="p1861116507441"></a><a name="p1861116507441"></a><strong id="b1356612271453"><a name="b1356612271453"></a><a name="b1356612271453"></a>float *distances</strong>：TopK特征的距离。</p>
<p id="p1767014017194"><a name="p1767014017194"></a><a name="p1767014017194"></a><strong id="b2308133192313"><a name="b2308133192313"></a><a name="b2308133192313"></a>uint32_t *validNums：</strong>每个query向量经过比对后得到的有效结果个数。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname14574204217120"><a name="parmname14574204217120"></a><a name="parmname14574204217120"></a>“count”</span>取值在[1, 10240]区间。</li><li><span class="parmname" id="parmname58974551313"><a name="parmname58974551313"></a><a name="parmname58974551313"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname9151511121"><a name="parmname9151511121"></a><a name="parmname9151511121"></a>“attrFilter”</span><strong id="b124012563238"><a name="b124012563238"></a><a name="b124012563238"></a>：</strong>当<strong id="b1724010565238"><a name="b1724010565238"></a><a name="b1724010565238"></a>shareAttrFilter</strong>为true时，长度为1；当<strong id="b1733219014246"><a name="b1733219014246"></a><a name="b1733219014246"></a>shareAttrFilter</strong>为false时，长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname015216455115"><a name="parmname015216455115"></a><a name="parmname015216455115"></a>“topk”</span>取值在[1, 100000]区间。</li><li><span class="parmname" id="parmname174713312215"><a name="parmname174713312215"></a><a name="parmname174713312215"></a>“labels”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname5667121332617"><a name="parmname5667121332617"></a><a name="parmname5667121332617"></a>“distances”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1315615911316"><a name="parmname1315615911316"></a><a name="parmname1315615911316"></a>“validNums”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### SearchWithExtraMask接口<a name="ZH-CN_TOPIC_0000001494506850"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p3463193419203"><a name="p3463193419203"></a><a name="p3463193419203"></a>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>计算输入特征和经过AttrFilter和外部Mask过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p2360715171619"><a name="p2360715171619"></a><a name="p2360715171619"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a>uint32_t count</strong>：待比较的特征数量。</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a>const void *features</strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat为float类型。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b161793465431"><a name="b161793465431"></a><a name="b161793465431"></a>const AttrFilter *attrFilter</strong>：属性过滤信息，具体请参见<a href="#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。</p>
<p id="p638962561711"><a name="p638962561711"></a><a name="p638962561711"></a><strong id="b72082417127"><a name="b72082417127"></a><a name="b72082417127"></a>bool shareAttrFilter</strong>：同一个query是否共享一个Mask。</p>
<p id="p263216178448"><a name="p263216178448"></a><a name="p263216178448"></a><strong id="b9237122811443"><a name="b9237122811443"></a><a name="b9237122811443"></a>uint32_t topk</strong>：计算余弦距离后需要保存的TopK大小。</p>
<p id="p1792132311"><a name="p1792132311"></a><a name="p1792132311"></a><strong id="b10193171182320"><a name="b10193171182320"></a><a name="b10193171182320"></a>const uint8_t *extraMask</strong>：外部输入的额外的过滤Mask，以bit为单位，0和1分别代表过滤或者选中该条特征。</p>
<p id="p1426510564258"><a name="p1426510564258"></a><a name="p1426510564258"></a><strong id="b11418152210274"><a name="b11418152210274"></a><a name="b11418152210274"></a>uint64_t extraMaskLenEachQuery</strong>：外部输入Mask的长度，单位为字节。</p>
<p id="p1165364810261"><a name="p1165364810261"></a><a name="p1165364810261"></a><strong id="b171528264278"><a name="b171528264278"></a><a name="b171528264278"></a>bool extraMaskIsAtDevice</strong>：用户外部输入的Mask是否已存在Device侧。</p>
<p id="p198449441221"><a name="p198449441221"></a><a name="p198449441221"></a><strong id="b194349562317"><a name="b194349562317"></a><a name="b194349562317"></a>bool enableTimeFilter</strong>：时间戳属性过滤开关，默认为<span class="parmvalue" id="parmvalue6379218248"><a name="parmvalue6379218248"></a><a name="parmvalue6379218248"></a>“true”</span>，当<strong id="b12428111462413"><a name="b12428111462413"></a><a name="b12428111462413"></a>enableTimeFilter = false</strong>时，不进行时间戳属性的过滤。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p695214202452"><a name="p695214202452"></a><a name="p695214202452"></a><strong id="b170516359452"><a name="b170516359452"></a><a name="b170516359452"></a>int64_t *labels</strong>：TopK特征的Label。</p>
<p id="p1861116507441"><a name="p1861116507441"></a><a name="p1861116507441"></a><strong id="b1356612271453"><a name="b1356612271453"></a><a name="b1356612271453"></a>float *distances</strong>：TopK特征的距离。</p>
<p id="p33413632111"><a name="p33413632111"></a><a name="p33413632111"></a><strong id="b2308133192313"><a name="b2308133192313"></a><a name="b2308133192313"></a>uint32_t *validNums：</strong>每个query向量经过比对后得到的有效结果个数。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><span class="parmname" id="parmname14574204217120"><a name="parmname14574204217120"></a><a name="parmname14574204217120"></a>“count”</span>取值在[1, 10240]区间。</li><li><span class="parmname" id="parmname015216455115"><a name="parmname015216455115"></a><a name="parmname015216455115"></a>“topk”</span>取值在[1, 100000]区间。</li><li><span class="parmname" id="parmname58974551313"><a name="parmname58974551313"></a><a name="parmname58974551313"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname9151511121"><a name="parmname9151511121"></a><a name="parmname9151511121"></a>“attrFilter”</span><strong id="b124012563238"><a name="b124012563238"></a><a name="b124012563238"></a>：</strong>当<strong id="b1724010565238"><a name="b1724010565238"></a><a name="b1724010565238"></a>shareAttrFilter</strong>为true时，长度为1；当<strong id="b1733219014246"><a name="b1733219014246"></a><a name="b1733219014246"></a>shareAttrFilter</strong>为false时，长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname5667121332617"><a name="parmname5667121332617"></a><a name="parmname5667121332617"></a>“distances”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1315615911316"><a name="parmname1315615911316"></a><a name="parmname1315615911316"></a>“validNums”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname174713312215"><a name="parmname174713312215"></a><a name="parmname174713312215"></a>“labels”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname9101105543017"><a name="parmname9101105543017"></a><a name="parmname9101105543017"></a>“extraMask”</span>：当<strong id="b7678141311311"><a name="b7678141311311"></a><a name="b7678141311311"></a>shareAttrFilter</strong>为true时，长度为<span class="parmvalue" id="parmvalue156636433214"><a name="parmvalue156636433214"></a><a name="parmvalue156636433214"></a>“extraMaskLenEachQuery”</span>；当<strong id="b76786138311"><a name="b76786138311"></a><a name="b76786138311"></a>shareAttrFilter</strong>为false时，长度为count * extraMaskLenEachQuery，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

#### SearchWithExtraMask带额外相似度接口<a name="ZH-CN_TOPIC_0000002373091106"></a>

<a name="table197013362381"></a>
<table><tbody><tr id="row597023693810"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p199703365385"><a name="p199703365385"></a><a name="p199703365385"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p99701836173815"><a name="p99701836173815"></a><a name="p99701836173815"></a>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, const uint16_t *extraScore, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</p>
</td>
</tr>
<tr id="row109701336173810"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p17970153683816"><a name="p17970153683816"></a><a name="p17970153683816"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p169706365385"><a name="p169706365385"></a><a name="p169706365385"></a>计算输入特征和经过AttrFilter和外部Mask过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</p>
</td>
</tr>
<tr id="row19702366386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p18970113673814"><a name="p18970113673814"></a><a name="p18970113673814"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p119701036143819"><a name="p119701036143819"></a><a name="p119701036143819"></a><strong id="b1897053618384"><a name="b1897053618384"></a><a name="b1897053618384"></a>uint32_t count</strong>：待比较的特征数量。</p>
<p id="p1697015367382"><a name="p1697015367382"></a><a name="p1697015367382"></a><strong id="b49704363384"><a name="b49704363384"></a><a name="b49704363384"></a>const void *features</strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8Flat为int8_t类型，FP16Flat为float类型。</p>
<p id="p12970173633811"><a name="p12970173633811"></a><a name="p12970173633811"></a><strong id="b2097015361386"><a name="b2097015361386"></a><a name="b2097015361386"></a>const AttrFilter *attrFilter</strong>：属性过滤信息，具体请参见<a href="#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。</p>
<p id="p18970193617382"><a name="p18970193617382"></a><a name="p18970193617382"></a><strong id="b16970133693819"><a name="b16970133693819"></a><a name="b16970133693819"></a>bool shareAttrFilter</strong>：同一个query是否共享一个Mask。</p>
<p id="p497013613389"><a name="p497013613389"></a><a name="p497013613389"></a><strong id="b3970123613810"><a name="b3970123613810"></a><a name="b3970123613810"></a>uint32_t topk</strong>：计算余弦距离后需要保存的TopK大小。</p>
<p id="p119709368387"><a name="p119709368387"></a><a name="p119709368387"></a><strong id="b109708366381"><a name="b109708366381"></a><a name="b109708366381"></a>const uint8_t *extraMask</strong>：外部输入的额外的过滤Mask，以bit为单位，0和1分别代表过滤或者选中该条特征。</p>
<p id="p5970123623814"><a name="p5970123623814"></a><a name="p5970123623814"></a><strong id="b2970103610387"><a name="b2970103610387"></a><a name="b2970103610387"></a>uint64_t extraMaskLenEachQuery</strong>：外部输入Mask的长度，单位为字节。</p>
<p id="p1970103663815"><a name="p1970103663815"></a><a name="p1970103663815"></a><strong id="b19701936173815"><a name="b19701936173815"></a><a name="b19701936173815"></a>bool extraMaskIsAtDevice</strong>：用户外部输入的Mask是否已存在Device侧。</p>
<p id="p472715355451"><a name="p472715355451"></a><a name="p472715355451"></a><strong id="b367692814812"><a name="b367692814812"></a><a name="b367692814812"></a>const uint16_t *extraScore：</strong>用户输入的额外相似度，长度为count*totalPad（totalPad为底库长度按照16对齐的大小）。</p>
<p id="p1397083623812"><a name="p1397083623812"></a><a name="p1397083623812"></a><strong id="b69700361388"><a name="b69700361388"></a><a name="b69700361388"></a>bool enableTimeFilter</strong>：时间戳属性过滤开关，默认为<span class="parmvalue" id="parmvalue1697015364388"><a name="parmvalue1697015364388"></a><a name="parmvalue1697015364388"></a>“true”</span>，当<strong id="b11970153611385"><a name="b11970153611385"></a><a name="b11970153611385"></a>enableTimeFilter = false</strong>时，不进行时间戳属性的过滤。</p>
</td>
</tr>
<tr id="row199701366383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p09701636163810"><a name="p09701636163810"></a><a name="p09701636163810"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p897011361383"><a name="p897011361383"></a><a name="p897011361383"></a><strong id="b15970336173812"><a name="b15970336173812"></a><a name="b15970336173812"></a>int64_t *labels</strong>：TopK特征的Label。若底库使用AddFeatureByIndice添加，则此处输出底库位置（indices）。</p>
<p id="p1497083653817"><a name="p1497083653817"></a><a name="p1497083653817"></a><strong id="b6970436153820"><a name="b6970436153820"></a><a name="b6970436153820"></a>float *distances</strong>：TopK特征的距离。</p>
<p id="p19701536113820"><a name="p19701536113820"></a><a name="p19701536113820"></a><strong id="b159701536183810"><a name="b159701536183810"></a><a name="b159701536183810"></a>uint32_t *validNums：</strong>每个query向量经过比对后得到的有效结果个数。</p>
</td>
</tr>
<tr id="row3970143643812"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p0970936143817"><a name="p0970936143817"></a><a name="p0970936143817"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p13970536113820"><a name="p13970536113820"></a><a name="p13970536113820"></a><strong id="b15970133653813"><a name="b15970133653813"></a><a name="b15970133653813"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row097033617385"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p20970153617385"><a name="p20970153617385"></a><a name="p20970153617385"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul16970136203817"></a><a name="ul16970136203817"></a><ul id="ul16970136203817"><li><span class="parmname" id="parmname897013653817"><a name="parmname897013653817"></a><a name="parmname897013653817"></a>“count”</span>取值在[1, 10240]区间。</li><li><span class="parmname" id="parmname5970153633819"><a name="parmname5970153633819"></a><a name="parmname5970153633819"></a>“topk”</span>取值在[1, 100000]区间。</li><li><span class="parmname" id="parmname17970153612387"><a name="parmname17970153612387"></a><a name="parmname17970153612387"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname497018361385"><a name="parmname497018361385"></a><a name="parmname497018361385"></a>“attrFilter”</span><strong id="b597033683812"><a name="b597033683812"></a><a name="b597033683812"></a>：</strong>当<strong id="b997013611387"><a name="b997013611387"></a><a name="b997013611387"></a>shareAttrFilter</strong>为true时，长度为1；当<strong id="b1297023623815"><a name="b1297023623815"></a><a name="b1297023623815"></a>shareAttrFilter</strong>为false时，长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1297043613384"><a name="parmname1297043613384"></a><a name="parmname1297043613384"></a>“distances”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname297053673812"><a name="parmname297053673812"></a><a name="parmname297053673812"></a>“validNums”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1297083603814"><a name="parmname1297083603814"></a><a name="parmname1297083603814"></a>“labels”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1971123610383"><a name="parmname1971123610383"></a><a name="parmname1971123610383"></a>“extraMask”</span>：当<strong id="b7971936163814"><a name="b7971936163814"></a><a name="b7971936163814"></a>shareAttrFilter</strong>为true时，长度为<span class="parmvalue" id="parmvalue11971173653810"><a name="parmvalue11971173653810"></a><a name="parmvalue11971173653810"></a>“extraMaskLenEachQuery”</span>；当<strong id="b109711367381"><a name="b109711367381"></a><a name="b109711367381"></a>shareAttrFilter</strong>为false时，长度为count * extraMaskLenEachQuery，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname16950829165517"><a name="parmname16950829165517"></a><a name="parmname16950829165517"></a>“extraScore”</span>：长度为count*totalPad（totalPad为底库长度按照16对齐的大小），否则可能出现越界读写错误并引起程序崩溃。实际对应float16_t类型，值的范围在-1.0到1.0之间。当前仅对Int8FlatCos和FlatIP非共享Mask有效，否则<span class="parmname" id="parmname196111118596"><a name="parmname196111118596"></a><a name="parmname196111118596"></a>“extraScore”</span>不参与计算。</li></ul>
</td>
</tr>
</tbody>
</table>

#### SearchWithExtraVal接口<a name="ZH-CN_TOPIC_0000002013215285"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1394085711557"><a name="p1394085711557"></a><a name="p1394085711557"></a>APP_ERROR SearchWithExtraVal(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, const ExtraValFilter *extraValFilter, bool enableTimeFilter = true);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p0795142313481"><a name="p0795142313481"></a><a name="p0795142313481"></a>计算输入特征和经过AttrFilter和ExtraValFilter过滤后的底库向量的距离并将距离进行TopK排序，返回对应的距离和下标。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p2360715171619"><a name="p2360715171619"></a><a name="p2360715171619"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a>uint32_t count</strong>：待比较的特征数量。</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a>const void *features</strong>：待比较的特征，汉明距离为uint8_t类型的数据，Int8cos为int8_t类型。当前仅支持int8cos（包括异构内存场景）和汉明距离。</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b161793465431"><a name="b161793465431"></a><a name="b161793465431"></a>const AttrFilter *attrFilter</strong>：属性过滤信息，具体请参见<a href="#ZH-CN_TOPIC_0000001458687398">AttrFilter</a>。</p>
<p id="p638962561711"><a name="p638962561711"></a><a name="p638962561711"></a><strong id="b72082417127"><a name="b72082417127"></a><a name="b72082417127"></a>bool shareAttrFilter</strong>：附加属性暂仅支持“false”，不同query非共享一个mask。</p>
<p id="p263216178448"><a name="p263216178448"></a><a name="p263216178448"></a><strong id="b9237122811443"><a name="b9237122811443"></a><a name="b9237122811443"></a>uint32_t topk</strong>：计算余弦距离后需要保存的TopK大小。</p>
<p id="p576645315463"><a name="p576645315463"></a><a name="p576645315463"></a><strong id="b1599945784610"><a name="b1599945784610"></a><a name="b1599945784610"></a>const ExtraValFilter *extraValFilter</strong>：附加属性过滤信息，具体请见<a href="#ZH-CN_TOPIC_0000002013200765">ExtraValFilter</a>。</p>
<p id="p198449441221"><a name="p198449441221"></a><a name="p198449441221"></a><strong id="b194349562317"><a name="b194349562317"></a><a name="b194349562317"></a>bool enableTimeFilter</strong>：时间戳属性过滤开关，默认为<span class="parmvalue" id="parmvalue6379218248"><a name="parmvalue6379218248"></a><a name="parmvalue6379218248"></a>“true”</span>，当<strong id="b12428111462413"><a name="b12428111462413"></a><a name="b12428111462413"></a>enableTimeFilter = false</strong>时，不进行时间戳属性的过滤。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p33413632111"><a name="p33413632111"></a><a name="p33413632111"></a><strong id="b2308133192313"><a name="b2308133192313"></a><a name="b2308133192313"></a>uint32_t *validNums：</strong>每个query向量经过比对后得到的有效结果个数。</p>
<p id="p695214202452"><a name="p695214202452"></a><a name="p695214202452"></a><strong id="b170516359452"><a name="b170516359452"></a><a name="b170516359452"></a>int64_t *labels</strong>：TopK特征的Label。</p>
<p id="p1861116507441"><a name="p1861116507441"></a><a name="p1861116507441"></a><strong id="b1356612271453"><a name="b1356612271453"></a><a name="b1356612271453"></a>float *distances</strong>：TopK特征的距离。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul13705385331"></a><a name="ul13705385331"></a><ul id="ul13705385331"><li><span class="parmname" id="parmname1437003823318"><a name="parmname1437003823318"></a><a name="parmname1437003823318"></a>“count”</span>取值在[1, 10240]区间。</li><li><span class="parmname" id="parmname4370133863320"><a name="parmname4370133863320"></a><a name="parmname4370133863320"></a>“features”</span>长度为count * 向量维度dim，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname637014382333"><a name="parmname637014382333"></a><a name="parmname637014382333"></a>“attrFilter”</span><strong id="b124012563238"><a name="b124012563238"></a><a name="b124012563238"></a>：</strong>当<strong id="b1724010565238"><a name="b1724010565238"></a><a name="b1724010565238"></a>shareAttrFilter</strong>为true时，长度为1；当<strong id="b1733219014246"><a name="b1733219014246"></a><a name="b1733219014246"></a>shareAttrFilter</strong>为false时，长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname2370173873315"><a name="parmname2370173873315"></a><a name="parmname2370173873315"></a>“topk”</span>取值在[1, 100000]区间。</li><li><span class="parmname" id="parmname183719384335"><a name="parmname183719384335"></a><a name="parmname183719384335"></a>“labels”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname5667121332617"><a name="parmname5667121332617"></a><a name="parmname5667121332617"></a>“distances”</span>长度为count * topk，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname2037173811331"><a name="parmname2037173811331"></a><a name="parmname2037173811331"></a>“validNums”</span>长度为count，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname1315315252354"><a name="parmname1315315252354"></a><a name="parmname1315315252354"></a>“extraValFilter”</span>取值为空指针或者长度为count，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE] 说明
>SearchWithExtraVal不能与Search接口混用。

#### SetHeteroParam接口<a name="ZH-CN_TOPIC_0000001630850578"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.1.1 "><p id="p312319560281"><a name="p312319560281"></a><a name="p312319560281"></a>APP_ERROR SetHeteroParam(size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>设置异构存储策略参数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.3.1 "><p id="p2908312183018"><a name="p2908312183018"></a><a name="p2908312183018"></a><strong id="b156074619418"><a name="b156074619418"></a><a name="b156074619418"></a>size_t deviceCapacity</strong>：异构内存策略下，Device侧存储底库容量（字节）。</p>
<p id="p1534173611301"><a name="p1534173611301"></a><a name="p1534173611301"></a><strong id="b81425114419"><a name="b81425114419"></a><a name="b81425114419"></a>size_t deviceBuffer</strong>：异构内存策略下，Device侧缓存容量（字节）。</p>
<p id="p194803440119"><a name="p194803440119"></a><a name="p194803440119"></a><strong id="b208671753121117"><a name="b208671753121117"></a><a name="b208671753121117"></a>size_t hostCapacity</strong>：异构内存策略下，Host侧存储底库容量（字节）。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.4.1 "><p id="p1658819198309"><a name="p1658819198309"></a><a name="p1658819198309"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.6.1 "><a name="ul19200250124118"></a><a name="ul19200250124118"></a><ul id="ul19200250124118"><li>该接口需在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>接口设置内存策略为<span class="parmvalue" id="parmvalue1252823154217"><a name="parmvalue1252823154217"></a><a name="parmvalue1252823154217"></a>“MemoryStrategy::HETERO_MEMORY”</span>（异构内存策略）后使用。</li><li><span class="parmname" id="parmname198386215419"><a name="parmname198386215419"></a><a name="parmname198386215419"></a>“deviceCapacity”</span>最小值为1G，最大值为Device实际剩余内存大小。</li><li><span class="parmname" id="parmname9129152594116"><a name="parmname9129152594116"></a><a name="parmname9129152594116"></a>“deviceBuffer”</span>最小值为2 * 262144 * dim，最大值为<span class="parmvalue" id="parmvalue578544611356"><a name="parmvalue578544611356"></a><a name="parmvalue578544611356"></a>“8G”</span>。请根据Device侧实际剩余内存大小进行设置。</li><li><strong id="b16232174415416"><a name="b16232174415416"></a><a name="b16232174415416"></a>deviceCapacity + deviceBuffer</strong>应小于Device实际剩余内存大小。</li><li><span class="parmname" id="parmname7229121816109"><a name="parmname7229121816109"></a><a name="parmname7229121816109"></a>“hostCapacity”</span>取值范围：[1G, 512G]，请根据Host侧实际内存可申请的大小进行配置。</li></ul>
</td>
</tr>
</tbody>
</table>

#### SetSaveHostMemory接口<a name="ZH-CN_TOPIC_0000002106649489"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.1.1 "><p id="p45681112145513"><a name="p45681112145513"></a><a name="p45681112145513"></a>APP_ERROR SetSaveHostMemory();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.2.1 "><p id="p1984752010553"><a name="p1984752010553"></a><a name="p1984752010553"></a>设置使用节约host内存模式，默认不使用。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.3.1 "><p id="p2908312183018"><a name="p2908312183018"></a><a name="p2908312183018"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.4.1 "><p id="p1658819198309"><a name="p1658819198309"></a><a name="p1658819198309"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.6.1 "><a name="ul13741324723"></a><a name="ul13741324723"></a><ul id="ul13741324723"><li>该接口需要在<a href="#ZH-CN_TOPIC_0000001458680014">Init</a>接口之后，底库为0时使用。</li><li>该接口可以节约host内存，但是会降低删除类型和获取类型接口的性能。</li><li>使用该模式时，无法使用<a href="#ZH-CN_TOPIC_0000001458680018">DeleteFeatureByToken</a>接口。</li><li>该接口只支持汉明距离。</li></ul>
</td>
</tr>
</tbody>
</table>

### AttrFilter<a id="ZH-CN_TOPIC_0000001458687398"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001507967265"></a>

特征属性过滤器，该结构体需要结合AscendIndexTS实例来使用，在特征检索时作为输入参数。

调用检索接口的所有query向量共享同一个过滤器，该过滤器会和底库中的每一个底库特征对应的属性进行匹配，可以比较的信息例如：时间、token ID。

匹配成功的底库特征会参与接下来的检索流程，即向量距离比对与TopK排序等。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

#### timesEnd接口<a name="ZH-CN_TOPIC_0000001458367566"></a>

int32\_t：过滤时间段的结束时间。

#### timesStart接口<a name="ZH-CN_TOPIC_0000001507647493"></a>

int32\_t：过滤时间段的开始时间。

#### tokenBitSet接口<a name="ZH-CN_TOPIC_0000001507887177"></a>

uint8\_t\*：特征token ID的列表，每个uint8\_t成员从低位到高位，按位记录token信息，1代表选中，0代表token未选中。

例如：一个过滤器的token列表包含两个非零的uint8\_t成员：\[7, 15, 0, 0, ……, 0\]，非零成员的二进制表示为00000111、00001111，则它们表达的有效token ID为：0，1，2，8，9，10，11。

> [!NOTE] 说明 
>“tokenBitSet”长度应为“tokenBitSetLen”，否则可能出现越界读写错误并引起程序崩溃。


#### tokenBitSetLen接口<a name="ZH-CN_TOPIC_0000001458687402"></a>

uint32\_t：指定过滤器AttrFilter中tokenBitSet字段的长度。

### ExtraValAttr<a id="ZH-CN_TOPIC_0000002013198657"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000002013039153"></a>

附加属性信息，入库时和特征向量一起添加。该结构体需要结合AscendIndexTS实例来使用。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

#### val接口<a name="ZH-CN_TOPIC_0000001976479160"></a>

int16\_t：记录当前特征的附加属性信息，用二进制表示，“1”代表“是”，“0”代表“否”。



### ExtraValFilter<a id="ZH-CN_TOPIC_0000002013200765"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001976640904"></a>

附加属性过滤器，该结构体需要结合AscendIndexTS实例来使用，在特征检索时作为输入参数。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

#### filterVal接口<a name="ZH-CN_TOPIC_0000001976481180"></a>

int16\_t：待查询的附加属性，用二进制表示，“1”表示保留附加属性，“0”表示过滤附加属性。


#### matchVal接口<a name="ZH-CN_TOPIC_0000002013041289"></a>

int16\_t：附加属性查询模式，分为模式0和模式1。

- 对于模式0，匹配条件为：**ExtraValAttr::val & ExtraValFilter::filterVal == ExtraValFilter::filterVal**
- 对于模式1，匹配条件为：**ExtraValAttr::val & ExtraValFilter::filterVal  \>  0**

### FeatureAttr<a id="ZH-CN_TOPIC_0000001507967381"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001458367674"></a>

特征属性信息，入库时和特征向量一起添加。该结构体需要结合AscendIndexTS实例来使用。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

#### time接口<a name="ZH-CN_TOPIC_0000001507647601"></a>

int32\_t：记录当前特征的时间信息，以时间戳（秒级）形式表示。

> [!NOTE] 说明 
>由于昇腾硬件限制，只能处理int32类型数据，因此用户需要保证当前时间戳不会超过int32的最大值，建议在实际操作时，将当前实际时间戳减去固定的一个历史时间戳，然后再存入。

#### tokenId接口<a name="ZH-CN_TOPIC_0000001507887269"></a>

uint32\_t：特征token ID，一个token ID对应多个特征，一个特征对应一个token ID，需要小于用户初始化AscendIndexTS时传入的tokenNum的值。
