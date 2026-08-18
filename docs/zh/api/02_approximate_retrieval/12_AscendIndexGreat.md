# AscendIndexGreat<a name="ZH-CN_TOPIC_0000002044829945"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002008751966"></a>

自研向量检索算法，为用户提供昇腾侧和鲲鹏侧高维大底库近似检索能力。使用自研检索策略在底库中检索得到topK个最近似向量结果。

存入底库的向量以及各个接口的query向量均需为归一化的float浮点数类型。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

此算法主要针对大底库场景的近似模糊搜索，相较暴力检索精度已有一定损失。在小底库场景，建议适当加大超参值，可改善精度损失问题。

> [!NOTE]
>
>- 创建Index实例时传入的参数params，需根据实际情况设置其中的dim。
>- Index分为两种算法模式：KMode仅使用鲲鹏侧算法，AKMode昇腾加鲲鹏算法，在AKMode模式下需要提前生成对应算子。
>- subSpaceDimnlist应与码本训练时对应参数保持一致。

## AscendIndexGreat接口<a name="ZH-CN_TOPIC_0000002044829953"></a>

<a name="table5404639201712"></a>
<table><tbody><tr id="row194338394172"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p843363911171"><a name="p843363911171"></a><a name="p843363911171"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p144331039111710"><a name="p144331039111710"></a><a name="p144331039111710"></a>AscendIndexGreat(const std::string&amp; mode, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</p>
</td>
</tr>
<tr id="row043313981717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p24337391179"><a name="p24337391179"></a><a name="p24337391179"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6433139151710"><a name="p6433139151710"></a><a name="p6433139151710"></a>AscendIndexGreat的构造函数，创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row124339399175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p164331391173"><a name="p164331391173"></a><a name="p164331391173"></a><strong id="b5433193991713"><a name="b5433193991713"></a><a name="b5433193991713"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12433193918175"><a name="p12433193918175"></a><a name="p12433193918175"></a><strong id="b16722123816518"><a name="b16722123816518"></a><a name="b16722123816518"></a>const std::string&amp; mode</strong>：指定算法模式。</p>
<p id="p104331039161715"><a name="p104331039161715"></a><a name="p104331039161715"></a><strong id="b769912415512"><a name="b769912415512"></a><a name="b769912415512"></a>const std::vector&lt;int&gt;&amp; deviceList</strong>：指定的NPU侧设备ID。</p>
<p id="p204336391174"><a name="p204336391174"></a><a name="p204336391174"></a><strong id="b10327344205112"><a name="b10327344205112"></a><a name="b10327344205112"></a>bool verbose</strong>：指定是否开启verbose选项，开启后部分操作提供额外的打印提示。默认值为<span class="parmvalue" id="parmvalue810213723716"><a name="parmvalue810213723716"></a><a name="parmvalue810213723716"></a>“false”</span>。</p>
</td>
</tr>
<tr id="row443343901718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p8433133912177"><a name="p8433133912177"></a><a name="p8433133912177"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p204331639171718"><a name="p204331639171718"></a><a name="p204331639171718"></a>无</p>
</td>
</tr>
<tr id="row1443373991714"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p743343921714"><a name="p743343921714"></a><a name="p743343921714"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p174331439151714"><a name="p174331439151714"></a><a name="p174331439151714"></a>无</p>
</td>
</tr>
<tr id="row04331639141719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p114331039171716"><a name="p114331039171716"></a><a name="p114331039171716"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1462219489335"></a><a name="ul1462219489335"></a><ul id="ul1462219489335"><li>mode：只支持“KMode”和“AKMode”两种模式。</li><li>deviceList：请使用<strong id="b3555159175110"><a name="b3555159175110"></a><a name="b3555159175110"></a>npu-smi</strong>命令查询对应的NPUID，仅支持一个device设备ID。</li><li>使用此构造函数创建Index实例后，需要先调用“LoadIndex”加载事先落盘后的Index实例，然后再进行其他操作。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table72261454131719"></a>
<table><tbody><tr id="row18251175431713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16251154121710"><a name="p16251154121710"></a><a name="p16251154121710"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p76841847185413"><a name="p76841847185413"></a><a name="p76841847185413"></a>explicit AscendIndexGreat(const AscendIndexGreatInitParams&amp; kModeInitParams);</p>
</td>
</tr>
<tr id="row11251125451717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p172516541177"><a name="p172516541177"></a><a name="p172516541177"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p11251135461713"><a name="p11251135461713"></a><a name="p11251135461713"></a>AscendIndexGreat的构造函数，创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row82514548179"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p325155441716"><a name="p325155441716"></a><a name="p325155441716"></a><strong id="b2251254161713"><a name="b2251254161713"></a><a name="b2251254161713"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p82516541176"><a name="p82516541176"></a><a name="p82516541176"></a>Index所需的初始化参数kModeInitParams，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>。</p>
</td>
</tr>
<tr id="row3251354151719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2252115451717"><a name="p2252115451717"></a><a name="p2252115451717"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1125255410178"><a name="p1125255410178"></a><a name="p1125255410178"></a>无</p>
</td>
</tr>
<tr id="row1725275471717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2252054121720"><a name="p2252054121720"></a><a name="p2252054121720"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row13252145413171"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p125285413172"><a name="p125285413172"></a><a name="p125285413172"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20252954121720"><a name="p20252954121720"></a><a name="p20252954121720"></a>参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>中的参数说明和参数约束。</p>
</td>
</tr>
</tbody>
</table>

<a name="table198261931819"></a>
<table><tbody><tr id="row78491591183"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1484915918184"><a name="p1484915918184"></a><a name="p1484915918184"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1584913901820"><a name="p1584913901820"></a><a name="p1584913901820"></a>AscendIndexGreat(const AscendIndexVstarInitParams&amp; aModeInitParams, const AscendIndexGreatInitParams&amp; kModeInitParams);</p>
</td>
</tr>
<tr id="row284999121814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p78498919183"><a name="p78498919183"></a><a name="p78498919183"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13849791188"><a name="p13849791188"></a><a name="p13849791188"></a>AscendIndexGreat的构造函数，创建Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row784912981810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1184919911189"><a name="p1184919911189"></a><a name="p1184919911189"></a><strong id="b9849692189"><a name="b9849692189"></a><a name="b9849692189"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1984999111812"><a name="p1984999111812"></a><a name="p1984999111812"></a>Index所需的初始化参数aModeInitParams和kModeInitParams，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>和<a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams接口">AscendIndexGreatInitParams</a>。</p>
</td>
</tr>
<tr id="row5850179121814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1285009121812"><a name="p1285009121812"></a><a name="p1285009121812"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14850159181811"><a name="p14850159181811"></a><a name="p14850159181811"></a>无</p>
</td>
</tr>
<tr id="row13850199151817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1185019912183"><a name="p1185019912183"></a><a name="p1185019912183"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p123604384355"><a name="p123604384355"></a><a name="p123604384355"></a><strong id="b536183815357"><a name="b536183815357"></a><a name="b536183815357"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row16850109161814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p985029101810"><a name="p985029101810"></a><a name="p985029101810"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p12850169151811"><a name="p12850169151811"></a><a name="p12850169151811"></a>参考<a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams接口">AscendIndexVstarInitParams</a>和<a href="./05_AscendIndexIVFSP.md#ascendindexcodebookinitparams接口">AscendIndexGreatInitParams</a>中的参数说明和参数约束。</p>
<p id="p122218114501"><a name="p122218114501"></a><a name="p122218114501"></a>aModeInitParams和kModeInitParams的dim必须保持一致。</p>
</td>
</tr>
</tbody>
</table>

<a name="table32891532172215"></a>
<table><tbody><tr id="row1731883213226"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p183181632182220"><a name="p183181632182220"></a><a name="p183181632182220"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3318832182215"><a name="p3318832182215"></a><a name="p3318832182215"></a>AscendIndexGreat(const AscendIndexGreat&amp;) = delete;</p>
</td>
</tr>
<tr id="row831813213224"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p231819329229"><a name="p231819329229"></a><a name="p231819329229"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6318232102217"><a name="p6318232102217"></a><a name="p6318232102217"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row731817327226"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p631813272217"><a name="p631813272217"></a><a name="p631813272217"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p153181732102215"><a name="p153181732102215"></a><a name="p153181732102215"></a><strong id="b113181932132210"><a name="b113181932132210"></a><a name="b113181932132210"></a>const AscendIndexGreat&amp;</strong>：常量AscendIndexGreat对象。</p>
</td>
</tr>
<tr id="row20318632102217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p431833212210"><a name="p431833212210"></a><a name="p431833212210"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15318173218228"><a name="p15318173218228"></a><a name="p15318173218228"></a>无</p>
</td>
</tr>
<tr id="row8318153215220"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1831811326227"><a name="p1831811326227"></a><a name="p1831811326227"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p83180322222"><a name="p83180322222"></a><a name="p83180322222"></a>无</p>
</td>
</tr>
<tr id="row131933214228"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p103190320223"><a name="p103190320223"></a><a name="p103190320223"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p7319133272214"><a name="p7319133272214"></a><a name="p7319133272214"></a>无</p>
</td>
</tr>
</tbody>
</table>

## \~AscendIndexGreat接口<a name="ZH-CN_TOPIC_0000002013257524"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p128341740125117"><a name="p128341740125117"></a><a name="p128341740125117"></a>virtual ~AscendIndexGreat() = default;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendIndexGreat的析构函数，销毁AscendIndexGreat对象，释放资源。</p>
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

## operator =接口<a name="ZH-CN_TOPIC_0000002008751990"></a>

<a name="table39961720122213"></a>
<table><tbody><tr id="row3176213227"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p71752119228"><a name="p71752119228"></a><a name="p71752119228"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1917321192217"><a name="p1917321192217"></a><a name="p1917321192217"></a>AscendIndexGreat &amp;operator=(const AscendIndexGreat&amp;) = delete;</p>
</td>
</tr>
<tr id="row111762152213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p71719215225"><a name="p71719215225"></a><a name="p71719215225"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p817921142214"><a name="p817921142214"></a><a name="p817921142214"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row1217121122217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p41710213220"><a name="p41710213220"></a><a name="p41710213220"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p111732182220"><a name="p111732182220"></a><a name="p111732182220"></a><strong id="b101742192218"><a name="b101742192218"></a><a name="b101742192218"></a>const AscendIndexGreat&amp;</strong>：常量AscendIndexGreat对象。</p>
</td>
</tr>
<tr id="row3171321172218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1717521132215"><a name="p1717521132215"></a><a name="p1717521132215"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p191714215226"><a name="p191714215226"></a><a name="p191714215226"></a>无</p>
</td>
</tr>
<tr id="row181713210221"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p917182117222"><a name="p917182117222"></a><a name="p917182117222"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p917921142212"><a name="p917921142212"></a><a name="p917921142212"></a>无</p>
</td>
</tr>
<tr id="row117172118229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p618162116224"><a name="p618162116224"></a><a name="p618162116224"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1318182111227"><a name="p1318182111227"></a><a name="p1318182111227"></a>无</p>
</td>
</tr>
</tbody>
</table>

## Add接口<a name="ZH-CN_TOPIC_0000002044950953"></a>

<a name="table11133547191811"></a>
<table><tbody><tr id="row1159447111810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12159184761818"><a name="p12159184761818"></a><a name="p12159184761818"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161591747181812"><a name="p161591747181812"></a><a name="p161591747181812"></a>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseRawData);</p>
</td>
</tr>
<tr id="row10159194716180"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p5159847181814"><a name="p5159847181814"></a><a name="p5159847181814"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015916478183"><a name="p2015916478183"></a><a name="p2015916478183"></a>向AscendIndexGreat底库中添加新的特征向量。</p>
</td>
</tr>
<tr id="row11159847131815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p115914712182"><a name="p115914712182"></a><a name="p115914712182"></a><strong id="b615918476186"><a name="b615918476186"></a><a name="b615918476186"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1215974741811"><a name="p1215974741811"></a><a name="p1215974741811"></a><strong id="b14159184715188"><a name="b14159184715188"></a><a name="b14159184715188"></a>const std::vector&lt;float&gt;&amp; baseRawData：</strong>添加进底库的特征向量。</p>
</td>
</tr>
<tr id="row11159647171819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18159184718181"><a name="p18159184718181"></a><a name="p18159184718181"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2159147101812"><a name="p2159147101812"></a><a name="p2159147101812"></a>无</p>
</td>
</tr>
<tr id="row615904711813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1815954712186"><a name="p1815954712186"></a><a name="p1815954712186"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row2160144719188"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19160947121812"><a name="p19160947121812"></a><a name="p19160947121812"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul19630537566"></a><a name="ul19630537566"></a><ul id="ul19630537566"><li>此处数组“baseRawData”的长度应该为dim * nTotal。nTotal为准备添加进入底库内部的向量数量，dim为每个向量的维度。</li><li>底库向量总数的取值范围：10000 ≤ nTotal ≤ 1e8。</li><li>该算法不支持添加完底库之后再次添加。Add接口不能与AddWithIds接口混用。</li></ul>
</td>
</tr>
</tbody>
</table>

## AddWithIds接口<a name="ZH-CN_TOPIC_0000002044829957"></a>

<a name="table2436200181918"></a>
<table><tbody><tr id="row6468120161919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p164681020199"><a name="p164681020199"></a><a name="p164681020199"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13468150121916"><a name="p13468150121916"></a><a name="p13468150121916"></a>APP_ERROR AddWithIds (const std::vector&lt;float&gt;&amp; baseRawData, const std::vector&lt;int64_t&gt;&amp; ids);</p>
</td>
</tr>
<tr id="row1846914041914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p546913041920"><a name="p546913041920"></a><a name="p546913041920"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1646980171915"><a name="p1646980171915"></a><a name="p1646980171915"></a>向AscendIndexGreat底库中添加新的特征向量。使用AddWithIds接口添加特征，对应特征的默认ids为[0, ntotal)。</p>
</td>
</tr>
<tr id="row174691500199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5469130151918"><a name="p5469130151918"></a><a name="p5469130151918"></a><strong id="b2469170181919"><a name="b2469170181919"></a><a name="b2469170181919"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul14839152610572"></a><a name="ul14839152610572"></a><ul id="ul14839152610572"><li><strong id="b480732014573"><a name="b480732014573"></a><a name="b480732014573"></a>const std::vector&lt;float&gt;&amp; baseRawData</strong>：添加进底库的特征向量。</li><li><strong id="b41751123125714"><a name="b41751123125714"></a><a name="b41751123125714"></a>const std::vector&lt;int64_t&gt;&amp; ids</strong>：添加进底库的特征向量ID。ID在Index实例中需唯一。</li></ul>
</td>
</tr>
<tr id="row64692010199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1446920091916"><a name="p1446920091916"></a><a name="p1446920091916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1046918051915"><a name="p1046918051915"></a><a name="p1046918051915"></a>无</p>
</td>
</tr>
<tr id="row34692006195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1546910171910"><a name="p1546910171910"></a><a name="p1546910171910"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row5469140141913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p154691800193"><a name="p154691800193"></a><a name="p154691800193"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul112903341573"></a><a name="ul112903341573"></a><ul id="ul112903341573"><li>此处数组“baseRawData”的长度应该为dim * nTotal。nTotal为准备添加进入底库内部的向量数量，dim为每个向量的维度。</li><li>底库向量总数的取值范围：10000 ≤ nTotal ≤ 1e8。</li><li><span class="parmname" id="parmname1942422062518"><a name="parmname1942422062518"></a><a name="parmname1942422062518"></a>“ids”</span>长度必须为nTotal，用户需要根据自己的业务场景，保证<span class="parmname" id="parmname20685131262110"><a name="parmname20685131262110"></a><a name="parmname20685131262110"></a>“ids”</span>的合法性，如底库中存在重复的ID，检索结果中的"label"将无法对应具体的底库向量。</li><li>该算法不支持添加完底库之后再次添加。AddWithIds接口不能与Add接口混用。</li></ul>
</td>
</tr>
</tbody>
</table>

## LoadIndex接口<a name="ZH-CN_TOPIC_0000002008751978"></a>

<a name="table17789162191912"></a>
<table><tbody><tr id="row8827202101911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18271621181913"><a name="p18271621181913"></a><a name="p18271621181913"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p20827821181912"><a name="p20827821181912"></a><a name="p20827821181912"></a>APP_ERROR LoadIndex(const std::string&amp; indexPath);</p>
</td>
</tr>
<tr id="row158271121181911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1482782119190"><a name="p1482782119190"></a><a name="p1482782119190"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p282792116190"><a name="p282792116190"></a><a name="p282792116190"></a>将Index结构从磁盘读入，包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row2082762118194"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p282832121910"><a name="p282832121910"></a><a name="p282832121910"></a><strong id="b178281421121916"><a name="b178281421121916"></a><a name="b178281421121916"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1828121191916"><a name="p1828121191916"></a><a name="p1828121191916"></a><strong id="b208843453576"><a name="b208843453576"></a><a name="b208843453576"></a>const std::string&amp; indexPath</strong>：加载KMode索引的路径。</p>
</td>
</tr>
<tr id="row168281213195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p4828621151916"><a name="p4828621151916"></a><a name="p4828621151916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p182842113195"><a name="p182842113195"></a><a name="p182842113195"></a>无</p>
</td>
</tr>
<tr id="row138282021121914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p8828521151915"><a name="p8828521151915"></a><a name="p8828521151915"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1282819214195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p198281721161912"><a name="p198281721161912"></a><a name="p198281721161912"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p8828122191913"><a name="p8828122191913"></a><a name="p8828122191913"></a>“indexPath”对应的文件为调用WriteIndex方法得到的落盘文件，程序执行用户对其有读权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

<a name="table98570373191"></a>
<table><tbody><tr id="row17884153751918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p15884537171910"><a name="p15884537171910"></a><a name="p15884537171910"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2884737121918"><a name="p2884737121918"></a><a name="p2884737121918"></a>APP_ERROR LoadIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</p>
</td>
</tr>
<tr id="row38841379192"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1588453721918"><a name="p1588453721918"></a><a name="p1588453721918"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1488510377191"><a name="p1488510377191"></a><a name="p1488510377191"></a>将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和原始数据。</p>
</td>
</tr>
<tr id="row888533717196"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128851337151917"><a name="p128851337151917"></a><a name="p128851337151917"></a><strong id="b588573715193"><a name="b588573715193"></a><a name="b588573715193"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1588563711918"><a name="p1588563711918"></a><a name="p1588563711918"></a><strong id="b89721956105715"><a name="b89721956105715"></a><a name="b89721956105715"></a>const std::string&amp; aModeIndexPath</strong>：加载AMode索引的路径。</p>
<p id="p11885193717191"><a name="p11885193717191"></a><a name="p11885193717191"></a><strong id="b1920219018585"><a name="b1920219018585"></a><a name="b1920219018585"></a>const std::string&amp; kModeIndexPath</strong>：加载KMode索引的路径。</p>
</td>
</tr>
<tr id="row7885163731911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2885133721916"><a name="p2885133721916"></a><a name="p2885133721916"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2088593741910"><a name="p2088593741910"></a><a name="p2088593741910"></a>无</p>
</td>
</tr>
<tr id="row208858371192"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p288543731913"><a name="p288543731913"></a><a name="p288543731913"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p19824113093620"><a name="p19824113093620"></a><a name="p19824113093620"></a><strong id="b1482412303365"><a name="b1482412303365"></a><a name="b1482412303365"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row788513751910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12885173771912"><a name="p12885173771912"></a><a name="p12885173771912"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20885143791917"><a name="p20885143791917"></a><a name="p20885143791917"></a>“aModeIndexPath”和“kModeIndexPath”对应的文件为调用WriteIndex方法得到的落盘文件，程序执行用户对其有读权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

## WriteIndex接口<a name="ZH-CN_TOPIC_0000002044950957"></a>

<a name="table84194504191"></a>
<table><tbody><tr id="row1244255016194"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11442165011196"><a name="p11442165011196"></a><a name="p11442165011196"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1442950161914"><a name="p1442950161914"></a><a name="p1442950161914"></a>APP_ERROR WriteIndex(const std::string&amp; indexPath);</p>
</td>
</tr>
<tr id="row4442135021918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p154421650201911"><a name="p154421650201911"></a><a name="p154421650201911"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p94421506192"><a name="p94421506192"></a><a name="p94421506192"></a>将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row19442050191916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p4442205041914"><a name="p4442205041914"></a><a name="p4442205041914"></a><strong id="b1044275020194"><a name="b1044275020194"></a><a name="b1044275020194"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6442250101914"><a name="p6442250101914"></a><a name="p6442250101914"></a>无</p>
</td>
</tr>
<tr id="row134421050151911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p444385091919"><a name="p444385091919"></a><a name="p444385091919"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p11443650111916"><a name="p11443650111916"></a><a name="p11443650111916"></a><strong id="b1940413117581"><a name="b1940413117581"></a><a name="b1940413117581"></a>const std::string&amp; indexPath</strong>：写入KMode索引的路径。</p>
</td>
</tr>
<tr id="row1844395011910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p644345031917"><a name="p644345031917"></a><a name="p644345031917"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1144355061917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p344375061912"><a name="p344375061912"></a><a name="p344375061912"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6443450161914"><a name="p6443450161914"></a><a name="p6443450161914"></a>用户需要保证“indexPath”文件路径所在的目录存在，且执行用户对目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

<a name="table14392122132014"></a>
<table><tbody><tr id="row441919215201"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9419132122010"><a name="p9419132122010"></a><a name="p9419132122010"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1741916213205"><a name="p1741916213205"></a><a name="p1741916213205"></a>APP_ERROR WriteIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</p>
</td>
</tr>
<tr id="row1141920242016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p134192292018"><a name="p134192292018"></a><a name="p134192292018"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p142017252016"><a name="p142017252016"></a><a name="p142017252016"></a>将Index结构写入磁盘，写入磁盘的数据包括压缩降维后的特征向量和码本数据。</p>
</td>
</tr>
<tr id="row18420827206"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p124203215208"><a name="p124203215208"></a><a name="p124203215208"></a><strong id="b4420132162013"><a name="b4420132162013"></a><a name="b4420132162013"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p18420520202"><a name="p18420520202"></a><a name="p18420520202"></a>无</p>
</td>
</tr>
<tr id="row642013252016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p154202232018"><a name="p154202232018"></a><a name="p154202232018"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><a name="ul71911043155818"></a><a name="ul71911043155818"></a><ul id="ul71911043155818"><li>const std::string&amp; aModeIndexPath：写入AMode索引的路径。</li><li>const std::string&amp; kModeIndexPath：写入KMode索引的路径。</li></ul>
</td>
</tr>
<tr id="row1842013232015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p842012215201"><a name="p842012215201"></a><a name="p842012215201"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7875154210361"><a name="p7875154210361"></a><a name="p7875154210361"></a><strong id="b187554215365"><a name="b187554215365"></a><a name="b187554215365"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row142011213209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1242010282015"><a name="p1242010282015"></a><a name="p1242010282015"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p54203216202"><a name="p54203216202"></a><a name="p54203216202"></a>用户需要保证“aModeIndexPath”和“kModeIndexPath”文件路径所在的目录存在，且执行用户对目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

## AddCodeBooks接口<a name="ZH-CN_TOPIC_0000002008751982"></a>

<a name="table339181620207"></a>
<table><tbody><tr id="row20640163209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1564616122013"><a name="p1564616122013"></a><a name="p1564616122013"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3641716102017"><a name="p3641716102017"></a><a name="p3641716102017"></a>APP_ERROR AddCodeBooks(const std::string&amp; codeBooksPath);</p>
</td>
</tr>
<tr id="row66411167203"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p2064121632013"><a name="p2064121632013"></a><a name="p2064121632013"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p964716102017"><a name="p964716102017"></a><a name="p964716102017"></a>加载已经生成完毕的码本到Index。</p>
</td>
</tr>
<tr id="row7647167203"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p764101617204"><a name="p764101617204"></a><a name="p764101617204"></a><strong id="b1664111618206"><a name="b1664111618206"></a><a name="b1664111618206"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1364116172012"><a name="p1364116172012"></a><a name="p1364116172012"></a><strong id="b178911721592"><a name="b178911721592"></a><a name="b178911721592"></a>const std::string&amp; codeBooksPath</strong>：加载已经生成完毕的码本路径。</p>
</td>
</tr>
<tr id="row1564121612202"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p46441612208"><a name="p46441612208"></a><a name="p46441612208"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1064916152011"><a name="p1064916152011"></a><a name="p1064916152011"></a>无</p>
</td>
</tr>
<tr id="row76410163209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p764016192011"><a name="p764016192011"></a><a name="p764016192011"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row16641816182015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15640163208"><a name="p15640163208"></a><a name="p15640163208"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15641164207"><a name="p15641164207"></a><a name="p15641164207"></a>该接口仅能在索引初始化“AKMode”时使用。</p>
<p id="p1865141682018"><a name="p1865141682018"></a><a name="p1865141682018"></a>用户应该保证“codeBooksPath”文件路径所在的目录存在，且该文件内容必须为有效的码本。出于安全加固考虑，目录层级中不能含有软链接。</p>
</td>
</tr>
</tbody>
</table>

## Search接口<a name="ZH-CN_TOPIC_0000002008910274"></a>

<a name="table537563852013"></a>
<table><tbody><tr id="row04171138192013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16417103813201"><a name="p16417103813201"></a><a name="p16417103813201"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1241714381201"><a name="p1241714381201"></a><a name="p1241714381201"></a>APP_ERROR Search(const AscendIndexSearchParams&amp; searchParams);</p>
</td>
</tr>
<tr id="row9417173892011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3417123810208"><a name="p3417123810208"></a><a name="p3417123810208"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p741743817204"><a name="p741743817204"></a><a name="p741743817204"></a>实现AscendIndexGreat特征向量查询接口，根据输入的特征向量返回最相似的“topK”条特征的距离及ID。</p>
</td>
</tr>
<tr id="row16417123814205"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7417173812013"><a name="p7417173812013"></a><a name="p7417173812013"></a><strong id="b4417738192010"><a name="b4417738192010"></a><a name="b4417738192010"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p963412085712"><a name="p963412085712"></a><a name="p963412085712"></a>searchParams结构体见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams接口</a>。</p>
<p id="p12417638202012"><a name="p12417638202012"></a><a name="p12417638202012"></a><strong id="b4417163812018"><a name="b4417163812018"></a><a name="b4417163812018"></a>size_t n：</strong>查询的特征向量的条数<strong id="b741718386209"><a name="b741718386209"></a><a name="b741718386209"></a>。</strong></p>
<p id="p101561712152015"><a name="p101561712152015"></a><a name="p101561712152015"></a><strong id="b5821914326"><a name="b5821914326"></a><a name="b5821914326"></a>std::vector&lt;float&gt;&amp; queryData：</strong>特征向量数据<strong id="b24171438182010"><a name="b24171438182010"></a><a name="b24171438182010"></a>。</strong></p>
<p id="p124173383205"><a name="p124173383205"></a><a name="p124173383205"></a><strong id="b94171338162017"><a name="b94171338162017"></a><a name="b94171338162017"></a>int topK：</strong>需要返回的最相似的结果个数<strong id="b441718389200"><a name="b441718389200"></a><a name="b441718389200"></a>。</strong></p>
</td>
</tr>
<tr id="row44171383207"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1441833813206"><a name="p1441833813206"></a><a name="p1441833813206"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p37091635161713"><a name="p37091635161713"></a><a name="p37091635161713"></a><strong id="b717718132327"><a name="b717718132327"></a><a name="b717718132327"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前“topK”个向量间的距离值。</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b185081513216"><a name="b185081513216"></a><a name="b185081513216"></a>std::vector&lt;int64_t&gt;&amp; labels</strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row3418638112011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p04181638182019"><a name="p04181638182019"></a><a name="p04181638182019"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1841853812014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p641863810207"><a name="p641863810207"></a><a name="p641863810207"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1571185610598"></a><a name="ul1571185610598"></a><ul id="ul1571185610598"><li>topK ∈ (0, 4096]</li><li><strong id="b189911330384"><a name="b189911330384"></a><a name="b189911330384"></a>n</strong>∈ (0, 10000]</li><li>queryData不能为空，且数据长度必须大于等于n * dim。</li><li>dists不能为空，且数据长度必须大于等于n * topK。</li><li>labels不能为空，且数据长度必须大于等于n * topK。</li></ul>
</td>
</tr>
</tbody>
</table>

## SearchWithMask接口<a name="ZH-CN_TOPIC_0000002044950961"></a>

<a name="table186956182018"></a>
<table><tbody><tr id="row1252165642015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p552956182011"><a name="p552956182011"></a><a name="p552956182011"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p165218566205"><a name="p165218566205"></a><a name="p165218566205"></a>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; searchParams, const std::vector&lt;uint8_t&gt;&amp; mask);</p>
</td>
</tr>
<tr id="row1352165682013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1052456192017"><a name="p1052456192017"></a><a name="p1052456192017"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p95317562207"><a name="p95317562207"></a><a name="p95317562207"></a>实现AscendIndexGreat特征向量查询接口，根据输入的特征向量返回最相似的“topK”条特征的距离及ID，且用户可以输入一个uint8数组来掩盖特定底库ID，使该ID对应的特征向量不参与检索。</p>
</td>
</tr>
<tr id="row13531356182016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p753135672013"><a name="p753135672013"></a><a name="p753135672013"></a><strong id="b17531566204"><a name="b17531566204"></a><a name="b17531566204"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p963412085712"><a name="p963412085712"></a><a name="p963412085712"></a>searchParams结构体见<a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams接口">AscendIndexSearchParams接口</a></p>
<p id="p153656142016"><a name="p153656142016"></a><a name="p153656142016"></a><strong id="b185325613204"><a name="b185325613204"></a><a name="b185325613204"></a>size_t n：</strong>查询的特征向量的条数。</p>
<p id="p1753145682011"><a name="p1753145682011"></a><a name="p1753145682011"></a><strong id="b145725499307"><a name="b145725499307"></a><a name="b145725499307"></a>std::vector&lt;float&gt;&amp; queryData：</strong>特征向量数据。</p>
<p id="p353256152013"><a name="p353256152013"></a><a name="p353256152013"></a><strong id="b853656202019"><a name="b853656202019"></a><a name="b853656202019"></a>int topK：</strong>需要返回的最相似的结果个数。</p>
<p id="p14531556102014"><a name="p14531556102014"></a><a name="p14531556102014"></a><strong id="b36191154133015"><a name="b36191154133015"></a><a name="b36191154133015"></a>const std::vector&lt;uint8_t&gt;&amp; mask</strong>：外部输入的额外的过滤mask，以bit为单位，0代表过滤该条特征；1代表选中该条特征。</p>
</td>
</tr>
<tr id="row19531156172019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p15355682013"><a name="p15355682013"></a><a name="p15355682013"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1620165312183"><a name="p1620165312183"></a><a name="p1620165312183"></a><strong id="b77168589309"><a name="b77168589309"></a><a name="b77168589309"></a>std::vector&lt;float&gt;&amp; dists</strong>：查询向量与距离最近的前“topK”个向量间的距离值。</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b1474221153119"><a name="b1474221153119"></a><a name="b1474221153119"></a>std::vector&lt;int64_t&gt;&amp;</strong> <strong id="b148237305104"><a name="b148237305104"></a><a name="b148237305104"></a>labels</strong>：查询的距离最近的前“topK”个向量的ID。当有效的检索结果不足“topK”个时，剩余无效label用-1填充。</p>
</td>
</tr>
<tr id="row65385692010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1531756152010"><a name="p1531756152010"></a><a name="p1531756152010"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1453105616207"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p253155611208"><a name="p253155611208"></a><a name="p253155611208"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul104452596013"></a><a name="ul104452596013"></a><ul id="ul104452596013"><li>topK ∈ (0, 4096]</li><li>n ∈ (0, 10000]</li><li>queryData不能为空，且数据长度必须大于等于n * dim。</li><li>dists不能为空，且指向的数据长度必须大于等于n * topK。</li><li>labels不能为空，且指向的数据长度必须大于等于n * topK。</li><li>mask指向的数据总量必须大于等于n * ceil(nTotal / 8)。</li></ul>
</td>
</tr>
</tbody>
</table>

## GetNTotal接口<a name="ZH-CN_TOPIC_0000002044829965"></a>

<a name="table971712872115"></a>
<table><tbody><tr id="row11742385218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p474298192115"><a name="p474298192115"></a><a name="p474298192115"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p47421585211"><a name="p47421585211"></a><a name="p47421585211"></a>APP_ERROR GetNTotal (uint64_t&amp; nTotal) const;</p>
</td>
</tr>
<tr id="row207427862111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1074298172118"><a name="p1074298172118"></a><a name="p1074298172118"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p117428872117"><a name="p117428872117"></a><a name="p117428872117"></a>获取AscendIndexGreat已添加进底库的特征向量数量。</p>
</td>
</tr>
<tr id="row47428812217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p19742168172110"><a name="p19742168172110"></a><a name="p19742168172110"></a><strong id="b107421685215"><a name="b107421685215"></a><a name="b107421685215"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p174218812216"><a name="p174218812216"></a><a name="p174218812216"></a>无</p>
</td>
</tr>
<tr id="row1574215810219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2742188142115"><a name="p2742188142115"></a><a name="p2742188142115"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p474213822110"><a name="p474213822110"></a><a name="p474213822110"></a><strong id="b53981719918"><a name="b53981719918"></a><a name="b53981719918"></a>uint64_t&amp; nTotal</strong>：已添加进底库的特征向量数量。</p>
</td>
</tr>
<tr id="row374210816218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p11742108182113"><a name="p11742108182113"></a><a name="p11742108182113"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1174319812119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p27431285217"><a name="p27431285217"></a><a name="p27431285217"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p157432812211"><a name="p157432812211"></a><a name="p157432812211"></a>无</p>
</td>
</tr>
</tbody>
</table>

## GetDim接口<a name="ZH-CN_TOPIC_0000002008751986"></a>

<a name="table113422226216"></a>
<table><tbody><tr id="row336622232116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p336632292112"><a name="p336632292112"></a><a name="p336632292112"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p836612222213"><a name="p836612222213"></a><a name="p836612222213"></a>APP_ERROR GetDim(int&amp; dim) const;</p>
</td>
</tr>
<tr id="row11366122212211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p123660225216"><a name="p123660225216"></a><a name="p123660225216"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p17366182232119"><a name="p17366182232119"></a><a name="p17366182232119"></a>获取AscendIndexGreat已添加进底库的特征向量的维度。</p>
</td>
</tr>
<tr id="row13366172242112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p9366112272118"><a name="p9366112272118"></a><a name="p9366112272118"></a><strong id="b10366142210217"><a name="b10366142210217"></a><a name="b10366142210217"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16366182210213"><a name="p16366182210213"></a><a name="p16366182210213"></a>无</p>
</td>
</tr>
<tr id="row10366152211214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p63661422162118"><a name="p63661422162118"></a><a name="p63661422162118"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18366422142117"><a name="p18366422142117"></a><a name="p18366422142117"></a><strong id="b736392718117"><a name="b736392718117"></a><a name="b736392718117"></a>int&amp; dim</strong>：已添加进底库的特征向量的维度。</p>
</td>
</tr>
<tr id="row9366182282110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12366122213212"><a name="p12366122213212"></a><a name="p12366122213212"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row14367522172117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p536720226216"><a name="p536720226216"></a><a name="p536720226216"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p5367922162119"><a name="p5367922162119"></a><a name="p5367922162119"></a>无</p>
</td>
</tr>
</tbody>
</table>

## Reset接口<a name="ZH-CN_TOPIC_0000002008910278"></a>

<a name="table1974793512118"></a>
<table><tbody><tr id="row5768235152116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1768183515214"><a name="p1768183515214"></a><a name="p1768183515214"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1676816355218"><a name="p1676816355218"></a><a name="p1676816355218"></a>APP_ERROR Reset();</p>
</td>
</tr>
<tr id="row1576843562115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p11768335132110"><a name="p11768335132110"></a><a name="p11768335132110"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p776983510216"><a name="p776983510216"></a><a name="p776983510216"></a>清空该Index数据保存的数据包括压缩降维后的特征向量和码本数据，同时保留用户初始化索引时输入的参数。</p>
</td>
</tr>
<tr id="row776916357212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17769235112111"><a name="p17769235112111"></a><a name="p17769235112111"></a><strong id="b7769123532119"><a name="b7769123532119"></a><a name="b7769123532119"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p187691735182114"><a name="p187691735182114"></a><a name="p187691735182114"></a>无</p>
</td>
</tr>
<tr id="row157697353212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9769193511213"><a name="p9769193511213"></a><a name="p9769193511213"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8769143514214"><a name="p8769143514214"></a><a name="p8769143514214"></a>无</p>
</td>
</tr>
<tr id="row77691535162118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1676963518218"><a name="p1676963518218"></a><a name="p1676963518218"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row4770735172115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12770123572120"><a name="p12770123572120"></a><a name="p12770123572120"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10770123572119"><a name="p10770123572119"></a><a name="p10770123572119"></a>无</p>
</td>
</tr>
</tbody>
</table>

## SetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002044950965"></a>

<a name="table1011347192118"></a>
<table><tbody><tr id="row7231478219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1824047142117"><a name="p1824047142117"></a><a name="p1824047142117"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1124114712115"><a name="p1124114712115"></a><a name="p1124114712115"></a>APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams&amp; params);</p>
</td>
</tr>
<tr id="row324114712112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p6241447112115"><a name="p6241447112115"></a><a name="p6241447112115"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p924194712114"><a name="p924194712114"></a><a name="p924194712114"></a>设置该Index检索时的超参。</p>
</td>
</tr>
<tr id="row22484711211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p132404719216"><a name="p132404719216"></a><a name="p132404719216"></a><strong id="b1424114782113"><a name="b1424114782113"></a><a name="b1424114782113"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p112464718211"><a name="p112464718211"></a><a name="p112464718211"></a><strong id="b105404714118"><a name="b105404714118"></a><a name="b105404714118"></a>const AscendIndexHyperParams&amp; params</strong>：检索时的超参，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexhyperparams接口">AscendIndexHyperParams</a>。</p>
</td>
</tr>
<tr id="row1824174715218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1241647102111"><a name="p1241647102111"></a><a name="p1241647102111"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16248474211"><a name="p16248474211"></a><a name="p16248474211"></a>无</p>
</td>
</tr>
<tr id="row1424647162118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p324194716213"><a name="p324194716213"></a><a name="p324194716213"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row12484710212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p72464711212"><a name="p72464711212"></a><a name="p72464711212"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4241047102110"><a name="p4241047102110"></a><a name="p4241047102110"></a>无</p>
</td>
</tr>
</tbody>
</table>

## GetHyperSearchParams接口<a name="ZH-CN_TOPIC_0000002400547905"></a>

<a name="table749915518225"></a>
<table><tbody><tr id="row18522851227"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p15221751229"><a name="p15221751229"></a><a name="p15221751229"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2052235122213"><a name="p2052235122213"></a><a name="p2052235122213"></a>APP_ERROR GetHyperSearchParams(AscendIndexHyperParams&amp; params) const;</p>
</td>
</tr>
<tr id="row1752219552217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1852235192213"><a name="p1852235192213"></a><a name="p1852235192213"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1752295162213"><a name="p1752295162213"></a><a name="p1752295162213"></a>获取该Index检索时的检索超参。</p>
</td>
</tr>
<tr id="row15522451223"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p45228522213"><a name="p45228522213"></a><a name="p45228522213"></a><strong id="b1052285172215"><a name="b1052285172215"></a><a name="b1052285172215"></a>输入</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p352213552213"><a name="p352213552213"></a><a name="p352213552213"></a>无</p>
</td>
</tr>
<tr id="row352210511229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14522155102216"><a name="p14522155102216"></a><a name="p14522155102216"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p141731750113612"><a name="p141731750113612"></a><a name="p141731750113612"></a><strong id="b17417143113299"><a name="b17417143113299"></a><a name="b17417143113299"></a>AscendIndexHyperParams&amp; params</strong>：检索时的超参，具体请参见<a href="./13_AscendIndexMixSearchParams.md#ascendindexhyperparams接口">AscendIndexHyperParams</a>。</p>
</td>
</tr>
<tr id="row1252219592211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p185221754223"><a name="p185221754223"></a><a name="p185221754223"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1024995915361"><a name="p1024995915361"></a><a name="p1024995915361"></a><strong id="b324975910365"><a name="b324975910365"></a><a name="b324975910365"></a>APP_ERROR</strong>：调用返回状态，具体请参见接口调用返回值参考。</p>
</td>
</tr>
<tr id="row1652316522217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1752318572219"><a name="p1752318572219"></a><a name="p1752318572219"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1252316522219"><a name="p1252316522219"></a><a name="p1252316522219"></a>无</p>
</td>
</tr>
</tbody>
</table>
