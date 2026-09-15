# `AscendIndexGreat`<a name="en-us_TOPIC_0000002044829945"></a>

## Function Description<a name="en-us_TOPIC_0000002008751966"></a>

This self-developed vector retrieval algorithm provides approximate retrieval for high-dimensional large base libraries on the Ascend side and the Kunpeng side. It uses a self-developed retrieval strategy to retrieve the top `K` most similar vectors from the base library.

The vectors stored in the base library and the query vectors passed to each API must be normalized `float` values.

This algorithm does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you must acquire a lock before use. Otherwise, the retrieval API may fail. It also does not support sharing one Device across threads.

This algorithm is mainly designed for approximate fuzzy search in large base library scenarios, and its accuracy is lower than brute-force retrieval. In small base library scenarios, you are advised to increase the hyperparameter values appropriately to reduce the loss in accuracy.

> [!NOTE]
>
> - When you create an `Index` instance, set `params.dim` according to the actual situation.
> - The `Index` has two algorithm modes: `KMode`, which uses only the Kunpeng-side algorithm, and `AKMode`, which uses the Ascend plus Kunpeng algorithm. In `AKMode`, you must generate the corresponding operators in advance.
> - Ensure that `subSpaceDim` and `nlist` match the corresponding parameters used for codebook training.

## `AscendIndexGreat`<a name="en-us_TOPIC_0000002044829953"></a>

<a name="table5404639201712"></a>
<table><tbody><tr id="row194338394172"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p843363911171"><a name="p843363911171"></a><a name="p843363911171"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p144331039111710"><a name="p144331039111710"></a><a name="p144331039111710"></a><code>AscendIndexGreat(const std::string&amp; mode, const std::vector&lt;int&gt;&amp; deviceList, bool verbose = false);</code></p>
</td>
</tr>
<tr id="row043313981717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p24337391179"><a name="p24337391179"></a><a name="p24337391179"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6433139151710"><a name="p6433139151710"></a><a name="p6433139151710"></a>Constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</p>
</td>
</tr>
<tr id="row124339399175"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p164331391173"><a name="p164331391173"></a><a name="p164331391173"></a><strong id="b5433193991713"><a name="b5433193991713"></a><a name="b5433193991713"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12433193918175"><a name="p12433193918175"></a><a name="p12433193918175"></a><strong id="b16722123816518"><a name="b16722123816518"></a><a name="b16722123816518"></a><code>const std::string&amp; mode</code></strong>: Specifies the algorithm mode.</p>
<p id="p104331039161715"><a name="p104331039161715"></a><a name="p104331039161715"></a><strong id="b769912415512"><a name="b769912415512"></a><a name="b769912415512"></a><code>const std::vector&lt;int&gt;&amp; deviceList</code></strong>: The specified NPU-side device IDs.</p>
<p id="p204336391174"><a name="p204336391174"></a><a name="p204336391174"></a><strong id="b10327344205112"><a name="b10327344205112"></a><a name="b10327344205112"></a><code>bool verbose</code></strong>: Specifies whether to enable the <code>verbose</code> option. When enabled, some operations provide additional print prompts. The default value is <code>false</code>.</p>
</td>
</tr>
<tr id="row443343901718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p8433133912177"><a name="p8433133912177"></a><a name="p8433133912177"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p204331639171718"><a name="p204331639171718"></a><a name="p204331639171718"></a>None</p>
</td>
</tr>
<tr id="row1443373991714"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p743343921714"><a name="p743343921714"></a><a name="p743343921714"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p174331439151714"><a name="p174331439151714"></a><a name="p174331439151714"></a>None</p>
</td>
</tr>
<tr id="row04331639141719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p114331039171716"><a name="p114331039171716"></a><a name="p114331039171716"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1462219489335"></a><a name="ul1462219489335"></a><ul id="ul1462219489335"><li><code>mode</code>: Only <code>KMode</code> and <code>AKMode</code> are supported.</li><li><code>deviceList</code>: Use the <strong id="b3555159175110"><a name="b3555159175110"></a><a name="b3555159175110"></a><code>npu-smi</code></strong> command to query the corresponding NPU IDs. Only one device ID is supported.</li><li>After you create an <code>Index</code> instance with this constructor, you must first call <code>LoadIndex</code> to load the pre-saved <code>Index</code> instance from disk, and then you can perform other operations.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table72261454131719"></a>
<table><tbody><tr id="row18251175431713"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16251154121710"><a name="p16251154121710"></a><a name="p16251154121710"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p76841847185413"><a name="p76841847185413"></a><a name="p76841847185413"></a><code>explicit AscendIndexGreat(const AscendIndexGreatInitParams&amp; kModeInitParams);</code></p>
</td>
</tr>
<tr id="row11251125451717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p172516541177"><a name="p172516541177"></a><a name="p172516541177"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p11251135461713"><a name="p11251135461713"></a><a name="p11251135461713"></a>Constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</p>
</td>
</tr>
<tr id="row82514548179"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p325155441716"><a name="p325155441716"></a><a name="p325155441716"></a><strong id="b2251254161713"><a name="b2251254161713"></a><a name="b2251254161713"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p82516541176"><a name="p82516541176"></a><a name="p82516541176"></a>Initialization parameters required by the <code>Index</code>, specifically <code>kModeInitParams</code>. For details, see <a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams"><code>AscendIndexGreatInitParams</code></a>.</p>
</td>
</tr>
<tr id="row3251354151719"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2252115451717"><a name="p2252115451717"></a><a name="p2252115451717"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1125255410178"><a name="p1125255410178"></a><a name="p1125255410178"></a>None</p>
</td>
</tr>
<tr id="row1725275471717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p2252054121720"><a name="p2252054121720"></a><a name="p2252054121720"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row13252145413171"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p125285413172"><a name="p125285413172"></a><a name="p125285413172"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20252954121720"><a name="p20252954121720"></a><a name="p20252954121720"></a>See the parameter descriptions and constraints in <a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams"><code>AscendIndexGreatInitParams</code></a>.</p>
</td>
</tr>
</tbody>
</table>

<a name="table198261931819"></a>
<table><tbody><tr id="row78491591183"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1484915918184"><a name="p1484915918184"></a><a name="p1484915918184"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1584913901820"><a name="p1584913901820"></a><a name="p1584913901820"></a><code>AscendIndexGreat(const AscendIndexVstarInitParams&amp; aModeInitParams, const AscendIndexGreatInitParams&amp; kModeInitParams);</code></p>
</td>
</tr>
<tr id="row284999121814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p78498919183"><a name="p78498919183"></a><a name="p78498919183"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p13849791188"><a name="p13849791188"></a><a name="p13849791188"></a>Constructor of <code>AscendIndexGreat</code>. It creates a retrieval <code>Index</code> on Ascend.</p>
</td>
</tr>
<tr id="row784912981810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p1184919911189"><a name="p1184919911189"></a><a name="p1184919911189"></a><strong id="b9849692189"><a name="b9849692189"></a><a name="b9849692189"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1984999111812"><a name="p1984999111812"></a><a name="p1984999111812"></a>Initialization parameters required by the <code>Index</code>, specifically <code>aModeInitParams</code> and <code>kModeInitParams</code>. For details, see <a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams"><code>AscendIndexVstarInitParams</code></a> and <a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams"><code>AscendIndexGreatInitParams</code></a>.</p>
</td>
</tr>
<tr id="row5850179121814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1285009121812"><a name="p1285009121812"></a><a name="p1285009121812"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p14850159181811"><a name="p14850159181811"></a><a name="p14850159181811"></a>None</p>
</td>
</tr>
<tr id="row13850199151817"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1185019912183"><a name="p1185019912183"></a><a name="p1185019912183"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p123604384355"><a name="p123604384355"></a><a name="p123604384355"></a><strong id="b536183815357"><a name="b536183815357"></a><a name="b536183815357"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row16850109161814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p985029101810"><a name="p985029101810"></a><a name="p985029101810"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p12850169151811"><a name="p12850169151811"></a><a name="p12850169151811"></a>Refer to the parameter descriptions and constraints in <a href="./13_AscendIndexMixSearchParams.md#ascendindexvstarinitparams"><code>AscendIndexVstarInitParams</code></a> and <a href="./13_AscendIndexMixSearchParams.md#ascendindexgreatinitparams"><code>AscendIndexGreatInitParams</code></a>.</p>
<p id="p122218114501"><a name="p122218114501"></a><a name="p122218114501"></a>The <code>dim</code> values of <code>aModeInitParams</code> and <code>kModeInitParams</code> must be the same.</p>
</td>
</tr>
</tbody>
</table>

<a name="table32891532172215"></a>
<table><tbody><tr id="row1731883213226"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p183181632182220"><a name="p183181632182220"></a><a name="p183181632182220"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3318832182215"><a name="p3318832182215"></a><a name="p3318832182215"></a><code>AscendIndexGreat(const AscendIndexGreat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row831813213224"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p231819329229"><a name="p231819329229"></a><a name="p231819329229"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6318232102217"><a name="p6318232102217"></a><a name="p6318232102217"></a>Declares the copy constructor of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row731817327226"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p631813272217"><a name="p631813272217"></a><a name="p631813272217"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p153181732102215"><a name="p153181732102215"></a><a name="p153181732102215"></a><strong id="b113181932132210"><a name="b113181932132210"></a><a name="b113181932132210"></a><code>const AscendIndexGreat&amp;</code></strong>: A constant <code>AscendIndexGreat</code> object.</p>
</td>
</tr>
<tr id="row20318632102217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p431833212210"><a name="p431833212210"></a><a name="p431833212210"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15318173218228"><a name="p15318173218228"></a><a name="p15318173218228"></a>None</p>
</td>
</tr>
<tr id="row8318153215220"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1831811326227"><a name="p1831811326227"></a><a name="p1831811326227"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p83180322222"><a name="p83180322222"></a><a name="p83180322222"></a>None</p>
</td>
</tr>
<tr id="row131933214228"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p103190320223"><a name="p103190320223"></a><a name="p103190320223"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p7319133272214"><a name="p7319133272214"></a><a name="p7319133272214"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~AscendIndexGreat`<a name="en-us_TOPIC_0000002013257524"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p128341740125117"><a name="p128341740125117"></a><a name="p128341740125117"></a><code>virtual ~AscendIndexGreat() = default;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Destructor of <code>AscendIndexGreat</code>. It destroys the <code>AscendIndexGreat</code> object and releases resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `operator=`<a name="en-us_TOPIC_0000002008751990"></a>

<a name="table39961720122213"></a>
<table><tbody><tr id="row3176213227"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p71752119228"><a name="p71752119228"></a><a name="p71752119228"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1917321192217"><a name="p1917321192217"></a><a name="p1917321192217"></a><code>AscendIndexGreat &amp;operator=(const AscendIndexGreat&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row111762152213"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p71719215225"><a name="p71719215225"></a><a name="p71719215225"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p817921142214"><a name="p817921142214"></a><a name="p817921142214"></a>Declares the assignment operator of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row1217121122217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p41710213220"><a name="p41710213220"></a><a name="p41710213220"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p111732182220"><a name="p111732182220"></a><a name="p111732182220"></a><strong id="b101742192218"><a name="b101742192218"></a><a name="b101742192218"></a><code>const AscendIndexGreat&amp;</code></strong>: A constant <code>AscendIndexGreat</code> object.</p>
</td>
</tr>
<tr id="row3171321172218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1717521132215"><a name="p1717521132215"></a><a name="p1717521132215"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p191714215226"><a name="p191714215226"></a><a name="p191714215226"></a>None</p>
</td>
</tr>
<tr id="row181713210221"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p917182117222"><a name="p917182117222"></a><a name="p917182117222"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p917921142212"><a name="p917921142212"></a><a name="p917921142212"></a>None</p>
</td>
</tr>
<tr id="row117172118229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p618162116224"><a name="p618162116224"></a><a name="p618162116224"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1318182111227"><a name="p1318182111227"></a><a name="p1318182111227"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Add`<a name="en-us_TOPIC_0000002044950953"></a>

<a name="table11133547191811"></a>
<table><tbody><tr id="row1159447111810"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12159184761818"><a name="p12159184761818"></a><a name="p12159184761818"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p161591747181812"><a name="p161591747181812"></a><a name="p161591747181812"></a><code>APP_ERROR Add(const std::vector&lt;float&gt;&amp; baseRawData);</code></p>
</td>
</tr>
<tr id="row10159194716180"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p5159847181814"><a name="p5159847181814"></a><a name="p5159847181814"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2015916478183"><a name="p2015916478183"></a><a name="p2015916478183"></a>Adds new feature vectors to the <code>AscendIndexGreat</code> base library.</p>
</td>
</tr>
<tr id="row11159847131815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p115914712182"><a name="p115914712182"></a><a name="p115914712182"></a><strong id="b615918476186"><a name="b615918476186"></a><a name="b615918476186"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1215974741811"><a name="p1215974741811"></a><a name="p1215974741811"></a><strong id="b14159184715188"><a name="b14159184715188"></a><a name="b14159184715188"></a><code>const std::vector&lt;float&gt;&amp; baseRawData</code></strong>: The feature vectors to add to the base library.</p>
</td>
</tr>
<tr id="row11159647171819"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p18159184718181"><a name="p18159184718181"></a><a name="p18159184718181"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2159147101812"><a name="p2159147101812"></a><a name="p2159147101812"></a>None</p>
</td>
</tr>
<tr id="row615904711813"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1815954712186"><a name="p1815954712186"></a><a name="p1815954712186"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row2160144719188"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p19160947121812"><a name="p19160947121812"></a><a name="p19160947121812"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul19630537566"></a><a name="ul19630537566"></a><ul id="ul19630537566"><li>The length of <code>baseRawData</code> must be <code>dim * nTotal</code>. <code>nTotal</code> is the number of vectors to add to the base library, and <code>dim</code> is the dimension of each vector.</li><li>The valid range of the total number of base library vectors is <code>10000 ≤ nTotal ≤ 1e8</code>.</li><li>This algorithm does not support adding data again after the base library has been added. The <code>Add</code> API cannot be used together with the <code>AddWithIds</code> API.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AddWithIds`<a name="en-us_TOPIC_0000002044829957"></a>

<a name="table2436200181918"></a>
<table><tbody><tr id="row6468120161919"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p164681020199"><a name="p164681020199"></a><a name="p164681020199"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13468150121916"><a name="p13468150121916"></a><a name="p13468150121916"></a><code>APP_ERROR AddWithIds (const std::vector&lt;float&gt;&amp; baseRawData, const std::vector&lt;int64_t&gt;&amp; ids);</code></p>
</td>
</tr>
<tr id="row1846914041914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p546913041920"><a name="p546913041920"></a><a name="p546913041920"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1646980171915"><a name="p1646980171915"></a><a name="p1646980171915"></a>Adds new feature vectors to the <code>AscendIndexGreat</code> base library. When features are added through <code>AddWithIds</code>, the default IDs for the corresponding features are <code>[0, ntotal)</code>.</p>
</td>
</tr>
<tr id="row174691500199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p5469130151918"><a name="p5469130151918"></a><a name="p5469130151918"></a><strong id="b2469170181919"><a name="b2469170181919"></a><a name="b2469170181919"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><a name="ul14839152610572"></a><a name="ul14839152610572"></a><ul id="ul14839152610572"><li><strong id="b480732014573"><a name="b480732014573"></a><a name="b480732014573"></a><code>const std::vector&lt;float&gt;&amp; baseRawData</code></strong>: The feature vectors to add to the base library.</li><li><strong id="b41751123125714"><a name="b41751123125714"></a><a name="b41751123125714"></a><code>const std::vector&lt;int64_t&gt;&amp; ids</code></strong>: IDs of the feature vectors to add to the base library. IDs must be unique within the <code>Index</code> instance.</li></ul>
</td>
</tr>
<tr id="row64692010199"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1446920091916"><a name="p1446920091916"></a><a name="p1446920091916"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1046918051915"><a name="p1046918051915"></a><a name="p1046918051915"></a>None</p>
</td>
</tr>
<tr id="row34692006195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1546910171910"><a name="p1546910171910"></a><a name="p1546910171910"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row5469140141913"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p154691800193"><a name="p154691800193"></a><a name="p154691800193"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul112903341573"></a><a name="ul112903341573"></a><ul id="ul112903341573"><li>The length of <code>baseRawData</code> must be <code>dim * nTotal</code>. <code>nTotal</code> is the number of vectors to add to the base library, and <code>dim</code> is the dimension of each vector.</li><li>The valid range of the total number of base library vectors is <code>10000 ≤ nTotal ≤ 1e8</code>.</li><li>The length of <code>ids</code> must be <code>nTotal</code>. Based on your own business scenario, ensure that <code>ids</code> are valid. If duplicate IDs exist in the base library, the <code>label</code> in the retrieval results cannot correspond to a specific base library vector.</li><li>This algorithm does not support adding data again after the base library has been added. The <code>AddWithIds</code> API cannot be used together with the <code>Add</code> API.</li></ul>
</td>
</tr>
</tbody>
</table>

## `LoadIndex`<a name="en-us_TOPIC_0000002008751978"></a>

<a name="table17789162191912"></a>
<table><tbody><tr id="row8827202101911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p18271621181913"><a name="p18271621181913"></a><a name="p18271621181913"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p20827821181912"><a name="p20827821181912"></a><a name="p20827821181912"></a><code>APP_ERROR LoadIndex(const std::string&amp; indexPath);</code></p>
</td>
</tr>
<tr id="row158271121181911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1482782119190"><a name="p1482782119190"></a><a name="p1482782119190"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p282792116190"><a name="p282792116190"></a><a name="p282792116190"></a>Loads the <code>Index</code> structure from disk, including compressed, dimension-reduced feature vectors and codebook data.</p>
</td>
</tr>
<tr id="row2082762118194"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p282832121910"><a name="p282832121910"></a><a name="p282832121910"></a><strong id="b178281421121916"><a name="b178281421121916"></a><a name="b178281421121916"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1828121191916"><a name="p1828121191916"></a><a name="p1828121191916"></a><strong id="b208843453576"><a name="b208843453576"></a><a name="b208843453576"></a><code>const std::string&amp; indexPath</code></strong>: Path to load the KMode index.</p>
</td>
</tr>
<tr id="row168281213195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p4828621151916"><a name="p4828621151916"></a><a name="p4828621151916"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p182842113195"><a name="p182842113195"></a><a name="p182842113195"></a>None</p>
</td>
</tr>
<tr id="row138282021121914"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p8828521151915"><a name="p8828521151915"></a><a name="p8828521151915"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row1282819214195"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p198281721161912"><a name="p198281721161912"></a><a name="p198281721161912"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p8828122191913"><a name="p8828122191913"></a><a name="p8828122191913"></a>The file corresponding to <code>indexPath</code> must be a persisted file generated by calling <code>WriteIndex</code>, and the running user must have read permission for it. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</p>
</td>
</tr>
</tbody>
</table>

<a name="table98570373191"></a>
<table><tbody><tr id="row17884153751918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p15884537171910"><a name="p15884537171910"></a><a name="p15884537171910"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2884737121918"><a name="p2884737121918"></a><a name="p2884737121918"></a><code>APP_ERROR LoadIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></p>
</td>
</tr>
<tr id="row38841379192"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1588453721918"><a name="p1588453721918"></a><a name="p1588453721918"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1488510377191"><a name="p1488510377191"></a><a name="p1488510377191"></a>Loads the <code>Index</code> structure from disk, including compressed, dimension-reduced feature vectors and the original data.</p>
</td>
</tr>
<tr id="row888533717196"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p128851337151917"><a name="p128851337151917"></a><a name="p128851337151917"></a><strong id="b588573715193"><a name="b588573715193"></a><a name="b588573715193"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1588563711918"><a name="p1588563711918"></a><a name="p1588563711918"></a><strong id="b89721956105715"><a name="b89721956105715"></a><a name="b89721956105715"></a><code>const std::string&amp; aModeIndexPath</code></strong>: Path to load the AMode index.</p>
<p id="p11885193717191"><a name="p11885193717191"></a><a name="p11885193717191"></a><strong id="b1920219018585"><a name="b1920219018585"></a><a name="b1920219018585"></a><code>const std::string&amp; kModeIndexPath</code></strong>: Path to load the KMode index.</p>
</td>
</tr>
<tr id="row7885163731911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2885133721916"><a name="p2885133721916"></a><a name="p2885133721916"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p2088593741910"><a name="p2088593741910"></a><a name="p2088593741910"></a>None</p>
</td>
</tr>
<tr id="row208858371192"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p288543731913"><a name="p288543731913"></a><a name="p288543731913"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p19824113093620"><a name="p19824113093620"></a><a name="p19824113093620"></a><strong id="b1482412303365"><a name="b1482412303365"></a><a name="b1482412303365"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row788513751910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12885173771912"><a name="p12885173771912"></a><a name="p12885173771912"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p20885143791917"><a name="p20885143791917"></a><a name="p20885143791917"></a>The files corresponding to <code>aModeIndexPath</code> and <code>kModeIndexPath</code> must be the persisted files generated by calling <code>WriteIndex</code>, and the running user must have read permission for them. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</p>
</td>
</tr>
</tbody>
</table>

## `WriteIndex`<a name="en-us_TOPIC_0000002044950957"></a>

<a name="table84194504191"></a>
<table><tbody><tr id="row1244255016194"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p11442165011196"><a name="p11442165011196"></a><a name="p11442165011196"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1442950161914"><a name="p1442950161914"></a><a name="p1442950161914"></a><code>APP_ERROR WriteIndex(const std::string&amp; indexPath);</code></p>
</td>
</tr>
<tr id="row4442135021918"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p154421650201911"><a name="p154421650201911"></a><a name="p154421650201911"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p94421506192"><a name="p94421506192"></a><a name="p94421506192"></a>Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data.</p>
</td>
</tr>
<tr id="row19442050191916"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p4442205041914"><a name="p4442205041914"></a><a name="p4442205041914"></a><strong id="b1044275020194"><a name="b1044275020194"></a><a name="b1044275020194"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p6442250101914"><a name="p6442250101914"></a><a name="p6442250101914"></a>None</p>
</td>
</tr>
<tr id="row134421050151911"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p444385091919"><a name="p444385091919"></a><a name="p444385091919"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p11443650111916"><a name="p11443650111916"></a><a name="p11443650111916"></a><strong id="b1940413117581"><a name="b1940413117581"></a><a name="b1940413117581"></a><code>const std::string&amp; indexPath</code></strong>: Path to write the KMode index.</p>
</td>
</tr>
<tr id="row1844395011910"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p644345031917"><a name="p644345031917"></a><a name="p644345031917"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row1144355061917"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p344375061912"><a name="p344375061912"></a><a name="p344375061912"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p6443450161914"><a name="p6443450161914"></a><a name="p6443450161914"></a>The user must ensure that the directory containing the <code>indexPath</code> file exists and that the running user has write permission for that directory. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</p>
</td>
</tr>
</tbody>
</table>

<a name="table14392122132014"></a>
<table><tbody><tr id="row441919215201"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p9419132122010"><a name="p9419132122010"></a><a name="p9419132122010"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1741916213205"><a name="p1741916213205"></a><a name="p1741916213205"></a><code>APP_ERROR WriteIndex(const std::string&amp; aModeIndexPath, const std::string&amp; kModeIndexPath);</code></p>
</td>
</tr>
<tr id="row1141920242016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p134192292018"><a name="p134192292018"></a><a name="p134192292018"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p142017252016"><a name="p142017252016"></a><a name="p142017252016"></a>Writes the <code>Index</code> structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data.</p>
</td>
</tr>
<tr id="row18420827206"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p124203215208"><a name="p124203215208"></a><a name="p124203215208"></a><strong id="b4420132162013"><a name="b4420132162013"></a><a name="b4420132162013"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p18420520202"><a name="p18420520202"></a><a name="p18420520202"></a>None</p>
</td>
</tr>
<tr id="row642013252016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p154202232018"><a name="p154202232018"></a><a name="p154202232018"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><a name="ul71911043155818"></a><a name="ul71911043155818"></a><ul id="ul71911043155818"><li><code>const std::string&amp; aModeIndexPath</code>: Path to write the AMode index.</li><li><code>const std::string&amp; kModeIndexPath</code>: Path to write the KMode index.</li></ul>
</td>
</tr>
<tr id="row1842013232015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p842012215201"><a name="p842012215201"></a><a name="p842012215201"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p7875154210361"><a name="p7875154210361"></a><a name="p7875154210361"></a><strong id="b187554215365"><a name="b187554215365"></a><a name="b187554215365"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row142011213209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1242010282015"><a name="p1242010282015"></a><a name="p1242010282015"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p54203216202"><a name="p54203216202"></a><a name="p54203216202"></a>The user must ensure that the directories containing the <code>aModeIndexPath</code> and <code>kModeIndexPath</code> file paths exist and that the running user has write permission for those directories. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</p>
</td>
</tr>
</tbody>
</table>

## `AddCodeBooks`<a name="en-us_TOPIC_0000002008751982"></a>

<a name="table339181620207"></a>
<table><tbody><tr id="row20640163209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1564616122013"><a name="p1564616122013"></a><a name="p1564616122013"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p3641716102017"><a name="p3641716102017"></a><a name="p3641716102017"></a><code>APP_ERROR AddCodeBooks(const std::string&amp; codeBooksPath);</code></p>
</td>
</tr>
<tr id="row66411167203"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p2064121632013"><a name="p2064121632013"></a><a name="p2064121632013"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p964716102017"><a name="p964716102017"></a><a name="p964716102017"></a>Loads an already generated codebook into the <code>Index</code>.</p>
</td>
</tr>
<tr id="row7647167203"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p764101617204"><a name="p764101617204"></a><a name="p764101617204"></a><strong id="b1664111618206"><a name="b1664111618206"></a><a name="b1664111618206"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1364116172012"><a name="p1364116172012"></a><a name="p1364116172012"></a><strong id="b178911721592"><a name="b178911721592"></a><a name="b178911721592"></a><code>const std::string&amp; codeBooksPath</code></strong>: Path to the generated codebook.</p>
</td>
</tr>
<tr id="row1564121612202"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p46441612208"><a name="p46441612208"></a><a name="p46441612208"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1064916152011"><a name="p1064916152011"></a><a name="p1064916152011"></a>None</p>
</td>
</tr>
<tr id="row76410163209"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p764016192011"><a name="p764016192011"></a><a name="p764016192011"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row16641816182015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p15640163208"><a name="p15640163208"></a><a name="p15640163208"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p15641164207"><a name="p15641164207"></a><a name="p15641164207"></a>This API can only be used when initializing an index in <code>AKMode</code>.</p>
<p id="p1865141682018"><a name="p1865141682018"></a><a name="p1865141682018"></a>The user must ensure that the directory containing the <code>codeBooksPath</code> file exists, and the file content must be a valid codebook. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy.</p>
</td>
</tr>
</tbody>
</table>

## `Search`<a name="en-us_TOPIC_0000002008910274"></a>

<a name="table537563852013"></a>
<table><tbody><tr id="row04171138192013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p16417103813201"><a name="p16417103813201"></a><a name="p16417103813201"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1241714381201"><a name="p1241714381201"></a><a name="p1241714381201"></a><code>APP_ERROR Search(const AscendIndexSearchParams&amp; searchParams);</code></p>
</td>
</tr>
<tr id="row9417173892011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p3417123810208"><a name="p3417123810208"></a><a name="p3417123810208"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p741743817204"><a name="p741743817204"></a><a name="p741743817204"></a>Implements the <code>AscendIndexGreat</code> feature-vector query API. Based on the input feature vectors, it returns the distances and IDs of the most similar <code>topK</code> features.</p>
</td>
</tr>
<tr id="row16417123814205"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p7417173812013"><a name="p7417173812013"></a><a name="p7417173812013"></a><strong id="b4417738192010"><a name="b4417738192010"></a><a name="b4417738192010"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p963412085712"><a name="p963412085712"></a><a name="p963412085712"></a>For the <code>searchParams</code> structure, see the <a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams"><code>AscendIndexSearchParams</code></a> API.</p>
<p id="p12417638202012"><a name="p12417638202012"></a><a name="p12417638202012"></a><strong id="b4417163812018"><a name="b4417163812018"></a><a name="b4417163812018"></a><code>size_t n</code></strong>: Number of query feature vectors.</p>
<p id="p101561712152015"><a name="p101561712152015"></a><a name="p101561712152015"></a><strong id="b5821914326"><a name="b5821914326"></a><a name="b5821914326"></a><code>std::vector&lt;float&gt;&amp; queryData</code></strong>: Feature vector data.</p>
<p id="p124173383205"><a name="p124173383205"></a><a name="p124173383205"></a><strong id="b94171338162017"><a name="b94171338162017"></a><a name="b94171338162017"></a><code>int topK</code></strong>: Number of most similar results to return.</p>
</td>
</tr>
<tr id="row44171383207"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1441833813206"><a name="p1441833813206"></a><a name="p1441833813206"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p37091635161713"><a name="p37091635161713"></a><a name="p37091635161713"></a><strong id="b717718132327"><a name="b717718132327"></a><a name="b717718132327"></a><code>std::vector&lt;float&gt;&amp; dists</code></strong>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b185081513216"><a name="b185081513216"></a><a name="b185081513216"></a><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</p>
</td>
</tr>
<tr id="row3418638112011"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p04181638182019"><a name="p04181638182019"></a><a name="p04181638182019"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row1841853812014"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p641863810207"><a name="p641863810207"></a><a name="p641863810207"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1571185610598"></a><a name="ul1571185610598"></a><ul id="ul1571185610598"><li><code>topK</code> ∈ (0, 4096].</li><li><code>n</code> ∈ (0, 10000].</li><li><code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>.</li><li><code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</li><li><code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithMask`<a name="en-us_TOPIC_0000002044950961"></a>

<a name="table186956182018"></a>
<table><tbody><tr id="row1252165642015"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p552956182011"><a name="p552956182011"></a><a name="p552956182011"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p165218566205"><a name="p165218566205"></a><a name="p165218566205"></a><code>APP_ERROR SearchWithMask(const AscendIndexSearchParams&amp; searchParams, const std::vector&lt;uint8_t&gt;&amp; mask);</code></p>
</td>
</tr>
<tr id="row1352165682013"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1052456192017"><a name="p1052456192017"></a><a name="p1052456192017"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p95317562207"><a name="p95317562207"></a><a name="p95317562207"></a>Implements the <code>AscendIndexGreat</code> feature-vector query API. Based on the input feature vectors, it returns the distances and IDs of the most similar <code>topK</code> features. In addition, the user can input a <code>uint8</code> array to mask specific base library IDs so that the feature vectors corresponding to those IDs are excluded from retrieval.</p>
</td>
</tr>
<tr id="row13531356182016"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p753135672013"><a name="p753135672013"></a><a name="p753135672013"></a><strong id="b17531566204"><a name="b17531566204"></a><a name="b17531566204"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p963412085712"><a name="p963412085712"></a><a name="p963412085712"></a>For the <code>searchParams</code> structure, see the <a href="./13_AscendIndexMixSearchParams.md#ascendindexsearchparams"><code>AscendIndexSearchParams</code></a> API.</p>
<p id="p153656142016"><a name="p153656142016"></a><a name="p153656142016"></a><strong id="b185325613204"><a name="b185325613204"></a><a name="b185325613204"></a><code>size_t n</code></strong>: Number of query feature vectors.</p>
<p id="p1753145682011"><a name="p1753145682011"></a><a name="p1753145682011"></a><strong id="b145725499307"><a name="b145725499307"></a><a name="b145725499307"></a><code>std::vector&lt;float&gt;&amp; queryData</code></strong>: Feature vector data.</p>
<p id="p353256152013"><a name="p353256152013"></a><a name="p353256152013"></a><strong id="b853656202019"><a name="b853656202019"></a><a name="b853656202019"></a><code>int topK</code></strong>: Number of most similar results to return.</p>
<p id="p14531556102014"><a name="p14531556102014"></a><a name="p14531556102014"></a><strong id="b36191154133015"><a name="b36191154133015"></a><a name="b36191154133015"></a><code>const std::vector&lt;uint8_t&gt;&amp; mask</code></strong>: External filtering mask, in bits. <code>0</code> means the feature is filtered out; <code>1</code> means the feature is selected.</p>
</td>
</tr>
<tr id="row19531156172019"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p15355682013"><a name="p15355682013"></a><a name="p15355682013"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1620165312183"><a name="p1620165312183"></a><a name="p1620165312183"></a><strong id="b77168589309"><a name="b77168589309"></a><a name="b77168589309"></a><code>std::vector&lt;float&gt;&amp; dists</code></strong>: Distance values between the query vectors and the top <code>topK</code> nearest vectors.</p>
<p id="p9372436142115"><a name="p9372436142115"></a><a name="p9372436142115"></a><strong id="b1474221153119"><a name="b1474221153119"></a><a name="b1474221153119"></a><code>std::vector&lt;int64_t&gt;&amp; labels</code></strong>: IDs of the top <code>topK</code> nearest vectors to the query. When the number of valid search results is less than <code>topK</code>, the remaining invalid labels are filled with <code>-1</code>.</p>
</td>
</tr>
<tr id="row65385692010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1531756152010"><a name="p1531756152010"></a><a name="p1531756152010"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row1453105616207"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p253155611208"><a name="p253155611208"></a><a name="p253155611208"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul104452596013"></a><a name="ul104452596013"></a><ul id="ul104452596013"><li><code>topK</code> ∈ (0, 4096].</li><li><code>n</code> ∈ (0, 10000].</li><li><code>queryData</code> cannot be empty, and its data length must be greater than or equal to <code>n * dim</code>.</li><li><code>dists</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</li><li><code>labels</code> cannot be empty, and the length of the data it points to must be greater than or equal to <code>n * topK</code>.</li><li>The total amount of data pointed to by <code>mask</code> must be greater than or equal to <code>n * ceil(nTotal / 8)</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetNTotal`<a name="en-us_TOPIC_0000002044829965"></a>

<a name="table971712872115"></a>
<table><tbody><tr id="row11742385218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p474298192115"><a name="p474298192115"></a><a name="p474298192115"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p47421585211"><a name="p47421585211"></a><a name="p47421585211"></a><code>APP_ERROR GetNTotal (uint64_t&amp; nTotal) const;</code></p>
</td>
</tr>
<tr id="row207427862111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1074298172118"><a name="p1074298172118"></a><a name="p1074298172118"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p117428872117"><a name="p117428872117"></a><a name="p117428872117"></a>Gets the number of feature vectors that have been added to the <code>AscendIndexGreat</code> base library.</p>
</td>
</tr>
<tr id="row47428812217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p19742168172110"><a name="p19742168172110"></a><a name="p19742168172110"></a><strong id="b107421685215"><a name="b107421685215"></a><a name="b107421685215"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p174218812216"><a name="p174218812216"></a><a name="p174218812216"></a>None</p>
</td>
</tr>
<tr id="row1574215810219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p2742188142115"><a name="p2742188142115"></a><a name="p2742188142115"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p474213822110"><a name="p474213822110"></a><a name="p474213822110"></a><strong id="b53981719918"><a name="b53981719918"></a><a name="b53981719918"></a><code>uint64_t&amp; nTotal</code></strong>: Number of feature vectors added to the base library.</p>
</td>
</tr>
<tr id="row374210816218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p11742108182113"><a name="p11742108182113"></a><a name="p11742108182113"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row1174319812119"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p27431285217"><a name="p27431285217"></a><a name="p27431285217"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p157432812211"><a name="p157432812211"></a><a name="p157432812211"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `GetDim`<a name="en-us_TOPIC_0000002008751986"></a>

<a name="table113422226216"></a>
<table><tbody><tr id="row336622232116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p336632292112"><a name="p336632292112"></a><a name="p336632292112"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p836612222213"><a name="p836612222213"></a><a name="p836612222213"></a><code>APP_ERROR GetDim(int&amp; dim) const;</code></p>
</td>
</tr>
<tr id="row11366122212211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p123660225216"><a name="p123660225216"></a><a name="p123660225216"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p17366182232119"><a name="p17366182232119"></a><a name="p17366182232119"></a>Gets the dimensionality of the feature vectors added to the <code>AscendIndexGreat</code> base library.</p>
</td>
</tr>
<tr id="row13366172242112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p9366112272118"><a name="p9366112272118"></a><a name="p9366112272118"></a><strong id="b10366142210217"><a name="b10366142210217"></a><a name="b10366142210217"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p16366182210213"><a name="p16366182210213"></a><a name="p16366182210213"></a>None</p>
</td>
</tr>
<tr id="row10366152211214"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p63661422162118"><a name="p63661422162118"></a><a name="p63661422162118"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p18366422142117"><a name="p18366422142117"></a><a name="p18366422142117"></a><strong id="b736392718117"><a name="b736392718117"></a><a name="b736392718117"></a><code>int&amp; dim</code></strong>: Dimensionality of the feature vectors added to the base library.</p>
</td>
</tr>
<tr id="row9366182282110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12366122213212"><a name="p12366122213212"></a><a name="p12366122213212"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row14367522172117"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p536720226216"><a name="p536720226216"></a><a name="p536720226216"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p5367922162119"><a name="p5367922162119"></a><a name="p5367922162119"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Reset`<a name="en-us_TOPIC_0000002008910278"></a>

<a name="table1974793512118"></a>
<table><tbody><tr id="row5768235152116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1768183515214"><a name="p1768183515214"></a><a name="p1768183515214"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1676816355218"><a name="p1676816355218"></a><a name="p1676816355218"></a><code>APP_ERROR Reset();</code></p>
</td>
</tr>
<tr id="row1576843562115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p11768335132110"><a name="p11768335132110"></a><a name="p11768335132110"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p776983510216"><a name="p776983510216"></a><a name="p776983510216"></a>Clears the data stored in this <code>Index</code>, including compressed, dimension-reduced feature vectors and codebook data, while retaining the parameters entered when the user initialized the index.</p>
</td>
</tr>
<tr id="row776916357212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17769235112111"><a name="p17769235112111"></a><a name="p17769235112111"></a><strong id="b7769123532119"><a name="b7769123532119"></a><a name="b7769123532119"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p187691735182114"><a name="p187691735182114"></a><a name="p187691735182114"></a>None</p>
</td>
</tr>
<tr id="row157697353212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9769193511213"><a name="p9769193511213"></a><a name="p9769193511213"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8769143514214"><a name="p8769143514214"></a><a name="p8769143514214"></a>None</p>
</td>
</tr>
<tr id="row77691535162118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1676963518218"><a name="p1676963518218"></a><a name="p1676963518218"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row4770735172115"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p12770123572120"><a name="p12770123572120"></a><a name="p12770123572120"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p10770123572119"><a name="p10770123572119"></a><a name="p10770123572119"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `SetHyperSearchParams`<a name="en-us_TOPIC_0000002044950965"></a>

<a name="table1011347192118"></a>
<table><tbody><tr id="row7231478219"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1824047142117"><a name="p1824047142117"></a><a name="p1824047142117"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1124114712115"><a name="p1124114712115"></a><a name="p1124114712115"></a><code>APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams&amp; params);</code></p>
</td>
</tr>
<tr id="row324114712112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p6241447112115"><a name="p6241447112115"></a><a name="p6241447112115"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p924194712114"><a name="p924194712114"></a><a name="p924194712114"></a>Sets the hyperparameters used when searching this <code>Index</code>.</p>
</td>
</tr>
<tr id="row22484711211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p132404719216"><a name="p132404719216"></a><a name="p132404719216"></a><strong id="b1424114782113"><a name="b1424114782113"></a><a name="b1424114782113"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p112464718211"><a name="p112464718211"></a><a name="p112464718211"></a><strong id="b105404714118"><a name="b105404714118"></a><a name="b105404714118"></a><code>const AscendIndexHyperParams&amp; params</code></strong>: Search hyperparameters. For details, see <a href="./13_AscendIndexMixSearchParams.md#ascendindexhyperparams"><code>AscendIndexHyperParams</code></a>.</p>
</td>
</tr>
<tr id="row1824174715218"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1241647102111"><a name="p1241647102111"></a><a name="p1241647102111"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p16248474211"><a name="p16248474211"></a><a name="p16248474211"></a>None</p>
</td>
</tr>
<tr id="row1424647162118"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p324194716213"><a name="p324194716213"></a><a name="p324194716213"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row12484710212"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p72464711212"><a name="p72464711212"></a><a name="p72464711212"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p4241047102110"><a name="p4241047102110"></a><a name="p4241047102110"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `GetHyperSearchParams`<a name="en-us_TOPIC_0000002400547905"></a>

<a name="table749915518225"></a>
<table><tbody><tr id="row18522851227"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p15221751229"><a name="p15221751229"></a><a name="p15221751229"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2052235122213"><a name="p2052235122213"></a><a name="p2052235122213"></a><code>APP_ERROR GetHyperSearchParams(AscendIndexHyperParams&amp; params) const;</code></p>
</td>
</tr>
<tr id="row1752219552217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1852235192213"><a name="p1852235192213"></a><a name="p1852235192213"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1752295162213"><a name="p1752295162213"></a><a name="p1752295162213"></a>Gets the search hyperparameters used when searching this <code>Index</code>.</p>
</td>
</tr>
<tr id="row15522451223"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p45228522213"><a name="p45228522213"></a><a name="p45228522213"></a><strong id="b1052285172215"><a name="b1052285172215"></a><a name="b1052285172215"></a>Input</strong></p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p352213552213"><a name="p352213552213"></a><a name="p352213552213"></a>None</p>
</td>
</tr>
<tr id="row352210511229"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p14522155102216"><a name="p14522155102216"></a><a name="p14522155102216"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p141731750113612"><a name="p141731750113612"></a><a name="p141731750113612"></a><strong id="b17417143113299"><a name="b17417143113299"></a><a name="b17417143113299"></a><code>AscendIndexHyperParams&amp; params</code></strong>: Search hyperparameters. For details, see <a href="./13_AscendIndexMixSearchParams.md#ascendindexhyperparams"><code>AscendIndexHyperParams</code></a>.</p>
</td>
</tr>
<tr id="row1252219592211"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p185221754223"><a name="p185221754223"></a><a name="p185221754223"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1024995915361"><a name="p1024995915361"></a><a name="p1024995915361"></a><strong id="b324975910365"><a name="b324975910365"></a><a name="b324975910365"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the API return value reference.</p>
</td>
</tr>
<tr id="row1652316522217"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1752318572219"><a name="p1752318572219"></a><a name="p1752318572219"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1252316522219"><a name="p1252316522219"></a><a name="p1252316522219"></a>None</p>
</td>
</tr>
</tbody>
</table>
