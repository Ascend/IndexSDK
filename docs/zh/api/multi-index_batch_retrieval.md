
## 多Index批量检索<a name="ZH-CN_TOPIC_0000001456535132"></a>

在检索距离（distances）值相同的情况下，相较于使用单Index检索功能，多Index批量检索使用的TopK排序算法不同，最终呈现的结果标签会存在一些差异，导致返回的TopK值的标签（label）存在差异。

### Search（AscendIndex）接口<a name="ZH-CN_TOPIC_0000001456854904"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a>void Search(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k,float *distances, idx_t *labels, bool merged);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p541145901013"><a name="p541145901013"></a><a name="p541145901013"></a>实现从多个AscendIndex库执行特征向量查询的接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname5172171112117"><a name="parmname5172171112117"></a><a name="parmname5172171112117"></a>“k”</span>条特征的距离及ID。</p>
<p id="p92091230151318"><a name="p92091230151318"></a><a name="p92091230151318"></a>当前支持以下算法：</p>
<a name="ul166141532131310"></a><a name="ul166141532131310"></a><ul id="ul166141532131310"><li>由Index派生而来的子类型AscendIndexSQ（QuantizerType为QT_8bit）。</li><li>由Index派生而来的子类型AscendIndexFlat（FlatIP、FlatL2）。</li><li>由Index派生而来的子类型AscendIndexIVFSP。</li></ul>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><strong id="b111044812599"><a name="b111044812599"></a><a name="b111044812599"></a>std::vector&lt;AscendIndex *&gt; indexes</strong>：待执行检索的多个index。</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><strong id="b247211017590"><a name="b247211017590"></a><a name="b247211017590"></a>idx_t n</strong>：执行检索的query数。</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><strong id="b19861161116594"><a name="b19861161116594"></a><a name="b19861161116594"></a>const float *x</strong>：执行检索的query特征向量。</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><strong id="b2622713165917"><a name="b2622713165917"></a><a name="b2622713165917"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><strong id="b1552816135913"><a name="b1552816135913"></a><a name="b1552816135913"></a>bool merged</strong>：是否要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b761372015919"><a name="b761372015919"></a><a name="b761372015919"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname10120740193414"><a name="parmname10120740193414"></a><a name="parmname10120740193414"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b796314235591"><a name="b796314235591"></a><a name="b796314235591"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname19961143717340"><a name="parmname19961143717340"></a><a name="parmname19961143717340"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li>参与检索的Index必须创建在同一张卡上。</li><li>当前<span class="parmname" id="parmname10134195594613"><a name="parmname10134195594613"></a><a name="parmname10134195594613"></a>“indexes”</span>支持类型参见如下。<a name="ul8694125212459"></a><a name="ul8694125212459"></a><ul id="ul8694125212459"><li><span class="parmname" id="parmname1266010303474"><a name="parmname1266010303474"></a><a name="parmname1266010303474"></a>“indexes”</span>为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。</li><li><span class="parmname" id="parmname44284353479"><a name="parmname44284353479"></a><a name="parmname44284353479"></a>“indexes”</span>为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。</li><li><span class="parmname" id="parmname1249124284716"><a name="parmname1249124284716"></a><a name="parmname1249124284716"></a>“indexes”</span>为AscendIndexFlat的指针并且需满足0 &lt; indexes.size() ≤ 10000。</li></ul>
</li><li>此处<span class="parmname" id="parmname38951254103411"><a name="parmname38951254103411"></a><a name="parmname38951254103411"></a>“n”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname87086562347"><a name="parmname87086562347"></a><a name="parmname87086562347"></a>“k”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname158711421902"><a name="parmname158711421902"></a><a name="parmname158711421902"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b122915596348"><a name="b122915596348"></a><a name="b122915596348"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname3388112914577"><a name="parmname3388112914577"></a><a name="parmname3388112914577"></a>“distances”</span>/<span class="parmname" id="parmname13760133918573"><a name="parmname13760133918573"></a><a name="parmname13760133918573"></a>“labels”</span>需要为非空指针，且满足：<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>


### Search（AscendIndexInt8）接口<a name="ZH-CN_TOPIC_0000001533044201"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p62099237551"><a name="p62099237551"></a><a name="p62099237551"></a>void Search(std::vector&lt;AscendIndexInt8 *&gt; indexes, idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, bool merged);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1558519285114"><a name="p1558519285114"></a><a name="p1558519285114"></a>实现从多个AscendIndexInt8库执行特征向量查询的接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname13364103911119"><a name="parmname13364103911119"></a><a name="parmname13364103911119"></a>“k”</span>条特征的距离及ID。</p>
<p id="p17585112881112"><a name="p17585112881112"></a><a name="p17585112881112"></a>当前仅支持由AscendIndexInt8派生而来的子类型AscendIndexInt8Flat。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><strong id="b111044812599"><a name="b111044812599"></a><a name="b111044812599"></a>std::vector&lt;AscendIndexInt8 *&gt; indexes</strong>：待执行检索的多个index。</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><strong id="b247211017590"><a name="b247211017590"></a><a name="b247211017590"></a>idx_t n</strong>：执行检索的query数。</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><strong id="b1312125520570"><a name="b1312125520570"></a><a name="b1312125520570"></a>const int8_t *x</strong>：执行检索的query特征向量。</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><strong id="b2622713165917"><a name="b2622713165917"></a><a name="b2622713165917"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><strong id="b1552816135913"><a name="b1552816135913"></a><a name="b1552816135913"></a>bool merged</strong>：是否要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b761372015919"><a name="b761372015919"></a><a name="b761372015919"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname10120740193414"><a name="parmname10120740193414"></a><a name="parmname10120740193414"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b796314235591"><a name="b796314235591"></a><a name="b796314235591"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname19961143717340"><a name="parmname19961143717340"></a><a name="parmname19961143717340"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4866551111619"></a><a name="ul4866551111619"></a><ul id="ul4866551111619"><li>参与检索的Index必须创建在同一张卡上。</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li>当前<span class="parmname" id="parmname1625020411592"><a name="parmname1625020411592"></a><a name="parmname1625020411592"></a>“indexes”</span>仅支持类型为AscendIndexInt8并且需满足<strong id="b1986155193315"><a name="b1986155193315"></a><a name="b1986155193315"></a>0 &lt; indexes.size() ≤ 10000</strong>。</li><li>此处<span class="parmname" id="parmname38951254103411"><a name="parmname38951254103411"></a><a name="parmname38951254103411"></a>“n”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname87086562347"><a name="parmname87086562347"></a><a name="parmname87086562347"></a>“k”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname158711421902"><a name="parmname158711421902"></a><a name="parmname158711421902"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b122915596348"><a name="b122915596348"></a><a name="b122915596348"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname3388112914577"><a name="parmname3388112914577"></a><a name="parmname3388112914577"></a>“distances”</span>/<span class="parmname" id="parmname13760133918573"><a name="parmname13760133918573"></a><a name="parmname13760133918573"></a>“labels”</span>需要为非空指针，且满足：<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>


### Search（FaissIndex）接口<a name="ZH-CN_TOPIC_0000001506334841"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a>void Search(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k,float *distances, idx_t *labels, bool merged);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p412313151969"><a name="p412313151969"></a><a name="p412313151969"></a>实现从多个Index库执行特征向量查询的接口，根据输入的特征向量返回最相似的<span class="parmname" id="parmname1978413251762"><a name="parmname1978413251762"></a><a name="parmname1978413251762"></a>“k”</span>条特征的距离及ID。</p>
<p id="p92091230151318"><a name="p92091230151318"></a><a name="p92091230151318"></a>当前支持以下算法：</p>
<a name="ul166141532131310"></a><a name="ul166141532131310"></a><ul id="ul166141532131310"><li>由Index派生而来的子类型AscendIndexSQ（QuantizerType为QT_8bit）。</li><li>由Index派生而来的子类型AscendIndexFlat（FlatIP、FlatL2）。</li><li>由Index派生而来的子类型AscendIndexIVFSP。</li></ul>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><strong id="b1787190205619"><a name="b1787190205619"></a><a name="b1787190205619"></a>std::vector&lt;Index *&gt; indexes</strong>：待执行检索的多个index。</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><strong id="b331114345618"><a name="b331114345618"></a><a name="b331114345618"></a>idx_t n</strong>：执行检索的query数。</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><strong id="b4951134155610"><a name="b4951134155610"></a><a name="b4951134155610"></a>const float *x</strong>：执行检索的query特征向量。</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><strong id="b739118955619"><a name="b739118955619"></a><a name="b739118955619"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><strong id="b524011115619"><a name="b524011115619"></a><a name="b524011115619"></a>bool merged</strong>：是否要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b1287210155563"><a name="b1287210155563"></a><a name="b1287210155563"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname2102195415326"><a name="parmname2102195415326"></a><a name="parmname2102195415326"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b6295141817564"><a name="b6295141817564"></a><a name="b6295141817564"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname1991145583212"><a name="parmname1991145583212"></a><a name="parmname1991145583212"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1876412513567"></a><a name="ul1876412513567"></a><ul id="ul1876412513567"><li>参与检索的Index必须创建在同一张卡上。</li><li>当前<span class="parmname" id="parmname10134195594613"><a name="parmname10134195594613"></a><a name="parmname10134195594613"></a>“indexes”</span>支持类型参见如下。<a name="ul8694125212459"></a><a name="ul8694125212459"></a><ul id="ul8694125212459"><li><span class="parmname" id="parmname1266010303474"><a name="parmname1266010303474"></a><a name="parmname1266010303474"></a>“indexes”</span>为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。</li><li><span class="parmname" id="parmname44284353479"><a name="parmname44284353479"></a><a name="parmname44284353479"></a>“indexes”</span>为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。</li><li><span class="parmname" id="parmname1249124284716"><a name="parmname1249124284716"></a><a name="parmname1249124284716"></a>“indexes”</span>为AscendIndexFlat的指针并且需满足0 &lt; indexes.size() ≤ 10000。</li></ul>
</li><li>此处<span class="parmname" id="parmname1380710581325"><a name="parmname1380710581325"></a><a name="parmname1380710581325"></a>“n”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname1758118083311"><a name="parmname1758118083311"></a><a name="parmname1758118083311"></a>“k”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname1277510331562"><a name="parmname1277510331562"></a><a name="parmname1277510331562"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b101220212337"><a name="b101220212337"></a><a name="b101220212337"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname3388112914577"><a name="parmname3388112914577"></a><a name="parmname3388112914577"></a>“distances”</span>/<span class="parmname" id="parmname13760133918573"><a name="parmname13760133918573"></a><a name="parmname13760133918573"></a>“labels”</span>需要为非空指针，且满足：<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>


### SearchWithFilter（FaissIndex单filter）接口<a name="ZH-CN_TOPIC_0000001521615937"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2063619251347"><a name="p2063619251347"></a><a name="p2063619251347"></a>void SearchWithFilter(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters, bool merged);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>多<span class="parmname" id="parmname35862155285"><a name="parmname35862155285"></a><a name="parmname35862155285"></a>“indexes”</span>执行检索，根据输入的特征向量返回最相似的<strong id="b11481201817286"><a name="b11481201817286"></a><a name="b11481201817286"></a>k</strong>条特征的ID。提供基于CID过滤的功能，<span class="parmname" id="parmname13652164425416"><a name="parmname13652164425416"></a><a name="parmname13652164425416"></a>“filters”</span>为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><strong id="b815784518279"><a name="b815784518279"></a><a name="b815784518279"></a>std::vector&lt;Index *&gt; indexes</strong>：待执行检索的多个index。</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><strong id="b12157345152710"><a name="b12157345152710"></a><a name="b12157345152710"></a>idx_t n</strong>：执行检索的query数。</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><strong id="b51573455275"><a name="b51573455275"></a><a name="b51573455275"></a>const float *x</strong>：执行检索的query特征向量。</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><strong id="b615718452274"><a name="b615718452274"></a><a name="b615718452274"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p425111215144"><a name="p425111215144"></a><a name="p425111215144"></a><strong id="b215744519278"><a name="b215744519278"></a><a name="b215744519278"></a>const void *filters</strong>：过滤条件。</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><strong id="b815734516277"><a name="b815734516277"></a><a name="b815734516277"></a>bool merged</strong>：是否要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b1515784572714"><a name="b1515784572714"></a><a name="b1515784572714"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname015834542715"><a name="parmname015834542715"></a><a name="parmname015834542715"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b115824511275"><a name="b115824511275"></a><a name="b115824511275"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname215814572713"><a name="parmname215814572713"></a><a name="parmname215814572713"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul151271598168"></a><a name="ul151271598168"></a><ul id="ul151271598168"><li>参与检索的Index必须创建在同一张卡上。</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li>当前<span class="parmname" id="parmname115834592717"><a name="parmname115834592717"></a><a name="parmname115834592717"></a>“indexes”</span>支持类型参见如下。<a name="ul153841243520"></a><a name="ul153841243520"></a><ul id="ul153841243520"><li><span class="parmname" id="parmname1583220233"><a name="parmname1583220233"></a><a name="parmname1583220233"></a>“indexes”</span>为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足<strong id="b1986155193315"><a name="b1986155193315"></a><a name="b1986155193315"></a>0 &lt; indexes.size() ≤ 10000</strong>。</li><li><span class="parmname" id="parmname19133151310317"><a name="parmname19133151310317"></a><a name="parmname19133151310317"></a>“indexes”</span>为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足<strong id="b12475133415315"><a name="b12475133415315"></a><a name="b12475133415315"></a>0 &lt; indexes.size() ≤ 10000</strong>。</li></ul>
</li><li>此处<span class="parmname" id="parmname515920459271"><a name="parmname515920459271"></a><a name="parmname515920459271"></a>“n”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname515964502715"><a name="parmname515964502715"></a><a name="parmname515964502715"></a>“k”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname51591645172717"><a name="parmname51591645172717"></a><a name="parmname51591645172717"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b201606457274"><a name="b201606457274"></a><a name="b201606457274"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname3388112914577"><a name="parmname3388112914577"></a><a name="parmname3388112914577"></a>“distances”</span>/<span class="parmname" id="parmname13760133918573"><a name="parmname13760133918573"></a><a name="parmname13760133918573"></a>“labels”</span>需要为非空指针，且满足：<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</li><li><span class="parmname" id="parmname146781148101815"><a name="parmname146781148101815"></a><a name="parmname146781148101815"></a>“filters”</span>需要为长度为n * 6的uint32_t的数组，否则可能出现越界读的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


### SearchWithFilter（AscendIndex单filter）接口<a name="ZH-CN_TOPIC_0000001521894949"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2063619251347"><a name="p2063619251347"></a><a name="p2063619251347"></a>void SearchWithFilter(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters, bool merged);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>多<span class="parmname" id="parmname17986206203016"><a name="parmname17986206203016"></a><a name="parmname17986206203016"></a>“indexes”</span>执行检索，根据输入的特征向量返回最相似的<strong id="b1083119993014"><a name="b1083119993014"></a><a name="b1083119993014"></a>k</strong>条特征的ID。提供基于CID过滤的功能，<span class="parmname" id="parmname13413121913527"><a name="parmname13413121913527"></a><a name="parmname13413121913527"></a>“filters”</span>为长度为n * 6的uint32_t数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><strong id="b111044812599"><a name="b111044812599"></a><a name="b111044812599"></a>std::vector&lt;AscendIndex *&gt; indexes</strong>：待执行检索的多个index。</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><strong id="b247211017590"><a name="b247211017590"></a><a name="b247211017590"></a>idx_t n</strong>：执行检索的query数。</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><strong id="b19861161116594"><a name="b19861161116594"></a><a name="b19861161116594"></a>const float *x</strong>：执行检索的query特征向量。</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><strong id="b2622713165917"><a name="b2622713165917"></a><a name="b2622713165917"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p425111215144"><a name="p425111215144"></a><a name="p425111215144"></a><strong id="b12391113061419"><a name="b12391113061419"></a><a name="b12391113061419"></a>const void *filters</strong>：过滤条件。</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><strong id="b1552816135913"><a name="b1552816135913"></a><a name="b1552816135913"></a>bool merged</strong>：是否要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b761372015919"><a name="b761372015919"></a><a name="b761372015919"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname10120740193414"><a name="parmname10120740193414"></a><a name="parmname10120740193414"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b796314235591"><a name="b796314235591"></a><a name="b796314235591"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname19961143717340"><a name="parmname19961143717340"></a><a name="parmname19961143717340"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul47662251713"></a><a name="ul47662251713"></a><ul id="ul47662251713"><li>参与检索的Index必须创建在同一张卡上。</li><li>当前<span class="parmname" id="parmname115834592717"><a name="parmname115834592717"></a><a name="parmname115834592717"></a>“indexes”</span>支持类型参见如下。<a name="ul153841243520"></a><a name="ul153841243520"></a><ul id="ul153841243520"><li><span class="parmname" id="parmname1583220233"><a name="parmname1583220233"></a><a name="parmname1583220233"></a>“indexes”</span>为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足<strong id="b1986155193315"><a name="b1986155193315"></a><a name="b1986155193315"></a>0 &lt; indexes.size() ≤ 10000</strong>。</li><li><span class="parmname" id="parmname19133151310317"><a name="parmname19133151310317"></a><a name="parmname19133151310317"></a>“indexes”</span>为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足<strong id="b12475133415315"><a name="b12475133415315"></a><a name="b12475133415315"></a>0 &lt; indexes.size() ≤ 10000</strong>。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。</li></ul>
</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li>此处<span class="parmname" id="parmname38951254103411"><a name="parmname38951254103411"></a><a name="parmname38951254103411"></a>“n”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname87086562347"><a name="parmname87086562347"></a><a name="parmname87086562347"></a>“k”</span>不超过1024。</li><li>此处<span class="parmname" id="parmname158711421902"><a name="parmname158711421902"></a><a name="parmname158711421902"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b122915596348"><a name="b122915596348"></a><a name="b122915596348"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname3388112914577"><a name="parmname3388112914577"></a><a name="parmname3388112914577"></a>“distances”</span>/<span class="parmname" id="parmname13760133918573"><a name="parmname13760133918573"></a><a name="parmname13760133918573"></a>“labels”</span>需要为非空指针，且满足：<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</li><li><span class="parmname" id="parmname146781148101815"><a name="parmname146781148101815"></a><a name="parmname146781148101815"></a>“filters”</span>需要为长度为n * 6的uint32_t的数组，否则可能出现越界读的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


### SearchWithFilter（FaissIndex多filter）接口<a name="ZH-CN_TOPIC_0000001635576093"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2063619251347"><a name="p2063619251347"></a><a name="p2063619251347"></a>void SearchWithFilter(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, void *filters[], bool merged);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3393022191317"><a name="p3393022191317"></a><a name="p3393022191317"></a>多<span class="parmname" id="parmname35862155285"><a name="parmname35862155285"></a><a name="parmname35862155285"></a>“indexes”</span>执行检索，根据输入的特征向量返回最相似的<span class="parmname" id="parmname741519541688"><a name="parmname741519541688"></a><a name="parmname741519541688"></a>“k”</span>条特征的ID。</p>
<p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>提供基于CID过滤的功能，<span class="parmname" id="parmname13642142815819"><a name="parmname13642142815819"></a><a name="parmname13642142815819"></a>“filters”</span>为大小为“n”的指针数组，<span class="parmname" id="parmname298238115819"><a name="parmname298238115819"></a><a name="parmname298238115819"></a>“filters”</span>数组中的每个指针指向indexes.size() * 6 个uint32_t的数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><strong id="b815784518279"><a name="b815784518279"></a><a name="b815784518279"></a>std::vector&lt;Index *&gt; indexes</strong>：待执行检索的多个index。</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><strong id="b12157345152710"><a name="b12157345152710"></a><a name="b12157345152710"></a>idx_t n</strong>：执行检索的query数。</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><strong id="b51573455275"><a name="b51573455275"></a><a name="b51573455275"></a>const float *x</strong>：执行检索的query特征向量。</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><strong id="b615718452274"><a name="b615718452274"></a><a name="b615718452274"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p425111215144"><a name="p425111215144"></a><a name="p425111215144"></a><strong id="b215744519278"><a name="b215744519278"></a><a name="b215744519278"></a>void *filters[]</strong>：过滤条件。</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><strong id="b815734516277"><a name="b815734516277"></a><a name="b815734516277"></a>bool merged</strong>：是否要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><strong id="b1515784572714"><a name="b1515784572714"></a><a name="b1515784572714"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname015834542715"><a name="parmname015834542715"></a><a name="parmname015834542715"></a>“k”</span>个向量间的距离值。</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><strong id="b115824511275"><a name="b115824511275"></a><a name="b115824511275"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname215814572713"><a name="parmname215814572713"></a><a name="parmname215814572713"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul151271598168"></a><a name="ul151271598168"></a><ul id="ul151271598168"><li>参与检索的Index必须创建在同一张卡上。</li></ul>
<a name="ul730213891116"></a><a name="ul730213891116"></a><ul id="ul730213891116"><li>当前“indexes”支持类型参见如下。<a name="ul11318141395713"></a><a name="ul11318141395713"></a><ul id="ul11318141395713"><li><span class="parmname" id="parmname199056549569"><a name="parmname199056549569"></a><a name="parmname199056549569"></a>“indexes”</span>为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。</li><li><span class="parmname" id="parmname1586015018577"><a name="parmname1586015018577"></a><a name="parmname1586015018577"></a>“indexes”</span>为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。</li></ul>
</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li><span class="parmname" id="parmname515920459271"><a name="parmname515920459271"></a><a name="parmname515920459271"></a>“n”</span>不超过1024。</li><li><span class="parmname" id="parmname515964502715"><a name="parmname515964502715"></a><a name="parmname515964502715"></a>“k”</span>不超过1024。</li><li><span class="parmname" id="parmname51591645172717"><a name="parmname51591645172717"></a><a name="parmname51591645172717"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b201606457274"><a name="b201606457274"></a><a name="b201606457274"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname3388112914577"><a name="parmname3388112914577"></a><a name="parmname3388112914577"></a>“distances”</span>/<span class="parmname" id="parmname13760133918573"><a name="parmname13760133918573"></a><a name="parmname13760133918573"></a>“labels”</span>需要为非空指针，且满足：<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</li><li><span class="parmname" id="parmname198471438101916"><a name="parmname198471438101916"></a><a name="parmname198471438101916"></a>“filters”</span>需要为长度为n的指针数组，且数组中每个指针指向长度为indexes.size() * 6的uint32_t的数组，否则可能出现越界读的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


### SearchWithFilter（AscendIndex多filter）接口<a name="ZH-CN_TOPIC_0000001635815493"></a>

<a name="table20177631161415"></a>
<table><tbody><tr id="row141771631111420"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1017733121417"><a name="p1017733121417"></a><a name="p1017733121417"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1817723161412"><a name="p1817723161412"></a><a name="p1817723161412"></a>void SearchWithFilter(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, void *filters[], bool merged);</p>
</td>
</tr>
<tr id="row141772313143"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p517753151420"><a name="p517753151420"></a><a name="p517753151420"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p734718314137"><a name="p734718314137"></a><a name="p734718314137"></a>多<span class="parmname" id="parmname35862155285"><a name="parmname35862155285"></a><a name="parmname35862155285"></a>“indexes”</span>执行检索，根据输入的特征向量返回最相似的<span class="parmname" id="parmname124272571248"><a name="parmname124272571248"></a><a name="parmname124272571248"></a>“k”</span>条特征的ID。</p>
<p id="p91771731111417"><a name="p91771731111417"></a><a name="p91771731111417"></a>提供基于CID过滤的功能，<span class="parmname" id="parmname13642142815819"><a name="parmname13642142815819"></a><a name="parmname13642142815819"></a>“filters”</span>为大小为“n”的指针数组，<span class="parmname" id="parmname298238115819"><a name="parmname298238115819"></a><a name="parmname298238115819"></a>“filters”</span>数组中的每个指针指向indexes.size() * 6 个uint32_t的数组，每6个uint32_t数值为一个filter。每个filter的前4个数字（128bit）表示对应的CID，后2个数字表示对应的时间戳左闭合的范围，即[x, y)。</p>
</td>
</tr>
<tr id="row121771315147"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p141779311146"><a name="p141779311146"></a><a name="p141779311146"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19177173111418"><a name="p19177173111418"></a><a name="p19177173111418"></a><strong id="b815784518279"><a name="b815784518279"></a><a name="b815784518279"></a>std::vector&lt;Index *&gt; indexes</strong>：待执行检索的多个index。</p>
<p id="p141771319144"><a name="p141771319144"></a><a name="p141771319144"></a><strong id="b12157345152710"><a name="b12157345152710"></a><a name="b12157345152710"></a>idx_t n</strong>：执行检索的query数。</p>
<p id="p1217711314143"><a name="p1217711314143"></a><a name="p1217711314143"></a><strong id="b51573455275"><a name="b51573455275"></a><a name="b51573455275"></a>const float *x</strong>：执行检索的query特征向量。</p>
<p id="p0177133112144"><a name="p0177133112144"></a><a name="p0177133112144"></a><strong id="b615718452274"><a name="b615718452274"></a><a name="b615718452274"></a>idx_t k</strong>：需要返回的最相似的结果个数。</p>
<p id="p1417863120145"><a name="p1417863120145"></a><a name="p1417863120145"></a><strong id="b215744519278"><a name="b215744519278"></a><a name="b215744519278"></a>void *filters[]</strong>：过滤条件。</p>
<p id="p1217823110149"><a name="p1217823110149"></a><a name="p1217823110149"></a><strong id="b815734516277"><a name="b815734516277"></a><a name="b815734516277"></a>bool merged</strong>：是否要合并多个Index上执行检索的结果。</p>
</td>
</tr>
<tr id="row19178531191410"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9178113114145"><a name="p9178113114145"></a><a name="p9178113114145"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p717803161415"><a name="p717803161415"></a><a name="p717803161415"></a><strong id="b1515784572714"><a name="b1515784572714"></a><a name="b1515784572714"></a>float *distances</strong>：查询向量与距离最近的前<span class="parmname" id="parmname015834542715"><a name="parmname015834542715"></a><a name="parmname015834542715"></a>“k”</span>个向量间的距离值。</p>
<p id="p10178133101412"><a name="p10178133101412"></a><a name="p10178133101412"></a><strong id="b115824511275"><a name="b115824511275"></a><a name="b115824511275"></a>idx_t *labels</strong>：查询的距离最近的前<span class="parmname" id="parmname215814572713"><a name="parmname215814572713"></a><a name="parmname215814572713"></a>“k”</span>个向量的ID。</p>
</td>
</tr>
<tr id="row1217833112143"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p131788313148"><a name="p131788313148"></a><a name="p131788313148"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1417819317143"><a name="p1417819317143"></a><a name="p1417819317143"></a>无</p>
</td>
</tr>
<tr id="row3178133111141"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p217843111416"><a name="p217843111416"></a><a name="p217843111416"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul151271598168"></a><a name="ul151271598168"></a><ul id="ul151271598168"><li>参与检索的Index必须创建在同一张卡上。</li></ul>
<a name="ul671038151117"></a><a name="ul671038151117"></a><ul id="ul671038151117"><li>当前“indexes”支持类型参见如下。<a name="ul11318141395713"></a><a name="ul11318141395713"></a><ul id="ul11318141395713"><li><span class="parmname" id="parmname199056549569"><a name="parmname199056549569"></a><a name="parmname199056549569"></a>“indexes”</span>为AscendIndexSQ的指针且QuantizerType为QT_8bit，并且需满足0 &lt; indexes.size() ≤ 10000。</li><li><span class="parmname" id="parmname1586015018577"><a name="parmname1586015018577"></a><a name="parmname1586015018577"></a>“indexes”</span>为AscendIndexIVFSP的指针且对应的QuantizerType为QT_8bit、MetricType为METRIC_L2，并且需满足0 &lt; indexes.size() ≤ 10000。参与检索的AscendIndexIVFSP类型index必须共享内存地址上的同一个码本，可以通过AscendIndexIVFSP提供的共享码本构造函数或loadAllData接口创建实例。</li></ul>
</li></ul>
<a name="ul51781331161411"></a><a name="ul51781331161411"></a><ul id="ul51781331161411"><li><span class="parmname" id="parmname515920459271"><a name="parmname515920459271"></a><a name="parmname515920459271"></a>“n”</span>不超过1024。</li><li><span class="parmname" id="parmname515964502715"><a name="parmname515964502715"></a><a name="parmname515964502715"></a>“k”</span>不超过1024。</li><li><span class="parmname" id="parmname51591645172717"><a name="parmname51591645172717"></a><a name="parmname51591645172717"></a>“x”</span>需要为非空指针，且长度应该为dim * <strong id="b201606457274"><a name="b201606457274"></a><a name="b201606457274"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li><li><span class="parmname" id="parmname3388112914577"><a name="parmname3388112914577"></a><a name="parmname3388112914577"></a>“distances”</span>/<span class="parmname" id="parmname13760133918573"><a name="parmname13760133918573"></a><a name="parmname13760133918573"></a>“labels”</span>需要为非空指针，且满足：<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>当merged = true，长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。</li><li>当merged = false，长度应该为indexes.size() * k * n，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</li><li><span class="parmname" id="parmname198471438101916"><a name="parmname198471438101916"></a><a name="parmname198471438101916"></a>“filters”</span>需要为长度为n的指针数组，且数组中每个指针指向长度为indexes.size() * 6的uint32_t的数组，否则可能出现越界读的错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>



